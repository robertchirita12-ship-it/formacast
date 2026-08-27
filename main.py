"""
FormaCast backend — auto-updating football model + odds API.

What it does, on a schedule:
  1. Downloads historical results + upcoming fixtures from football-data.co.uk
     (works from a server: no browser CORS, plus retry/backoff for the 429s).
  2. Fits a time-weighted Dixon-Coles model per league on REAL results
     (full-time goals) AND a second model on first-half goals (HTHG/HTAG).
  3. Builds a per-team corner model from real corner counts (HC/AC).
  4. For every upcoming fixture, derives all markets with model probabilities,
     fair odds, real bookmaker odds where available, and the edge between them.
  5. Serves everything as JSON + serves the static frontend.

Run locally:   uvicorn main:app --host 0.0.0.0 --port 8000
Deps:          see requirements.txt
"""
import math, time, io, csv, threading, hashlib, os
from datetime import datetime, date
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import kickoffs
import journal
import ledger
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE = "https://www.football-data.co.uk"
# div code -> (nice name, [season codes, newest last]).  Season "2526" = 2025/26.
LEAGUES = {
    "E0":  ("Anglia · Premier League",   ["2425", "2526"]),
    "E1":  ("Anglia · Championship",      ["2425", "2526"]),
    "SP1": ("Spania · La Liga",          ["2425", "2526"]),
    "SP2": ("Spania · La Liga 2",        ["2425", "2526"]),
    "I1":  ("Italia · Serie A",          ["2425", "2526"]),
    "I2":  ("Italia · Serie B",          ["2425", "2526"]),
    "D1":  ("Germania · Bundesliga",     ["2425", "2526"]),
    "D2":  ("Germania · Bundesliga 2",   ["2425", "2526"]),
    "F1":  ("Franța · Ligue 1",          ["2425", "2526"]),
    "F2":  ("Franța · Ligue 2",          ["2425", "2526"]),
    "N1":  ("Olanda · Eredivisie",       ["2425", "2526"]),
    "P1":  ("Portugalia · Primeira Liga",["2425", "2526"]),
    "T1":  ("Turcia · Super Lig",        ["2425", "2526"]),
    "B1":  ("Belgia · Jupiler Pro League",["2425", "2526"]),
    "G1":  ("Grecia · Super League",     ["2425", "2526"]),
    "SC0": ("Scoția · Premiership",      ["2425", "2526"]),
    "SC1": ("Scoția · Championship",     ["2425", "2526"]),
}
HALF_LIFE_DAYS = 180      # time-decay for form weighting
FH_SHARE_FALLBACK = 0.45  # only if a league has no half-time data
MARGIN = 0.06             # margin added to fair odds for markets without real odds
REFRESH_HOURS = 6
RHO = -0.08
MAXG = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (FormaCast data fetcher)"}

CACHE: Dict = {"updated_at": None, "leagues": [], "fixtures": [], "error": None}
HIST: Dict = {}        # div -> raw historical matches (reused by backtest)
BACKTESTS: Dict = {}   # div -> {status, metrics...}
LOCK = threading.Lock()

# ----------------------------------------------------------------------------
# FETCH (with retry/backoff for 429)
# ----------------------------------------------------------------------------
def fetch_text(url: str, tries: int = 5) -> Optional[str]:
    delay = 3
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.content.decode("latin-1", errors="ignore")
            if r.status_code == 429:
                time.sleep(delay); delay *= 2; continue
            return None
        except requests.RequestException:
            time.sleep(delay); delay *= 2
    return None

# ----------------------------------------------------------------------------
# PARSE
# ----------------------------------------------------------------------------
def num(v):
    try:
        return float(str(v).replace(",", "."))
    except (ValueError, TypeError):
        return None

def parse_date(s):
    if not s:
        return None
    p = str(s).strip().split("/")
    if len(p) < 3:
        return None
    try:
        d, m, y = int(p[0]), int(p[1]), int(p[2])
        if y < 100:
            y += 2000
        return date(y, m, d)
    except ValueError:
        return None

def get(row, *keys):
    for k in keys:
        if k in row and row[k] not in ("", None):
            return row[k]
    return None

def parse_csv(text: str) -> List[dict]:
    out = []
    rdr = csv.DictReader(io.StringIO(text))
    rdr.fieldnames = [(f or "").strip().lstrip("\ufeff") for f in (rdr.fieldnames or [])]
    for row in rdr:
        home, away = get(row, "HomeTeam", "Home"), get(row, "AwayTeam", "Away")
        if not home or not away:
            continue
        fthg, ftag = num(get(row, "FTHG", "HG")), num(get(row, "FTAG", "AG"))
        out.append({
            "div": (get(row, "Div") or "").strip(),
            "date": parse_date(get(row, "Date")),
            "home": home.strip(), "away": away.strip(),
            "fthg": fthg, "ftag": ftag,
            "hthg": num(get(row, "HTHG")), "htag": num(get(row, "HTAG")),
            "hc": num(get(row, "HC")), "ac": num(get(row, "AC")),
            "played": fthg is not None and ftag is not None,
            "oH": num(get(row, "AvgH", "B365H", "PSH")),
            "oD": num(get(row, "AvgD", "B365D", "PSD")),
            "oA": num(get(row, "AvgA", "B365A", "PSA")),
            "oO25": num(get(row, "Avg>2.5", "B365>2.5")),
            "oU25": num(get(row, "Avg<2.5", "B365<2.5")),
        })
    return out

# ----------------------------------------------------------------------------
# DIXON-COLES (time-weighted MLE, gradient ascent) — pure python
# ----------------------------------------------------------------------------
def clip(v, lo, hi): return lo if v < lo else hi if v > hi else v

def fit_dc(games, ref: date, gkey_h="fthg", gkey_a="ftag", half_life=HALF_LIFE_DAYS,
           iters=250, lr=0.9, reg=0.02):
    games = [g for g in games if g.get(gkey_h) is not None and g.get(gkey_a) is not None and g["date"]]
    teams = sorted({t for g in games for t in (g["home"], g["away"])})
    if len(teams) < 2 or len(games) < 20:
        return None
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    att = [0.0] * T; dfc = [0.0] * T; gamma = 0.3
    decay = math.log(2) / half_life
    H = [ti[g["home"]] for g in games]; A = [ti[g["away"]] for g in games]
    GH = [g[gkey_h] for g in games];    GA = [g[gkey_a] for g in games]
    W = [math.exp(-decay * max(0, (ref - g["date"]).days)) for g in games]
    totW = sum(W) or 1.0
    for _ in range(iters):
        gA = [0.0] * T; gD = [0.0] * T; gG = 0.0
        for m in range(len(games)):
            h, a, w = H[m], A[m], W[m]
            lh = math.exp(clip(att[h] - dfc[a] + gamma, -3, 3))
            la = math.exp(clip(att[a] - dfc[h], -3, 3))
            eh, ea = GH[m] - lh, GA[m] - la
            gA[h] += w * eh; gA[a] += w * ea
            gD[a] += w * (-eh); gD[h] += w * (-ea)
            gG += w * eh
        for i in range(T):
            att[i] += lr * (gA[i] / totW - reg * att[i])
            dfc[i] += lr * (gD[i] / totW - reg * dfc[i])
        gamma += lr * (gG / totW)
        ma = sum(att) / T; md = sum(dfc) / T
        for i in range(T):
            att[i] -= ma; dfc[i] -= md
    # estimate rho on low-score games
    best_rho, best_ll = RHO, -1e18
    for k in range(-10, 1):
        rho = k * 0.02; ll = 0.0
        for m in range(len(games)):
            if GH[m] > 1 or GA[m] > 1:
                continue
            lh = math.exp(clip(att[H[m]] - dfc[A[m]] + gamma, -3, 3))
            la = math.exp(clip(att[A[m]] - dfc[H[m]], -3, 3))
            t = tau(int(GH[m]), int(GA[m]), lh, la, rho)
            if t > 0:
                ll += math.log(t)
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return {"ti": ti, "att": att, "dfc": dfc, "gamma": gamma, "rho": best_rho}

def lambdas(model, home, away):
    if not model or home not in model["ti"] or away not in model["ti"]:
        return None
    h, a = model["ti"][home], model["ti"][away]
    lh = math.exp(clip(model["att"][h] - model["dfc"][a] + model["gamma"], -3, 3))
    la = math.exp(clip(model["att"][a] - model["dfc"][h], -3, 3))
    return lh, la

# ----------------------------------------------------------------------------
# POISSON / MARKET MATH
# ----------------------------------------------------------------------------
def pois(k, l): return math.exp(k * math.log(l) - l - math.lgamma(k + 1))
def tau(x, y, l, m, rho):
    if x == 0 and y == 0: return 1 - l * m * rho
    if x == 0 and y == 1: return 1 + l * rho
    if x == 1 and y == 0: return 1 + m * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def matrix(lh, la, rho):
    M = [[0.0] * (MAXG + 1) for _ in range(MAXG + 1)]; s = 0.0
    for x in range(MAXG + 1):
        for y in range(MAXG + 1):
            p = max(0.0, tau(x, y, lh, la, rho) * pois(x, lh) * pois(y, la))
            M[x][y] = p; s += p
    for x in range(MAXG + 1):
        for y in range(MAXG + 1):
            M[x][y] /= s
    return M

def derive(M):
    p1 = px = p2 = btts = hs = as_ = 0.0
    over = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}; scores = []
    for x in range(MAXG + 1):
        for y in range(MAXG + 1):
            p = M[x][y]
            if x > y: p1 += p
            elif x == y: px += p
            else: p2 += p
            if x >= 1 and y >= 1: btts += p
            if x >= 1: hs += p
            if y >= 1: as_ += p
            t = x + y
            for ln in over:
                if t > ln: over[ln] += p
            scores.append((x, y, p))
    scores.sort(key=lambda s: -s[2])
    return {"p1": p1, "px": px, "p2": p2, "btts": btts, "hs": hs, "as": as_,
            "over": over, "scores": scores[:5]}

def pois_over(mu, line):
    ov = 0.0
    for i in range(0, 40):
        if i > line: ov += pois(i, mu)
    return ov

def fair(p): return round(max(1.01, 1 / p), 2) if p > 0.001 else 99.0
def market_odd(p): return round(max(1.01, (1 / p) / (1 + MARGIN)), 2) if p > 0.001 else 99.0
def implied3(oH, oD, oA):
    if not (oH and oD and oA): return None
    r = [1/oH, 1/oD, 1/oA]; s = sum(r)
    return [x/s for x in r]

# ----------------------------------------------------------------------------
# CORNERS (real per-team averages)
# ----------------------------------------------------------------------------
def corner_stats(games):
    cf, ca, n_home, n_away = {}, {}, {}, {}
    for g in games:
        if g["hc"] is None or g["ac"] is None:
            continue
        cf.setdefault(g["home"], [0,0]); cf.setdefault(g["away"], [0,0])
        ca.setdefault(g["home"], [0,0]); ca.setdefault(g["away"], [0,0])
        cf[g["home"]][0] += g["hc"]; cf[g["home"]][1] += 1
        cf[g["away"]][0] += g["ac"]; cf[g["away"]][1] += 1
        ca[g["home"]][0] += g["ac"]; ca[g["home"]][1] += 1
        ca[g["away"]][0] += g["hc"]; ca[g["away"]][1] += 1
    league_avg = None
    tot = [ (g["hc"]+g["ac"]) for g in games if g["hc"] is not None and g["ac"] is not None]
    if tot: league_avg = sum(tot)/len(tot)
    return {"cf": cf, "ca": ca, "league": league_avg}

def expected_corners(cs, home, away):
    if not cs["league"]:
        return None
    def avg(d, t):
        v = d.get(t); return v[0]/v[1] if v and v[1] else None
    hf, aa = avg(cs["cf"], home), avg(cs["ca"], away)   # home corners for, away corners allowed
    af, ha = avg(cs["cf"], away), avg(cs["ca"], home)
    half = cs["league"] / 2
    exp_home = ((hf or half) + (aa or half)) / 2
    exp_away = ((af or half) + (ha or half)) / 2
    return exp_home + exp_away

# ----------------------------------------------------------------------------
# BUILD PREDICTIONS
# ----------------------------------------------------------------------------
def build_markets(ft_model, fh_model, cstats, f):
    lam = lambdas(ft_model, f["home"], f["away"])
    if not lam:
        return None
    lh, la = lam
    ft = derive(matrix(lh, la, ft_model["rho"]))
    mu = lh + la

    # first half from real HT model if available, else fallback share
    if fh_model and lambdas(fh_model, f["home"], f["away"]):
        flh, fla = lambdas(fh_model, f["home"], f["away"])
        fh = derive(matrix(flh, fla, fh_model["rho"])); fh_real = True
    else:
        fh = derive(matrix(lh*FH_SHARE_FALLBACK, la*FH_SHARE_FALLBACK, ft_model["rho"])); fh_real = False

    imp = implied3(f.get("oH"), f.get("oD"), f.get("oA"))

    def pk(label, p, real_odd=None, imp_p=None):
        d = {"label": label, "p": round(p, 4), "fairOdd": fair(p),
             "odd": real_odd if real_odd else market_odd(p), "real": bool(real_odd)}
        if real_odd and imp_p is not None:
            d["edge"] = round(p - imp_p, 4)
        return d

    groups = []
    groups.append({"name": "Rezultat", "picks": [
        pk(f"1 {f['home']}", ft["p1"], f.get("oH"), imp[0] if imp else None),
        pk("X Egal", ft["px"], f.get("oD"), imp[1] if imp else None),
        pk(f"2 {f['away']}", ft["p2"], f.get("oA"), imp[2] if imp else None),
        pk("1X", ft["p1"]+ft["px"]), pk("X2", ft["px"]+ft["p2"]), pk("12", ft["p1"]+ft["p2"]),
    ]})
    goals = [pk("Peste 1.5", ft["over"][1.5]), pk("Sub 1.5", 1-ft["over"][1.5])]
    if f.get("oO25"):
        io_ = implied3(f["oO25"], 1e9, f["oU25"]) if f.get("oU25") else None
        goals.append(pk("Peste 2.5", ft["over"][2.5], f["oO25"]))
        goals.append(pk("Sub 2.5", 1-ft["over"][2.5], f.get("oU25")))
    else:
        goals.append(pk("Peste 2.5", ft["over"][2.5])); goals.append(pk("Sub 2.5", 1-ft["over"][2.5]))
    goals += [pk("Peste 3.5", ft["over"][3.5]), pk("Ambele înscriu", ft["btts"]),
              pk("Nu ambele", 1-ft["btts"]), pk(f"{f['home']} înscrie", ft["hs"]),
              pk(f"{f['away']} înscrie", ft["as"])]
    groups.append({"name": "Goluri", "note": f"total așteptat {mu:.2f}", "picks": goals})

    groups.append({"name": "Prima repriză", "note": "din date reale" if fh_real else "estimat",
                   "picks": [pk("Peste 0.5", fh["over"][0.5]), pk("Sub 0.5", 1-fh["over"][0.5]),
                             pk("Peste 1.5", fh["over"][1.5]), pk("GG repr.1", fh["btts"]),
                             pk(f"1 {f['home']} repr.1", fh["p1"]), pk("X repr.1", fh["px"]),
                             pk(f"2 {f['away']} repr.1", fh["p2"])]})

    exp_c = expected_corners(cstats, f["home"], f["away"]) if cstats else None
    if exp_c:
        groups.append({"name": "Cornere", "note": f"total așteptat {exp_c:.1f} (real)",
                       "picks": [pk("Peste 8.5", pois_over(exp_c, 8.5)),
                                 pk("Peste 9.5", pois_over(exp_c, 9.5)),
                                 pk("Peste 10.5", pois_over(exp_c, 10.5))]})

    return {"lh": round(lh, 2), "la": round(la, 2), "mu": round(mu, 2),
            "scores": [{"x": x, "y": y, "p": round(p, 4)} for x, y, p in ft["scores"]],
            "groups": groups}

# ----------------------------------------------------------------------------
# REFRESH JOB
# ----------------------------------------------------------------------------
def refresh():
    try:
        fx_text = fetch_text(f"{BASE}/fixtures.csv")
        fixtures_all = parse_csv(fx_text) if fx_text else []
        out_leagues, out_fixtures = [], []
        today = date.today()

        for div, (name, seasons) in LEAGUES.items():
            hist = []
            for s in seasons:
                t = fetch_text(f"{BASE}/mmz4281/{s}/{div}.csv")
                if t:
                    hist += [g for g in parse_csv(t) if g["played"]]
                time.sleep(1)  # be polite to the source
            if not hist:
                continue
            ref = max(g["date"] for g in hist if g["date"])
            ft_model = fit_dc(hist, ref, "fthg", "ftag")
            has_ht = any(g["hthg"] is not None for g in hist)
            fh_model = fit_dc(hist, ref, "hthg", "htag") if has_ht else None
            cstats = corner_stats([g for g in hist if g["date"] and (ref - g["date"]).days < 400])
            if not ft_model:
                continue
            HIST[div] = hist  # reused by /api/backtest
            out_leagues.append({"div": div, "league": name})

            for f in fixtures_all:
                if f.get("played"):
                    continue
                # assign each fixture to the right league by its Div code (fallback: team membership)
                if f.get("div"):
                    if f["div"] != div:
                        continue
                elif f["home"] not in ft_model["ti"] or f["away"] not in ft_model["ti"]:
                    continue
                if f["home"] not in ft_model["ti"] or f["away"] not in ft_model["ti"]:
                    continue
                if not f["date"] or f["date"] < today:
                    continue
                mk = build_markets(ft_model, fh_model, cstats, f)
                if not mk:
                    continue
                fid = hashlib.sha1(f"{div}{f['date']}{f['home']}{f['away']}".encode()).hexdigest()[:12]
                out_fixtures.append({
                    "id": fid, "div": div, "league": name,
                    "date": f["date"].isoformat(), "home": f["home"], "away": f["away"],
                    "kickoff": kickoffs.kickoff_for(f["home"], f["away"], f["date"].isoformat()),
                    **mk,
                })

        out_fixtures.sort(key=lambda x: x["date"])
        with LOCK:
            CACHE.update({"updated_at": datetime.utcnow().isoformat() + "Z",
                          "leagues": out_leagues, "fixtures": out_fixtures, "error": None})
        # prediction journal: record new pre-match forecasts, settle any now played
        try:
            journal.record(out_fixtures)
            journal.settle(HIST)
            ledger.auto_settle(HIST)   # close my placed bets that have been played
        except Exception as je:  # never let the journal break a refresh
            print("journal hook error:", je)
    except Exception as e:  # noqa
        with LOCK:
            CACHE["error"] = str(e)

# ----------------------------------------------------------------------------
# BACKTEST (walk-forward on one league's history)
# ----------------------------------------------------------------------------
def load_history(div):
    if div in HIST:
        return HIST[div]
    name, seasons = LEAGUES[div]
    hist = []
    for s in seasons:
        t = fetch_text(f"{BASE}/mmz4281/{s}/{div}.csv")
        if t:
            hist += [g for g in parse_csv(t) if g["played"]]
        time.sleep(1)
    HIST[div] = hist
    return hist

def run_backtest(div, edge_thr=0.05, min_train=40, min_games=4):
    BACKTESTS[div] = {"status": "running", "div": div}
    try:
        hist = [g for g in load_history(div) if g["date"]]
        hist.sort(key=lambda g: g["date"])
        if len(hist) < min_train + 20:
            BACKTESTS[div] = {"status": "error", "div": div, "error": "prea puține meciuri"}
            return
        full = fit_dc(hist, hist[-1]["date"])
        rho = full["rho"] if full else RHO
        gp = {}
        correct = tested = 0
        logloss = brier = 0.0
        stake = profit = bets = won = 0.0
        equity = []
        model = None; last_fit = None
        for idx, g in enumerate(hist):
            enough = gp.get(g["home"], 0) >= min_games and gp.get(g["away"], 0) >= min_games
            if idx >= min_train and enough:
                if last_fit != g["date"]:
                    model = fit_dc(hist[:idx], g["date"], iters=140)
                    last_fit = g["date"]
                lam = lambdas(model, g["home"], g["away"]) if model else None
                if lam:
                    d = derive(matrix(lam[0], lam[1], rho))
                    p = {"H": d["p1"], "D": d["px"], "A": d["p2"]}
                    actual = "H" if g["fthg"] > g["ftag"] else "D" if g["fthg"] == g["ftag"] else "A"
                    pred = max(p, key=p.get)
                    tested += 1
                    if pred == actual:
                        correct += 1
                    logloss += -math.log(max(1e-9, p[actual]))
                    for k in "HDA":
                        brier += (p[k] - (1 if actual == k else 0)) ** 2
                    imp = implied3(g.get("oH"), g.get("oD"), g.get("oA"))
                    if imp:
                        edges = [("H", p["H"] - imp[0], g.get("oH")),
                                 ("D", p["D"] - imp[1], g.get("oD")),
                                 ("A", p["A"] - imp[2], g.get("oA"))]
                        edges.sort(key=lambda e: -e[1])
                        k, e, odd = edges[0]
                        if e >= edge_thr and odd:
                            bets += 1; stake += 1
                            if k == actual:
                                profit += odd - 1; won += 1
                            else:
                                profit -= 1
                            equity.append(round(profit, 2))
            gp[g["home"]] = gp.get(g["home"], 0) + 1
            gp[g["away"]] = gp.get(g["away"], 0) + 1
        BACKTESTS[div] = {
            "status": "done", "div": div, "edge_thr": edge_thr,
            "tested": tested, "correct": correct,
            "acc": round(correct / tested, 4) if tested else 0,
            "logloss": round(logloss / tested, 4) if tested else 0,
            "brier": round(brier / tested, 4) if tested else 0,
            "bets": int(bets), "won": int(won),
            "hitrate": round(won / bets, 4) if bets else 0,
            "profit": round(profit, 2), "roi": round(profit / stake, 4) if stake else 0,
            "equity": equity[-200:],
        }
    except Exception as e:  # noqa
        BACKTESTS[div] = {"status": "error", "div": div, "error": str(e)}

# ----------------------------------------------------------------------------
# KEEP-ALIVE (stop Render free tier from sleeping)
# ----------------------------------------------------------------------------
def keep_alive():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url or os.environ.get("KEEP_ALIVE", "1") == "0":
        return
    while True:
        time.sleep(600)
        try:
            requests.get(url.rstrip("/") + "/api/health", timeout=15)
        except Exception:  # noqa
            pass

# ----------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------
app = FastAPI(title="FormaCast API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    with LOCK:
        return {"updated_at": CACHE["updated_at"], "leagues": len(CACHE["leagues"]),
                "fixtures": len(CACHE["fixtures"]), "error": CACHE["error"]}

@app.get("/api/leagues")
def get_leagues():
    with LOCK:
        return CACHE["leagues"]

def _overlay_kickoffs(fx):
    """Attach the freshest kickoff time at serve-time (not just at refresh-time),
    so kickoffs that finish loading AFTER a fixtures refresh still show up
    immediately on the next request instead of waiting up to REFRESH_HOURS."""
    out = []
    for f in fx:
        f2 = dict(f)
        f2["kickoff"] = kickoffs.kickoff_for(f["home"], f["away"], f["date"]) or f.get("kickoff")
        out.append(f2)
    return out

@app.get("/api/fixtures")
def get_fixtures(div: Optional[str] = Query(None)):
    with LOCK:
        fx = CACHE["fixtures"]
        if div and div != "ALL":
            fx = [f for f in fx if f["div"] == div]
        return {"updated_at": CACHE["updated_at"], "fixtures": _overlay_kickoffs(fx)}

@app.get("/api/backtest")
def get_backtest(div: str, edge: float = 0.05):
    if div not in LEAGUES:
        return JSONResponse({"status": "error", "error": "ligă necunoscută"}, status_code=400)
    bt = BACKTESTS.get(div)
    if bt and bt.get("status") == "done" and abs(bt.get("edge_thr", -1) - edge) < 1e-9:
        return bt
    if not bt or bt.get("status") != "running":
        threading.Thread(target=run_backtest, args=(div, edge), daemon=True).start()
        return {"status": "running", "div": div}
    return bt

# ---- Live radar: pre-computed targets for in-play over/BTTS betting ----
def live_radar():
    """Rank today's fixtures by how likely goals are to flow (good for live 'over')."""
    with LOCK:
        fx = list(CACHE["fixtures"])
    fx = _overlay_kickoffs(fx)
    out = []
    for f in fx:
        goals = next((g for g in f["groups"] if g["name"] == "Goluri"), None)
        if not goals:
            continue
        def pick_p(label):
            for p in goals["picks"]:
                if p["label"] == label:
                    return p["p"]
            return None
        o15 = pick_p("Peste 1.5"); o25 = pick_p("Peste 2.5")
        gg = pick_p("Ambele înscriu")
        mu = f.get("mu")
        if mu is None or o25 is None:
            continue
        # live-over target score: high expected goals + high BTTS + high over2.5
        score = 0.45 * min(mu / 3.5, 1.0) + 0.30 * (o25 or 0) + 0.25 * (gg or 0)
        # "wait until" minute at 0-0: rough guide from expected goals (more goals -> wait less)
        wait_min = int(max(20, min(55, 75 - mu * 14)))
        out.append({
            "id": f["id"], "home": f["home"], "away": f["away"],
            "date": f["date"], "kickoff": f.get("kickoff"), "league": f["league"],
            "lh": f["lh"], "la": f["la"], "mu": mu,
            "o15": o15, "o25": o25, "gg": gg,
            "score": round(score, 4), "wait_min": wait_min,
        })
    out.sort(key=lambda x: -x["score"])
    return out

@app.get("/api/live-radar")
def get_live_radar():
    with LOCK:
        updated = CACHE["updated_at"]
    return {"updated_at": updated, "targets": live_radar()}

@app.get("/api/kickoffs/debug")
def get_kickoffs_debug():
    with kickoffs.KICK_LOCK:
        return {**kickoffs.KICK_STATE, "key_set": bool(kickoffs.AF_KEY), "tz": kickoffs.TZ}

@app.get("/api/euro-fixtures")
def get_euro_fixtures():
    """Champions/Europa/Conference League: schedule only (no xG — see note in kickoffs.py)."""
    return {"fixtures": kickoffs.get_euro_fixtures(), "note": "fără predicție xG — doar programul"}

@app.get("/api/journal")
def get_journal(limit: int = Query(100, ge=1, le=500), settled: bool = False):
    return {"entries": journal.entries(limit=limit, only_settled=settled),
            "backend": journal.STATE["backend"]}

@app.get("/api/journal/stats")
def get_journal_stats():
    return journal.stats()

@app.get("/api/journal/debug")
def get_journal_debug():
    return journal.STATE

# ---- bet ledger (only bets I actually place) ----
@app.get("/api/ledger/summary")
def ledger_summary():
    return ledger.summary()

@app.get("/api/ledger/config")
def ledger_get_config():
    return ledger.get_config()

@app.post("/api/ledger/config")
def ledger_set_config(payload: dict = Body(...)):
    return ledger.set_config(payload or {})

@app.get("/api/bets")
def ledger_bets(limit: int = Query(200, ge=1, le=500), status: str = None):
    return {"bets": ledger.bets(limit=limit, status=status),
            "suggested_stake": ledger.suggest_stake(), "bank": ledger.current_bank()}

@app.post("/api/bets")
def ledger_add_bet(payload: dict = Body(...)):
    bid = ledger.add_bet(payload or {})
    if not bid:
        return JSONResponse({"ok": False, "error": "date invalide (cotă/miză)"}, status_code=400)
    return {"ok": True, "id": bid}

@app.post("/api/bets/{bet_id}/settle")
def ledger_settle_bet(bet_id: str, payload: dict = Body(...)):
    ok = ledger.settle_manual(bet_id, (payload or {}).get("status", ""))
    return JSONResponse({"ok": ok}, status_code=200 if ok else 400)

@app.post("/api/bets/{bet_id}/delete")
def ledger_delete_bet(bet_id: str):
    return {"ok": ledger.delete_bet(bet_id)}

@app.get("/api/ledger/debug")
def ledger_debug():
    return ledger.STATE

# ----------------------------------------------------------------------------
# PWA: manifest, icons (base64-embedded so no binary files to upload), service worker
# ----------------------------------------------------------------------------
import base64 as _b64
from fastapi.responses import Response as _Resp

_ICON_192 = _b64.b64decode("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")
_ICON_512 = _b64.b64decode("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")
_ICON_180 = _b64.b64decode("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")

_MANIFEST = {
    "name": "FormaCast", "short_name": "FormaCast",
    "description": "Predictii fotbal si generator de bilet",
    "start_url": "/", "scope": "/", "display": "standalone",
    "orientation": "portrait",
    "background_color": "#0E1520", "theme_color": "#0E1520",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
    ],
}

_SW = """
const CACHE='formacast-v1';
self.addEventListener('install',e=>{self.skipWaiting();});
self.addEventListener('activate',e=>{e.waitUntil(clients.claim());});
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url);
  // network-first for API, cache-first for shell
  if(u.pathname.startsWith('/api/')){
    e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));
  } else {
    e.respondWith(
      caches.match(e.request).then(r=>r||fetch(e.request).then(resp=>{
        const c=resp.clone(); caches.open(CACHE).then(ch=>ch.put(e.request,c)); return resp;
      }).catch(()=>caches.match('/')))
    );
  }
});
"""

@app.get("/manifest.json")
def _manifest():
    return JSONResponse(_MANIFEST)

@app.get("/icon-192.png")
def _icon192():
    return _Resp(content=_ICON_192, media_type="image/png")

@app.get("/icon-512.png")
def _icon512():
    return _Resp(content=_ICON_512, media_type="image/png")

@app.get("/apple-touch-icon.png")
def _appleicon():
    return _Resp(content=_ICON_180, media_type="image/png")

@app.get("/apple-touch-icon-precomposed.png")
def _appleicon2():
    return _Resp(content=_ICON_180, media_type="image/png")

@app.get("/sw.js")
def _sw():
    return _Resp(content=_SW, media_type="application/javascript")

# serve frontend (index.html placed next to this file)
HERE = os.path.dirname(os.path.abspath(__file__))
@app.get("/")
def index():
    p = os.path.join(HERE, "index.html")
    return FileResponse(p) if os.path.exists(p) else JSONResponse({"msg": "FormaCast API. See /api/fixtures"})

# scheduler
scheduler = BackgroundScheduler(daemon=True)

@app.on_event("startup")
def startup():
    journal.init()  # prediction journal storage
    ledger.init()   # bet ledger storage
    threading.Thread(target=refresh, daemon=True).start()   # first load in background
    threading.Thread(target=keep_alive, daemon=True).start()  # prevent free-tier sleep
    scheduler.add_job(refresh, "interval", hours=REFRESH_HOURS, id="refresh")
    kickoffs.start_kickoff_scheduler(scheduler)  # kickoff times from API-Football
    scheduler.start()

@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown(wait=False)
