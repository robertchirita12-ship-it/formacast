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
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
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
                    **mk,
                })

        out_fixtures.sort(key=lambda x: x["date"])
        with LOCK:
            CACHE.update({"updated_at": datetime.utcnow().isoformat() + "Z",
                          "leagues": out_leagues, "fixtures": out_fixtures, "error": None})
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

@app.get("/api/fixtures")
def get_fixtures(div: Optional[str] = Query(None)):
    with LOCK:
        fx = CACHE["fixtures"]
        if div and div != "ALL":
            fx = [f for f in fx if f["div"] == div]
        return {"updated_at": CACHE["updated_at"], "fixtures": fx}

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
            "date": f["date"], "league": f["league"],
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
    threading.Thread(target=refresh, daemon=True).start()   # first load in background
    threading.Thread(target=keep_alive, daemon=True).start()  # prevent free-tier sleep
    scheduler.add_job(refresh, "interval", hours=REFRESH_HOURS, id="refresh")
    scheduler.start()

@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown(wait=False)
