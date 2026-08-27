"""
FormaCast — bet ledger (Stratul 2: doar pariurile pe care le joc).

Principiul: modelul propune candidați (predictions/journal). AICI rămân doar
pariurile pe care le pun eu efectiv. Măsurăm cinstit, pe volum mare:
  - banca reală (start + P/L închis), miză proporțională (2% din bancă)
  - CLV pe prematch (cota luată vs cota de închidere) = predictorul edge-ului
  - yield, ROI, rată de succes, defalcare prematch/live

Stocare: refolosește conexiunea din journal.py (Postgres/Neon persistent).
"""
import datetime
import uuid
from typing import Dict, List, Optional

import journal  # reuse _connect(), _q(), IS_PG

STATE = {"error": None}

DEFAULTS = {
    "start_bank": "100",     # banca de referință (soldul real de la care pornim)
    "unit_pct": "0.02",      # miză = 2% din banca curentă (proporțional)
    "max_pct": "0.05",       # plafon absolut 5% / pariu
    "style": "proportional",
    "target": "1000",        # obiectiv (doar reper, nu bază de miză)
}

DDL_CONFIG = "CREATE TABLE IF NOT EXISTS ledger_config (k TEXT PRIMARY KEY, v TEXT)"
DDL_BETS = """
CREATE TABLE IF NOT EXISTS bets (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    match_id TEXT, league TEXT, home TEXT, away TEXT, match_date TEXT, kickoff TEXT,
    div TEXT, market TEXT, selection TEXT,
    bet_type TEXT,                       -- 'prematch' | 'live'
    odds REAL, model_prob REAL, stake REAL,
    status TEXT,                         -- 'pending' | 'won' | 'lost' | 'void'
    payout REAL, pl REAL,
    closing_odds REAL, clv_pct REAL,
    bank_before REAL, settled_at TEXT, notes TEXT
)
"""


def init():
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute(DDL_CONFIG)
        cur.execute(DDL_BETS)
        # seed defaults if absent
        for k, v in DEFAULTS.items():
            cur.execute(journal._q(
                "INSERT INTO ledger_config (k,v) VALUES (?,?) ON CONFLICT (k) DO NOTHING"), (k, v))
        conn.commit(); conn.close()
        STATE["error"] = None
    except Exception as e:
        STATE["error"] = f"init: {e}"; print("ledger init error:", e)


def get_config() -> Dict[str, str]:
    cfg = dict(DEFAULTS)
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute("SELECT k,v FROM ledger_config")
        for k, v in cur.fetchall():
            cfg[k] = v
        conn.close()
    except Exception as e:
        STATE["error"] = f"get_config: {e}"
    return cfg


def set_config(updates: Dict[str, str]) -> Dict[str, str]:
    allowed = set(DEFAULTS.keys())
    try:
        conn = journal._connect(); cur = conn.cursor()
        for k, v in updates.items():
            if k in allowed:
                cur.execute(journal._q(
                    "INSERT INTO ledger_config (k,v) VALUES (?,?) ON CONFLICT (k) DO UPDATE SET v=?"),
                    (k, str(v), str(v)))
        conn.commit(); conn.close()
    except Exception as e:
        STATE["error"] = f"set_config: {e}"; print("ledger set_config error:", e)
    return get_config()


def current_bank() -> float:
    cfg = get_config()
    start = float(cfg.get("start_bank") or 0)
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute("SELECT COALESCE(SUM(pl),0) FROM bets WHERE status IN ('won','lost','void')")
        settled_pl = cur.fetchone()[0] or 0
        conn.close()
    except Exception as e:
        STATE["error"] = f"current_bank: {e}"; settled_pl = 0
    return round(start + float(settled_pl), 2)


def suggest_stake() -> float:
    cfg = get_config()
    bank = current_bank()
    pct = float(cfg.get("unit_pct") or 0.02)
    cap = bank * float(cfg.get("max_pct") or 0.05)
    stake = min(bank * pct, cap)
    return round(max(stake, 0), 2)


def add_bet(d: Dict) -> Optional[str]:
    if STATE["error"]:
        init()
    bid = uuid.uuid4().hex[:12]
    now = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        odds = float(d.get("odds") or 0)
        stake = float(d.get("stake") or 0)
    except (TypeError, ValueError):
        return None
    if odds <= 1 or stake <= 0:
        return None
    row = {
        "id": bid, "created_at": now,
        "match_id": d.get("match_id"), "league": d.get("league"),
        "home": d.get("home"), "away": d.get("away"),
        "match_date": d.get("match_date"), "kickoff": d.get("kickoff"),
        "div": d.get("div"), "market": d.get("market"), "selection": d.get("selection"),
        "bet_type": "live" if d.get("bet_type") == "live" else "prematch",
        "odds": odds, "model_prob": d.get("model_prob"), "stake": stake,
        "status": "pending", "payout": None, "pl": None,
        "closing_odds": None, "clv_pct": None,
        "bank_before": current_bank(), "settled_at": None, "notes": d.get("notes"),
    }
    sql = journal._q("""INSERT INTO bets
        (id,created_at,match_id,league,home,away,match_date,kickoff,div,market,selection,
         bet_type,odds,model_prob,stake,status,payout,pl,closing_odds,clv_pct,bank_before,settled_at,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""")
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute(sql, tuple(row[k] for k in
            ["id","created_at","match_id","league","home","away","match_date","kickoff","div",
             "market","selection","bet_type","odds","model_prob","stake","status","payout","pl",
             "closing_odds","clv_pct","bank_before","settled_at","notes"]))
        conn.commit(); conn.close()
        return bid
    except Exception as e:
        STATE["error"] = f"add_bet: {e}"; print("ledger add_bet error:", e)
        return None


def _apply_result(status: str, odds: float, stake: float):
    if status == "won":
        payout = stake * odds
    elif status == "void":
        payout = stake
    else:  # lost
        payout = 0.0
    return round(payout, 2), round(payout - stake, 2)


def settle_manual(bet_id: str, status: str) -> bool:
    if status not in ("won", "lost", "void"):
        return False
    now = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute(journal._q("SELECT odds,stake FROM bets WHERE id=? AND status='pending'"), (bet_id,))
        r = cur.fetchone()
        if not r:
            conn.close(); return False
        payout, pl = _apply_result(status, float(r[0]), float(r[1]))
        cur.execute(journal._q(
            "UPDATE bets SET status=?, payout=?, pl=?, settled_at=? WHERE id=?"),
            (status, payout, pl, now, bet_id))
        conn.commit(); conn.close()
        return True
    except Exception as e:
        STATE["error"] = f"settle_manual: {e}"; print("ledger settle_manual error:", e)
        return False


def delete_bet(bet_id: str) -> bool:
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute(journal._q("DELETE FROM bets WHERE id=?"), (bet_id,))
        conn.commit(); conn.close()
        return True
    except Exception as e:
        STATE["error"] = f"delete_bet: {e}"; return False


# ---- market outcome resolution for auto-settle ----------------------------
def _outcome(market: str, selection: str, g: Dict) -> Optional[bool]:
    """Return True (won) / False (lost) / None (can't resolve) for a selection."""
    fthg, ftag = g.get("fthg"), g.get("ftag")
    if fthg is None or ftag is None:
        return None
    total = fthg + ftag
    res = "1" if fthg > ftag else ("2" if ftag > fthg else "X")
    sel = (selection or "").strip()
    low = sel.lower()

    if market == "Rezultat":
        if sel.startswith("1 ") or sel == "1": return res == "1"
        if sel.startswith("2 ") or sel == "2": return res == "2"
        if low.startswith("x egal") or sel == "X": return res == "X"
        if sel == "1X": return res in ("1", "X")
        if sel == "X2": return res in ("X", "2")
        if sel == "12": return res in ("1", "2")
        return None

    if market == "Goluri":
        import re
        m = re.search(r"(\d+\.\d+)", sel)
        if low.startswith("peste") and m: return total > float(m.group(1))
        if low.startswith("sub") and m:   return total < float(m.group(1))
        if low.startswith("ambele"):      return fthg > 0 and ftag > 0
        if low.startswith("nu ambele"):   return not (fthg > 0 and ftag > 0)
        if "înscrie" in low or "inscrie" in low:
            # "{home} înscrie" / "{away} înscrie"
            if g.get("home") and g["home"].lower() in low: return fthg > 0
            if g.get("away") and g["away"].lower() in low: return ftag > 0
            return None
        return None

    if market == "Prima repriză":
        hthg, htag = g.get("hthg"), g.get("htag")
        if hthg is None or htag is None:
            return None
        import re
        ht = hthg + htag
        hres = "1" if hthg > htag else ("2" if htag > hthg else "X")
        m = re.search(r"(\d+\.\d+)", sel)
        if low.startswith("peste") and m: return ht > float(m.group(1))
        if low.startswith("sub") and m:   return ht < float(m.group(1))
        if low.startswith("gg"):          return hthg > 0 and htag > 0
        if "repr.1" in low or "repr1" in low:
            if sel.startswith("1 "): return hres == "1"
            if sel.startswith("2 "): return hres == "2"
            if low.startswith("x"):  return hres == "X"
        return None

    if market == "Cornere":
        hc, ac = g.get("hc"), g.get("ac")
        if hc is None or ac is None:
            return None
        import re
        tc = hc + ac
        m = re.search(r"(\d+\.\d+)", sel)
        if low.startswith("peste") and m: return tc > float(m.group(1))
        if low.startswith("sub") and m:   return tc < float(m.group(1))
        return None

    return None


def _closing_odds(market: str, selection: str, g: Dict) -> Optional[float]:
    """Closing odds for CLV — only for markets present in football-data results."""
    sel = (selection or "").strip(); low = sel.lower()
    if market == "Rezultat":
        if sel.startswith("1 "): return g.get("oH")
        if low.startswith("x egal"): return g.get("oD")
        if sel.startswith("2 "): return g.get("oA")
    if market == "Goluri":
        if low.startswith("peste 2.5"): return g.get("oO25")
        if low.startswith("sub 2.5"): return g.get("oU25")
    return None


def auto_settle(hist_by_div: Dict[str, list]) -> int:
    """Close pending bets whose match has been played; compute P/L and CLV."""
    if STATE["error"]:
        init()
    idx = {}
    for div, games in (hist_by_div or {}).items():
        for gg in games:
            if gg.get("played") and gg.get("home") and gg.get("away"):
                idx.setdefault((div, gg["home"], gg["away"]), []).append(gg)
    if not idx:
        return 0
    now = datetime.datetime.utcnow().isoformat() + "Z"
    n = 0
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute("""SELECT id,div,home,away,match_date,market,selection,odds,stake,bet_type
                       FROM bets WHERE status='pending'""")
        pend = cur.fetchall()
        for row in pend:
            bid, div, home, away, mdate, market, selection, odds, stake, btype = row
            cand = idx.get((div, home, away)) if div else None
            if not cand:
                # try to find by teams across any div
                for (d2, h2, a2), games in idx.items():
                    if h2 == home and a2 == away:
                        cand = games; break
            if not cand:
                continue
            g = journal._nearest(cand, mdate or "")
            if not g:
                continue
            won = _outcome(market, selection, g)
            if won is None:
                continue  # leave pending for manual settle
            status = "won" if won else "lost"
            payout, pl = _apply_result(status, float(odds), float(stake))
            clv_pct = None; closing = None
            if btype == "prematch":
                closing = _closing_odds(market, selection, g)
                if closing and closing > 1:
                    clv_pct = round(float(odds) / float(closing) - 1, 4)
            cur.execute(journal._q(
                """UPDATE bets SET status=?, payout=?, pl=?, closing_odds=?, clv_pct=?, settled_at=?
                   WHERE id=?"""),
                (status, payout, pl, closing, clv_pct, now, bid))
            n += 1
        conn.commit(); conn.close()
    except Exception as e:
        STATE["error"] = f"auto_settle: {e}"; print("ledger auto_settle error:", e)
    return n


def bets(limit: int = 200, status: Optional[str] = None) -> List[Dict]:
    if STATE["error"]:
        init()
    where = ""
    if status == "pending":
        where = "WHERE status='pending'"
    elif status == "settled":
        where = "WHERE status IN ('won','lost','void')"
    sql = f"SELECT * FROM bets {where} ORDER BY created_at DESC LIMIT {int(limit)}"
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        out = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return out
    except Exception as e:
        STATE["error"] = f"bets: {e}"; return []


def summary() -> Dict:
    if STATE["error"]:
        init()
    cfg = get_config()
    start = float(cfg.get("start_bank") or 0)
    target = float(cfg.get("target") or 0)
    try:
        conn = journal._connect(); cur = conn.cursor()
        cur.execute("""SELECT status,bet_type,odds,stake,pl,clv_pct,created_at,bank_before,payout
                       FROM bets ORDER BY created_at ASC""")
        rows = cur.fetchall(); conn.close()
    except Exception as e:
        STATE["error"] = f"summary: {e}"
        return {"error": STATE["error"]}

    settled = [r for r in rows if r[0] in ("won", "lost", "void")]
    pending = [r for r in rows if r[0] == "pending"]
    staked = sum(float(r[3]) for r in settled) or 0.0
    profit = sum(float(r[4] or 0) for r in settled) or 0.0
    wins = sum(1 for r in settled if r[0] == "won")
    pending_stake = sum(float(r[3]) for r in pending) or 0.0
    bank = round(start + profit, 2)

    clv_rows = [float(r[5]) for r in settled if r[5] is not None]
    avg_clv = round(sum(clv_rows) / len(clv_rows), 4) if clv_rows else None

    def split(bt):
        s = [r for r in settled if r[1] == bt]
        st = sum(float(r[3]) for r in s) or 0.0
        pr = sum(float(r[4] or 0) for r in s) or 0.0
        w = sum(1 for r in s if r[0] == "won")
        return {"n": len(s), "staked": round(st, 2), "profit": round(pr, 2),
                "yield": round(pr / st, 4) if st else None,
                "strike": round(w / len(s), 3) if s else None}

    # bank curve over settled bets (running)
    curve = []
    run = start
    for r in settled:
        run += float(r[4] or 0)
        curve.append(round(run, 2))

    return {
        "start_bank": round(start, 2), "target": round(target, 2),
        "bank": bank, "pending_n": len(pending), "pending_stake": round(pending_stake, 2),
        "settled_n": len(settled), "staked": round(staked, 2), "profit": round(profit, 2),
        "yield": round(profit / staked, 4) if staked else None,
        "roi_bank": round((bank - start) / start, 4) if start else None,
        "strike": round(wins / len(settled), 3) if settled else None,
        "avg_clv": avg_clv,
        "prematch": split("prematch"), "live": split("live"),
        "suggested_stake": suggest_stake(),
        "unit_pct": float(cfg.get("unit_pct") or 0.02),
        "curve": curve,
    }
