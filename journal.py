"""
FormaCast — prediction journal (the foundation of the "measurement project").

Records every pre-match prediction the model makes, then settles it against the
real result (from the same football-data results we already download). Over time
this gives an HONEST, live track record — immune to backtest self-deception.

Storage:
  - If env DATABASE_URL is set  -> Postgres (persistent, survives redeploys). USE THIS.
  - Otherwise                   -> local SQLite file (works now, but WIPED on every
                                   Render free-tier redeploy/cold start).

Tables
  predictions(id PK, div, league, date, kickoff, home, away,
              p_home, p_draw, p_away, fav, fav_p, mu, lh, la,
              p_over25, p_btts, snapshot_at,
              result, fthg, ftag, total, over25, btts, correct1x2, settled_at)
"""
import os, sqlite3, datetime
from typing import List, Dict, Optional

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_PG = DATABASE_URL.startswith("postgres")
HERE = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.environ.get("JOURNAL_DB", os.path.join(HERE, "journal.db"))

_pg = None
if IS_PG:
    try:
        import psycopg2 as _pg  # type: ignore
        import psycopg2.extras  # noqa
    except Exception as e:  # pragma: no cover
        print("journal: psycopg2 missing, falling back to SQLite:", e)
        IS_PG = False


def _connect():
    if IS_PG:
        return _pg.connect(DATABASE_URL, sslmode=os.environ.get("PGSSLMODE", "require"))
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _q(sql: str) -> str:
    """Translate '?' placeholders to '%s' for Postgres."""
    return sql.replace("?", "%s") if IS_PG else sql


DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    div TEXT, league TEXT, date TEXT, kickoff TEXT, home TEXT, away TEXT,
    p_home REAL, p_draw REAL, p_away REAL, fav TEXT, fav_p REAL,
    mu REAL, lh REAL, la REAL, p_over25 REAL, p_btts REAL,
    snapshot_at TEXT,
    result TEXT, fthg INTEGER, ftag INTEGER, total INTEGER,
    over25 INTEGER, btts INTEGER, correct1x2 INTEGER, settled_at TEXT
);
"""

STATE = {"backend": "postgres" if IS_PG else "sqlite", "error": None,
         "last_record": None, "last_settle": None}


def init():
    try:
        conn = _connect(); cur = conn.cursor()
        cur.execute(DDL)
        conn.commit(); conn.close()
        STATE["error"] = None
    except Exception as e:
        STATE["error"] = f"init: {e}"
        print("journal init error:", e)


def _pick(group, label):
    for p in group.get("picks", []):
        if p["label"] == label:
            return p["p"]
    return None


def _extract(f: Dict) -> Optional[Dict]:
    """Pull the fields we track out of a fixture/prediction dict."""
    res = next((g for g in f.get("groups", []) if g["name"] == "Rezultat"), None)
    goals = next((g for g in f.get("groups", []) if g["name"] == "Goluri"), None)
    if not res:
        return None
    ph = _pick(res, f"1 {f['home']}"); px = _pick(res, "X Egal"); pa = _pick(res, f"2 {f['away']}")
    if ph is None or px is None or pa is None:
        return None
    trio = [("1", ph), ("X", px), ("2", pa)]
    fav, fav_p = max(trio, key=lambda t: t[1])
    return {
        "id": f["id"], "div": f.get("div"), "league": f.get("league"),
        "date": f.get("date"), "kickoff": f.get("kickoff"),
        "home": f["home"], "away": f["away"],
        "p_home": ph, "p_draw": px, "p_away": pa, "fav": fav, "fav_p": fav_p,
        "mu": f.get("mu"), "lh": f.get("lh"), "la": f.get("la"),
        "p_over25": _pick(goals, "Peste 2.5") if goals else None,
        "p_btts": _pick(goals, "Ambele înscriu") if goals else None,
    }


def record(fixtures: List[Dict]):
    """Insert the FIRST (earliest) prediction snapshot per fixture. Idempotent."""
    if STATE["error"]:
        init()
    rows = [r for r in (_extract(f) for f in fixtures) if r]
    if not rows:
        return 0
    now = datetime.datetime.utcnow().isoformat() + "Z"
    sql = _q("""INSERT INTO predictions
        (id,div,league,date,kickoff,home,away,p_home,p_draw,p_away,fav,fav_p,mu,lh,la,p_over25,p_btts,snapshot_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT (id) DO NOTHING""")
    n = 0
    try:
        conn = _connect(); cur = conn.cursor()
        for r in rows:
            cur.execute(sql, (r["id"], r["div"], r["league"], r["date"], r["kickoff"],
                              r["home"], r["away"], r["p_home"], r["p_draw"], r["p_away"],
                              r["fav"], r["fav_p"], r["mu"], r["lh"], r["la"],
                              r["p_over25"], r["p_btts"], now))
            n += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit(); conn.close()
        STATE["last_record"] = now
    except Exception as e:
        STATE["error"] = f"record: {e}"; print("journal record error:", e)
    return n


def settle(hist_by_div: Dict[str, list]):
    """Attach real results to any prediction that has now been played."""
    if STATE["error"]:
        init()
    # index results: (div, home, away) -> list of played games
    idx = {}
    for div, games in (hist_by_div or {}).items():
        for g in games:
            if g.get("played") and g.get("home") and g.get("away"):
                idx.setdefault((div, g["home"], g["away"]), []).append(g)
    if not idx:
        return 0
    now = datetime.datetime.utcnow().isoformat() + "Z"
    upd = _q("""UPDATE predictions SET result=?, fthg=?, ftag=?, total=?, over25=?, btts=?,
                correct1x2=?, settled_at=? WHERE id=?""")
    n = 0
    try:
        conn = _connect(); cur = conn.cursor()
        cur.execute("SELECT id,div,home,away,date,fav FROM predictions WHERE result IS NULL")
        pending = cur.fetchall()
        for row in pending:
            pid, div, home, away, pdate = row[0], row[1], row[2], row[3], row[4]
            fav = row[5]
            cand = idx.get((div, home, away))
            if not cand:
                continue
            g = _nearest(cand, pdate)
            if not g:
                continue
            fthg, ftag = g["fthg"], g["ftag"]
            result = "1" if fthg > ftag else ("2" if ftag > fthg else "X")
            total = fthg + ftag
            over25 = 1 if total >= 3 else 0
            btts = 1 if (fthg > 0 and ftag > 0) else 0
            correct = 1 if fav == result else 0
            cur.execute(upd, (result, fthg, ftag, total, over25, btts, correct, now, pid))
            n += 1
        conn.commit(); conn.close()
        STATE["last_settle"] = now
    except Exception as e:
        STATE["error"] = f"settle: {e}"; print("journal settle error:", e)
    return n


def _nearest(games, pdate):
    """Pick the played game closest in date to the predicted date (handles postponements)."""
    if not games:
        return None
    try:
        pd = datetime.date.fromisoformat((pdate or "")[:10])
    except Exception:
        return games[0]
    best, bestd = None, 99
    for g in games:
        gd = g.get("date")
        if not gd:
            continue
        try:
            diff = abs((gd - pd).days)
        except Exception:
            diff = 0
        if diff <= 5 and diff < bestd:
            best, bestd = g, diff
    return best


def entries(limit: int = 100, only_settled: bool = False) -> List[Dict]:
    if STATE["error"]:
        init()
    where = "WHERE result IS NOT NULL" if only_settled else ""
    sql = f"SELECT * FROM predictions {where} ORDER BY date DESC, snapshot_at DESC LIMIT {int(limit)}"
    try:
        conn = _connect(); cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        out = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return out
    except Exception as e:
        STATE["error"] = f"entries: {e}"; print("journal entries error:", e)
        return []


def stats() -> Dict:
    """Honest live track record computed from settled predictions."""
    if STATE["error"]:
        init()
    try:
        conn = _connect(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]
        cur.execute("""SELECT league,p_home,p_draw,p_away,fav,result,correct1x2,
                       p_over25,over25,p_btts,btts FROM predictions WHERE result IS NOT NULL""")
        rows = cur.fetchall(); conn.close()
    except Exception as e:
        STATE["error"] = f"stats: {e}"; print("journal stats error:", e)
        return {"error": STATE["error"]}

    settled = len(rows)
    if settled == 0:
        return {"total": total, "settled": 0, "pending": total,
                "note": "Încă niciun meci închis. Revino după ce se joacă meciurile prezise."}

    correct = sum(r[6] or 0 for r in rows)
    # multiclass Brier for 1x2
    brier = 0.0
    for r in rows:
        ph, pd_, pa, result = r[1] or 0, r[2] or 0, r[3] or 0, r[5]
        y = {"1": (1, 0, 0), "X": (0, 1, 0), "2": (0, 0, 1)}.get(result, (0, 0, 0))
        brier += (ph - y[0])**2 + (pd_ - y[1])**2 + (pa - y[2])**2
    brier /= settled

    # over 2.5 calibration
    o_rows = [r for r in rows if r[7] is not None and r[8] is not None]
    o_pred = sum(r[7] for r in o_rows) / len(o_rows) if o_rows else None
    o_real = sum(r[8] for r in o_rows) / len(o_rows) if o_rows else None
    # btts calibration
    b_rows = [r for r in rows if r[9] is not None and r[10] is not None]
    b_pred = sum(r[9] for r in b_rows) / len(b_rows) if b_rows else None
    b_real = sum(r[10] for r in b_rows) / len(b_rows) if b_rows else None

    # per-league (full name, so tiers like "Anglia · Premier League" vs
    # "Anglia · Championship" don't collapse into the same bucket)
    byl = {}
    for r in rows:
        lg = r[0] or "?"
        d = byl.setdefault(lg, {"n": 0, "correct": 0})
        d["n"] += 1; d["correct"] += r[6] or 0
    per_league = sorted(
        [{"league": k, "n": v["n"], "acc": round(v["correct"]/v["n"], 3)} for k, v in byl.items()],
        key=lambda x: -x["n"])

    return {
        "total": total, "settled": settled, "pending": total - settled,
        "acc1x2": round(correct/settled, 3),
        "brier": round(brier, 4),
        "over25": {"pred": round(o_pred, 3) if o_pred is not None else None,
                   "real": round(o_real, 3) if o_real is not None else None,
                   "n": len(o_rows)},
        "btts": {"pred": round(b_pred, 3) if b_pred is not None else None,
                 "real": round(b_real, 3) if b_real is not None else None,
                 "n": len(b_rows)},
        "per_league": per_league,
    }
