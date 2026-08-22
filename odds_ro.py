"""
FormaCast — Romanian bookmaker odds: arbitrage + value scanner.

Uses OddsPapi (https://oddspapi.io) which covers RO books (Superbet RO, Betano RO,
WinBet RO, ...). The API key is read from the ODDS_API_KEY environment variable,
NEVER hardcoded. Set it in Render: Environment -> Add -> ODDS_API_KEY = <key>.

Because API response shapes vary, parsing is defensive and logs what it sees so
the exact field names can be tuned after the first live call (GET /api/odds/debug).
"""
import os, time, threading, itertools
from typing import Dict, List, Optional

import requests

ODDS_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_BASE = os.environ.get("ODDS_API_BASE", "https://api.oddspapi.io/v4")
# RO-licensed books we care about (slugs as OddsPapi lists them; tuned after first call)
RO_BOOKS = [b.strip() for b in os.environ.get(
    "RO_BOOKS", "superbet.ro,betano.ro,winbet.ro,unibet.ro,fortuna.ro,netbet.ro"
).split(",") if b.strip()]
SPORT_ID = int(os.environ.get("ODDS_SPORT_ID", "10"))  # 10 = soccer on OddsPapi
ODDS_REFRESH_MIN = int(os.environ.get("ODDS_REFRESH_MIN", "20"))

ODDS_CACHE: Dict = {"updated_at": None, "events": [], "arbs": [], "values": [], "error": None, "raw_sample": None}
ODDS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------
def fetch_odds() -> Optional[dict]:
    if not ODDS_KEY:
        return {"_error": "ODDS_API_KEY nu e setat (Render -> Environment)."}
    try:
        r = requests.get(
            f"{ODDS_BASE}/odds",
            params={"apiKey": ODDS_KEY, "sportId": SPORT_ID, "bookmakers": ",".join(RO_BOOKS)},
            timeout=30,
        )
        if r.status_code != 200:
            return {"_error": f"OddsPapi {r.status_code}: {r.text[:180]}"}
        return r.json()
    except requests.RequestException as e:
        return {"_error": f"conexiune: {e}"}


# ---------------------------------------------------------------------------
# PARSE  (defensive: tolerates a couple of common response shapes)
# ---------------------------------------------------------------------------
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def extract_1x2(event) -> Dict[str, Dict[str, float]]:
    """Return {book: {'H':odd,'D':odd,'A':odd}} for the 1X2 / moneyline market."""
    out = {}
    books = event.get("bookmakerOdds") or event.get("bookmakers") or {}
    # shape A: {book: {"markets": {"1x2"/"moneyline": {...}}}}
    if isinstance(books, dict):
        for book, data in books.items():
            trip = _find_1x2(data)
            if trip:
                out[book] = trip
    # shape B: {book: [ {"name":"Moneyline","odds":[{"home":..,"draw":..,"away":..}]}, ...]}
    elif isinstance(books, list):
        for entry in books:
            book = entry.get("name") or entry.get("bookmaker")
            trip = _find_1x2(entry)
            if book and trip:
                out[book] = trip
    return out

def _find_1x2(data) -> Optional[Dict[str, float]]:
    # dig for something that has home/draw/away (or 1/X/2)
    def grab(d):
        if not isinstance(d, dict):
            return None
        h = _f(d.get("home") if "home" in d else d.get("1"))
        dr = _f(d.get("draw") if "draw" in d else d.get("X") or d.get("x"))
        a = _f(d.get("away") if "away" in d else d.get("2"))
        if h and dr and a:
            return {"H": h, "D": dr, "A": a}
        return None
    if isinstance(data, dict):
        # markets container
        markets = data.get("markets") or data
        for key in ("1x2", "1X2", "moneyline", "Moneyline", "match_odds", "result"):
            m = markets.get(key) if isinstance(markets, dict) else None
            if m:
                got = grab(m) or (grab(m[0]) if isinstance(m, list) and m else None) or grab(m.get("odds", [{}])[0] if isinstance(m, dict) else {})
                if got:
                    return got
        got = grab(data)
        if got:
            return got
    if isinstance(data, dict) and isinstance(data.get("odds"), list) and data["odds"]:
        return grab(data["odds"][0])
    return None


# ---------------------------------------------------------------------------
# ARBITRAGE + VALUE MATH
# ---------------------------------------------------------------------------
def best_odds(book_trips: Dict[str, Dict[str, float]]):
    """Best odd per outcome across books, plus which book offers it."""
    best = {}
    for outcome in ("H", "D", "A"):
        pick = None
        for book, t in book_trips.items():
            o = t.get(outcome)
            if o and (pick is None or o > pick[1]):
                pick = (book, o)
        if pick:
            best[outcome] = {"book": pick[0], "odd": pick[1]}
    return best

def arbitrage(best):
    """If sum(1/best_odd) < 1 -> guaranteed profit. Returns stakes for 100 unit total."""
    if len(best) < 3:
        return None
    inv = sum(1.0 / best[o]["odd"] for o in ("H", "D", "A"))
    if inv >= 1.0:
        return None
    profit_pct = (1.0 / inv - 1.0) * 100.0
    stakes = {o: round(100.0 * (1.0 / best[o]["odd"]) / inv, 2) for o in ("H", "D", "A")}
    return {"profit_pct": round(profit_pct, 2), "inv_sum": round(inv, 4), "stakes": stakes}

def consensus_probs(book_trips: Dict[str, Dict[str, float]]):
    """De-margined average implied prob across all RO books (market consensus)."""
    accs = {"H": [], "D": [], "A": []}
    for t in book_trips.values():
        s = sum(1.0 / t[o] for o in ("H", "D", "A") if t.get(o))
        if s <= 0:
            continue
        for o in ("H", "D", "A"):
            if t.get(o):
                accs[o].append((1.0 / t[o]) / s)
    if not accs["H"]:
        return None
    return {o: sum(accs[o]) / len(accs[o]) for o in ("H", "D", "A")}

def value_bets(book_trips, best, min_edge=0.03):
    """A book's odd implies a lower prob than the RO consensus -> +EV vs market."""
    cons = consensus_probs(book_trips)
    if not cons:
        return []
    out = []
    for o in ("H", "D", "A"):
        b = best.get(o)
        if not b:
            continue
        p = cons[o]                      # our 'true' estimate = RO consensus
        edge = p * b["odd"] - 1.0        # EV per 1 unit staked
        if edge >= min_edge:
            out.append({"outcome": o, "book": b["book"], "odd": b["odd"],
                        "prob": round(p, 4), "edge": round(edge, 4)})
    return out

def kelly(prob, odd, fraction=0.25):
    b = odd - 1.0
    if b <= 0:
        return 0.0
    f = (prob * odd - 1.0) / b
    return max(0.0, round(f * fraction, 4))


# ---------------------------------------------------------------------------
# REFRESH JOB
# ---------------------------------------------------------------------------
def refresh_odds():
    data = fetch_odds()
    if not data or "_error" in (data or {}):
        with ODDS_LOCK:
            ODDS_CACHE["error"] = (data or {}).get("_error", "necunoscut")
        return
    events_in = data.get("data") or data.get("events") or (data if isinstance(data, list) else [])
    arbs, values, events_out = [], [], []
    sample = None
    for ev in events_in:
        try:
            parts = ev.get("participants") or {}
            home = (parts.get("home") or {}).get("name") if isinstance(parts, dict) else ev.get("home")
            away = (parts.get("away") or {}).get("name") if isinstance(parts, dict) else ev.get("away")
            home = home or ev.get("home") or "?"
            away = away or ev.get("away") or "?"
            trips = extract_1x2(ev)
            if sample is None:
                sample = {"home": home, "away": away, "books_found": list(trips.keys())}
            if len(trips) < 2:
                continue
            best = best_odds(trips)
            date = ev.get("date") or ev.get("start") or ""
            league = (ev.get("league") or {}).get("name") if isinstance(ev.get("league"), dict) else ev.get("league")
            base = {"home": home, "away": away, "date": date, "league": league,
                    "best": best, "books": {b: trips[b] for b in trips}}
            events_out.append(base)
            arb = arbitrage(best)
            if arb:
                arbs.append({**base, "arb": arb})
            vs = value_bets(trips, best)
            for v in vs:
                v["kelly"] = kelly(v["prob"], v["odd"])
                values.append({**{k: base[k] for k in ("home", "away", "date", "league")}, **v})
        except Exception:  # noqa
            continue
    arbs.sort(key=lambda x: -x["arb"]["profit_pct"])
    values.sort(key=lambda x: -x["edge"])
    with ODDS_LOCK:
        ODDS_CACHE.update({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "events": events_out, "arbs": arbs, "values": values,
                           "error": None, "raw_sample": sample})


def start_odds_scheduler(scheduler):
    threading.Thread(target=refresh_odds, daemon=True).start()
    scheduler.add_job(refresh_odds, "interval", minutes=ODDS_REFRESH_MIN, id="odds")
