"""
FormaCast — kickoff times from API-Football (v3), with robust name matching
and persistent caching (Postgres via journal._connect).

GET https://v3.football.api-sports.io/fixtures?date=YYYY-MM-DD&timezone=<tz>
Header: x-apisports-key: <API_FOOTBALL_KEY>. Free plan: 100 req/day.

Why the rewrite:
  - football-data abbreviates ("Man City", "Ath Madrid", "Sp Gijon") while
    API-Football uses full names -> exact match failed. Now: alias map + fuzzy
    (difflib) matching within the same date.
  - We persist kickoffs in the DB and only re-fetch when stale, so restarts /
    redeploys don't burn the daily quota (which caused count:0).
"""
import os, time, threading, re, unicodedata, difflib
from datetime import datetime, timedelta, timezone, date as _date
from typing import Dict, List

import requests
import journal  # reuse persistent DB connection

AF_KEY = os.environ.get("API_FOOTBALL_KEY", "")
AF_BASE = "https://v3.football.api-sports.io"
TZ = os.environ.get("KICKOFF_TZ", "Europe/Bucharest")
DAYS_AHEAD = int(os.environ.get("KICKOFF_DAYS_AHEAD", "7"))
THROTTLE_HOURS = float(os.environ.get("KICKOFF_THROTTLE_H", "5"))  # min gap between API fetches

# date(YYYY-MM-DD) -> list of {"h":norm,"a":norm,"iso":iso}
BYDATE: Dict[str, List[dict]] = {}
KICK_LOCK = threading.Lock()
KICK_STATE = {"updated_at": None, "error": None, "count": 0, "source": None}

# UEFA club competitions (API-Football league ids). Fixtures-only: no historical
# results source (football-data.co.uk doesn't cover them), so no xG prediction —
# just an honest schedule, captured for free from the same by-date fetch below.
EURO_LEAGUES = {2: "Champions League", 3: "Europa League", 848: "Conference League"}
EURO_FIXTURES: List[dict] = []  # [{comp, home, away, iso, status}]
EURO_LOCK = threading.Lock()

# common football-data -> canonical tokens (helps exact + fuzzy)
ALIASES = {
    "man city": "manchester city", "man utd": "manchester united",
    "man united": "manchester united", "nott'm forest": "nottingham forest",
    "sheffield weds": "sheffield wednesday", "sheffield united": "sheffield united",
    "west brom": "west bromwich albion", "wolves": "wolverhampton wanderers",
    "newcastle": "newcastle united", "qpr": "queens park rangers",
    "ath madrid": "atletico madrid", "ath bilbao": "athletic bilbao",
    "atletico": "atletico madrid", "sp gijon": "sporting gijon",
    "espanol": "espanyol", "betis": "real betis", "sociedad": "real sociedad",
    "vallecano": "rayo vallecano", "celta": "celta vigo", "cadiz": "cadiz",
    "la coruna": "deportivo", "alaves": "alaves",
    "inter": "inter milan", "milan": "ac milan", "juventus": "juventus",
    "roma": "as roma", "napoli": "napoli", "verona": "hellas verona",
    "paris sg": "paris saint germain", "psg": "paris saint germain",
    "st etienne": "saint etienne", "m'gladbach": "borussia monchengladbach",
    "leverkusen": "bayer leverkusen", "dortmund": "borussia dortmund",
    "ein frankfurt": "eintracht frankfurt", "fc koln": "fc koln",
    "bayern munich": "bayern munich", "schalke": "schalke",
    "ajax": "ajax", "psv eindhoven": "psv", "az alkmaar": "az",
    "sporting": "sporting cp", "porto": "fc porto", "benfica": "benfica",
}


def _norm(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    s = ALIASES.get(s, s)
    for junk in [" fc", " cf", " afc", " sc", " ac", " if", " bk", " sv", " calcio",
                 "1.", "fc ", "cf ", "sv ", "ss ", "us ", "as ", "rc ", "cd ", "sd "]:
        s = s.replace(junk, " ")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


# ---- persistence ----------------------------------------------------------
def _ensure_tables():
    conn = journal._connect(); cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS kickoffs (date TEXT, h TEXT, a TEXT, iso TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS kickoffs_meta (k TEXT PRIMARY KEY, v TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS euro_fixtures (comp TEXT, home TEXT, away TEXT, iso TEXT, status TEXT)")
    conn.commit(); conn.close()


def _load_from_db():
    try:
        _ensure_tables()
        conn = journal._connect(); cur = conn.cursor()
        cur.execute("SELECT date,h,a,iso FROM kickoffs")
        m = {}
        for d, h, a, iso in cur.fetchall():
            m.setdefault(d, []).append({"h": h, "a": a, "iso": iso})
        cur.execute("SELECT comp,home,away,iso,status FROM euro_fixtures ORDER BY iso ASC")
        euro = [{"comp": c, "home": h, "away": a, "iso": iso, "status": s}
                for c, h, a, iso, s in cur.fetchall()]
        cur.execute("SELECT v FROM kickoffs_meta WHERE k='last_fetch'")
        row = cur.fetchone()
        conn.close()
        with KICK_LOCK:
            BYDATE.clear(); BYDATE.update(m)
            KICK_STATE["count"] = sum(len(v) for v in m.values())
            KICK_STATE["updated_at"] = row[0] if row else None
            KICK_STATE["source"] = "db"
        with EURO_LOCK:
            EURO_FIXTURES.clear(); EURO_FIXTURES.extend(euro)
        return row[0] if row else None
    except Exception as e:
        KICK_STATE["error"] = f"load_db: {e}"; print("kickoffs load_db error:", e)
        return None


def _save_to_db(bydate, euro, when_iso):
    try:
        _ensure_tables()
        conn = journal._connect(); cur = conn.cursor()
        cur.execute("DELETE FROM kickoffs")
        ins = journal._q("INSERT INTO kickoffs (date,h,a,iso) VALUES (?,?,?,?)")
        for d, games in bydate.items():
            for g in games:
                cur.execute(ins, (d, g["h"], g["a"], g["iso"]))
        cur.execute("DELETE FROM euro_fixtures")
        ins2 = journal._q("INSERT INTO euro_fixtures (comp,home,away,iso,status) VALUES (?,?,?,?,?)")
        for e in euro:
            cur.execute(ins2, (e["comp"], e["home"], e["away"], e["iso"], e.get("status")))
        cur.execute(journal._q(
            "INSERT INTO kickoffs_meta (k,v) VALUES ('last_fetch',?) ON CONFLICT (k) DO UPDATE SET v=?"),
            (when_iso, when_iso))
        conn.commit(); conn.close()
    except Exception as e:
        KICK_STATE["error"] = f"save_db: {e}"; print("kickoffs save_db error:", e)


def _stale(last_iso) -> bool:
    if not last_iso:
        return True
    try:
        last = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - last).total_seconds() > THROTTLE_HOURS * 3600
    except Exception:
        return True


# ---- fetch ----------------------------------------------------------------
def refresh_kickoffs(force: bool = False):
    if not AF_KEY:
        KICK_STATE["error"] = "API_FOOTBALL_KEY nu e setat (Render -> Environment)."
        return
    last = _load_from_db()
    if not force and not _stale(last):
        return  # cache still fresh -> save quota
    headers = {"x-apisports-key": AF_KEY}
    bydate = {}; euro = []; err = None
    try:
        today = datetime.now(timezone.utc)
        for i in range(DAYS_AHEAD):
            d = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            r = requests.get(f"{AF_BASE}/fixtures", headers=headers,
                             params={"date": d, "timezone": TZ}, timeout=30)
            if r.status_code != 200:
                err = f"API-Football {r.status_code}: {r.text[:120]}"; break
            js = r.json()
            api_errors = js.get("errors")
            if api_errors:  # e.g. quota reached returns 200 + errors
                err = f"API errors: {str(api_errors)[:120]}"; break
            for item in js.get("response", []):
                fixture = item.get("fixture") or {}
                iso = fixture.get("date")
                teams = item.get("teams") or {}
                home = (teams.get("home") or {}).get("name")
                away = (teams.get("away") or {}).get("name")
                if not (iso and home and away):
                    continue
                bydate.setdefault(iso[:10], []).append(
                    {"h": _norm(home), "a": _norm(away), "iso": iso})
                league_id = (item.get("league") or {}).get("id")
                if league_id in EURO_LEAGUES:
                    euro.append({"comp": EURO_LEAGUES[league_id], "home": home, "away": away,
                                "iso": iso, "status": (fixture.get("status") or {}).get("short")})
            time.sleep(1)
    except requests.RequestException as e:
        err = f"conexiune: {e}"

    now_iso = datetime.now(timezone.utc).isoformat()
    if bydate or euro:
        euro.sort(key=lambda e: e["iso"])
        _save_to_db(bydate, euro, now_iso)
        with KICK_LOCK:
            BYDATE.clear(); BYDATE.update(bydate)
            KICK_STATE.update({"updated_at": now_iso, "error": err,
                               "count": sum(len(v) for v in bydate.values()), "source": "api"})
        with EURO_LOCK:
            EURO_FIXTURES.clear(); EURO_FIXTURES.extend(euro)
    else:
        # keep whatever we had in DB; just record the error
        KICK_STATE["error"] = err or "niciun rezultat"


# ---- matching -------------------------------------------------------------
def kickoff_for(home: str, away: str, date_iso: str):
    h, a = _norm(home), _norm(away)
    day = (date_iso or "")[:10]
    with KICK_LOCK:
        games = list(BYDATE.get(day, []))
        # also consider +/- 1 day for tz/rollover edge cases
        for off in (1, -1):
            try:
                d2 = (_date.fromisoformat(day) + timedelta(days=off)).isoformat()
                games += BYDATE.get(d2, [])
            except Exception:
                pass
    if not games:
        return None
    # 1) exact
    for g in games:
        if g["h"] == h and g["a"] == a:
            return g["iso"]
    # 2) fuzzy: both sides must be close, AND the best combined score must
    # clearly beat the runner-up (protects against prefix-alike collisions
    # like "Manchester City" vs "Manchester United" scoring similarly).
    scored = []
    for g in games:
        sh = difflib.SequenceMatcher(None, h, g["h"]).ratio()
        sa = difflib.SequenceMatcher(None, a, g["a"]).ratio()
        if sh >= 0.60 and sa >= 0.60:
            scored.append((sh + sa, g))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top_score, top_g = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if (top_score - runner_up) >= 0.15 or len(scored) == 1:
        return top_g["iso"]
    return None


def get_all_upcoming(min_date_iso: str = None) -> List[dict]:
    """Flat list of {h,a,iso} (normalized names) across the cached window,
    used to SUPPLEMENT divisions where football-data.co.uk's shared
    fixtures.csv has no upcoming rows (it only lists 'the next round',
    so most divisions are often empty on any given day)."""
    with KICK_LOCK:
        out = []
        for d, games in BYDATE.items():
            if min_date_iso and d < min_date_iso:
                continue
            out.extend(games)
        return out


def resolve_team(norm_name: str, norm_to_raw: Dict[str, str]) -> str:
    """Map a normalized API-Football team name to the exact string used in a
    division's model (e.g. football-data's 'Man City'), via alias/exact match
    first, then a careful fuzzy fallback. Character-ratio alone can't tell
    'Manchester City' from 'Manchester United' (both score ~0.80 against
    similarly-prefixed names), so we also require the best match to clearly
    beat the runner-up — an ambiguous case returns None rather than a guess."""
    if norm_name in norm_to_raw:
        return norm_to_raw[norm_name]
    scored = []
    for cand, raw in norm_to_raw.items():
        s = difflib.SequenceMatcher(None, norm_name, cand).ratio()
        scored.append((s, raw))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top_score, top_raw = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if top_score >= 0.82 and (top_score - runner_up) >= 0.12:
        return top_raw
    return None


def get_euro_fixtures():
    with EURO_LOCK:
        return list(EURO_FIXTURES)


def start_kickoff_scheduler(scheduler):
    def boot():
        _load_from_db()            # instant: use cached times
        refresh_kickoffs()          # fetch only if stale
    threading.Thread(target=boot, daemon=True).start()
    scheduler.add_job(refresh_kickoffs, "interval", hours=6, id="kickoffs")
