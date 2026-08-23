"""
FormaCast — kickoff times from API-Football (v3).

GET https://v3.football.api-sports.io/fixtures?date=YYYY-MM-DD&timezone=Europe/Bucharest
Header: x-apisports-key: <API_FOOTBALL_KEY>
Kickoff is fixture.date (ISO 8601 with tz). Free plan: 100 req/day -> we fetch a
few days once and cache; matched to our fixtures by fuzzy team-name.

Key from env API_FOOTBALL_KEY (never hardcoded). Set in Render -> Environment.
"""
import os, time, threading, re, unicodedata
from datetime import datetime, timedelta, timezone
from typing import Dict

import requests

AF_KEY = os.environ.get("API_FOOTBALL_KEY", "")
AF_BASE = "https://v3.football.api-sports.io"
TZ = os.environ.get("KICKOFF_TZ", "Europe/Bucharest")
DAYS_AHEAD = int(os.environ.get("KICKOFF_DAYS_AHEAD", "4"))

# fixtureKey "home|away|date" -> ISO kickoff string ; plus a name-only fallback index
KICKOFFS: Dict[str, str] = {}
KICK_LOCK = threading.Lock()
KICK_STATE = {"updated_at": None, "error": None, "count": 0}


def _norm(name: str) -> str:
    """Normalise a team name for fuzzy matching across data sources."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower()
    for junk in [" fc", " cf", " afc", " sc", " ac", " if", " bk", " sv", " calcio",
                 "1.", "fc ", "cf ", "sv ", "ss ", "us ", "as ", "rc ", "cd ", "sd "]:
        s = s.replace(junk, " ")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def refresh_kickoffs():
    if not AF_KEY:
        with KICK_LOCK:
            KICK_STATE.update({"error": "API_FOOTBALL_KEY nu e setat (Render -> Environment)."})
        return
    headers = {"x-apisports-key": AF_KEY}
    new_map = {}
    err = None
    try:
        today = datetime.now(timezone.utc)
        for i in range(DAYS_AHEAD):
            d = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            r = requests.get(f"{AF_BASE}/fixtures", headers=headers,
                             params={"date": d, "timezone": TZ}, timeout=30)
            if r.status_code != 200:
                err = f"API-Football {r.status_code}: {r.text[:150]}"
                break
            js = r.json()
            for item in js.get("response", []):
                fx = item.get("fixture", {})
                iso = fx.get("date")
                teams = item.get("teams", {})
                home = (teams.get("home") or {}).get("name")
                away = (teams.get("away") or {}).get("name")
                if not (iso and home and away):
                    continue
                daypart = iso[:10]
                # store by normalised home|away|date AND by home|away (loose)
                new_map[f"{_norm(home)}|{_norm(away)}|{daypart}"] = iso
                new_map.setdefault(f"{_norm(home)}|{_norm(away)}", iso)
            time.sleep(1)  # be gentle with the free 100/day quota
    except requests.RequestException as e:
        err = f"conexiune: {e}"
    if new_map:
        with KICK_LOCK:
            KICKOFFS.clear(); KICKOFFS.update(new_map)
            KICK_STATE.update({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "error": err, "count": len(new_map)})
    else:
        with KICK_LOCK:
            KICK_STATE["error"] = err or "niciun rezultat"


def kickoff_for(home: str, away: str, date_iso: str):
    """Return ISO kickoff for a fixture, matching by normalised names (+date if possible)."""
    h, a = _norm(home), _norm(away)
    with KICK_LOCK:
        return (KICKOFFS.get(f"{h}|{a}|{date_iso[:10]}")
                or KICKOFFS.get(f"{h}|{a}")
                or None)


def start_kickoff_scheduler(scheduler):
    threading.Thread(target=refresh_kickoffs, daemon=True).start()
    scheduler.add_job(refresh_kickoffs, "interval", hours=6, id="kickoffs")
