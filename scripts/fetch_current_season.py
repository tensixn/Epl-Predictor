"""Write played matches of an in-progress season to data/raw/season-XXYY.csv.

The football-datasets mirror only publishes a season once it is over, so the
current season comes from the Fantasy Premier League API, which posts scores
within hours of the final whistle.  If FPL fails (it is blocked from our cloud
sessions, and it resets between seasons) this falls back to openfootball
(github.com/openfootball/football.json), which can lag a week or more behind.
Both have scores but no shots, cards or referees, so those columns are left
empty; only openfootball has half-time scores.

    python scripts/fetch_current_season.py            # season containing today
    python scripts/fetch_current_season.py 2627       # a specific season
"""
import csv
import json
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
URL = "https://raw.githubusercontent.com/openfootball/football.json/master/20{a}-{b}/en.1.json"
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
           "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]

# openfootball name -> football-data.co.uk spelling used in the rest of data/raw
TEAMS = {
    "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion FC": "Brighton",
    "Birmingham City FC": "Birmingham", "Blackburn Rovers FC": "Blackburn",
    "Bolton Wanderers FC": "Bolton", "Bradford City AFC": "Bradford",
    "Cardiff City FC": "Cardiff", "Charlton Athletic FC": "Charlton",
    "Coventry City FC": "Coventry", "Derby County FC": "Derby",
    "Huddersfield Town AFC": "Huddersfield", "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich", "Leeds United FC": "Leeds", "Leicester City FC": "Leicester",
    "Luton Town FC": "Luton", "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle", "Norwich City FC": "Norwich",
    "Nottingham Forest FC": "Nott'm Forest", "Queens Park Rangers FC": "QPR",
    "Sheffield United FC": "Sheffield United", "Stoke City FC": "Stoke",
    "Sunderland AFC": "Sunderland", "Swansea City AFC": "Swansea",
    "Tottenham Hotspur FC": "Tottenham", "West Bromwich Albion FC": "West Brom",
    "West Ham United FC": "West Ham", "Wigan Athletic FC": "Wigan",
    "Wolverhampton Wanderers FC": "Wolves",
}


# FPL team name -> football-data.co.uk spelling (other FPL names already match it)
FPL_TEAMS = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United",
             "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich"}


def team(name):
    if name in TEAMS:
        return TEAMS[name]
    if name in TEAMS.values():  # already football-data spelling (FPL matches are converted on fetch)
        return name
    short = name.removesuffix(" FC").removesuffix(" AFC").removeprefix("AFC ")
    if short in ("Arsenal", "Aston Villa", "Brentford", "Burnley", "Chelsea", "Crystal Palace",
                 "Everton", "Fulham", "Liverpool", "Middlesbrough", "Portsmouth", "Reading",
                 "Southampton", "Watford", "Blackpool"):
        return short
    sys.exit(f"unknown team name '{name}': add it to TEAMS in {Path(__file__).name}")


def result(h, a):
    return "H" if h > a else "A" if h < a else "D"


def current_season(today=None):
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def full_time(m):
    """[home goals, away goals] of a finished match, else None."""
    score = m.get("score")
    # finished matches have score {"ft": [h, a], "ht": [...]}; some 0-0s are a bare [0, 0]
    return (score.get("ft") if isinstance(score, dict) else score) or None


def rows(matches):
    for m in matches:
        ft = full_time(m)
        if not ft:
            continue
        score = m["score"]
        ht = score.get("ht") if isinstance(score, dict) else None
        row = dict.fromkeys(COLUMNS, "")
        row.update(Date=m["date"], HomeTeam=team(m["team1"]), AwayTeam=team(m["team2"]),
                   FTHG=ft[0], FTAG=ft[1], FTR=result(*ft))
        if ht:
            row.update(HTHG=ht[0], HTAG=ht[1], HTR=result(*ht))
        yield row


def fetch_openfootball(season):
    url = URL.format(a=season[:2], b=season[2:])
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)["matches"]


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epl-predictor)"})  # FPL rejects urllib's
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def known_teams():
    """Every team name in data/raw, i.e. the football-data.co.uk spellings the model knows."""
    names = set()
    for f in RAW_DIR.glob("season-*.csv"):
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                names.update((r["HomeTeam"], r["AwayTeam"]))
    return names


def fpl_matches(fixtures, teams, season, known=None):
    """FPL fixtures as openfootball-style matches (team names already in football-data spelling).

    `teams` is bootstrap-static's "teams" list.  Raises ValueError if the data is for another season or
    has a team name the model doesn't know, so the caller can fall back to openfootball.
    """
    names = {t["id"]: FPL_TEAMS.get(t["name"], t["name"]) for t in teams}
    known = known_teams() if known is None else known
    if unknown := sorted(set(names.values()) - known):
        raise ValueError(f"unknown FPL team names {unknown}: add them to FPL_TEAMS")
    dated = [f for f in fixtures if f.get("kickoff_time")]  # postponed games have no date until rescheduled
    first = min((f["kickoff_time"] for f in dated), default="")
    if not first.startswith(f"20{season[:2]}"):
        raise ValueError(f"FPL holds another season (first kickoff {first or 'none'}), not {season}")
    out = []
    for f in sorted(dated, key=lambda f: (f["kickoff_time"], f["id"])):
        ko = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00")).astimezone(ZoneInfo("Europe/London"))
        m = {"round": f"Matchday {f['event']}", "date": ko.strftime("%Y-%m-%d"), "time": ko.strftime("%H:%M"),
             "team1": names[f["team_h"]], "team2": names[f["team_a"]]}
        # finished_provisional is set at the final whistle; finished only once FPL confirms bonus points
        if (f.get("finished") or f.get("finished_provisional")) and f.get("team_h_score") is not None:
            m["score"] = {"ft": [f["team_h_score"], f["team_a_score"]]}
        out.append(m)
    return out


def fetch_fpl(season):
    return fpl_matches(_get_json(FPL_FIXTURES), _get_json(FPL_BOOTSTRAP)["teams"], season)


def fetch_matches_with_source(season):
    """(matches, source): FPL for the current season, else (or if FPL fails) openfootball."""
    if season == current_season():
        try:
            return fetch_fpl(season), "FPL"
        except Exception as e:  # noqa: BLE001 - any FPL failure means use the slower source, not no data
            print(f"FPL fetch failed ({e!r}); falling back to openfootball", file=sys.stderr)
    return fetch_openfootball(season), "openfootball"


def fetch_matches(season):
    return fetch_matches_with_source(season)[0]


def upcoming(matches):
    """Unplayed matches as dicts with round, date, time, home and away (data/raw spelling)."""
    for m in matches:
        if not m.get("score"):
            yield {"round": m["round"], "date": m["date"], "time": m.get("time", ""),
                   "home": team(m["team1"]), "away": team(m["team2"])}


def main(args):
    season = args[0] if args else current_season()
    matches, source = fetch_matches_with_source(season)
    played = sorted(rows(matches), key=lambda r: (r["Date"], r["HomeTeam"]))
    if not played:
        sys.exit(f"no played matches yet for {season}")
    out = RAW_DIR / f"season-{season}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(played)
    print(f"{out.name}: {len(played)} played matches up to {played[-1]['Date']} ({source})")


if __name__ == "__main__":
    main(sys.argv[1:])
