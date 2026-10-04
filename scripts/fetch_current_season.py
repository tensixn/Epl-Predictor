"""Write played matches of an in-progress season to data/raw/season-XXYY.csv.

The football-datasets mirror only publishes a season once it is over, so the
current season comes from openfootball (github.com/openfootball/football.json),
which is updated after each matchday.  It has scores but no shots, cards or
referees, so those columns are left empty.

    python scripts/fetch_current_season.py            # season containing today
    python scripts/fetch_current_season.py 2627       # a specific season
"""
import csv
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
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


def team(name):
    if name in TEAMS:
        return TEAMS[name]
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


def rows(matches):
    for m in matches:
        score = m.get("score")
        # finished matches have score {"ft": [h, a], "ht": [...]}; some 0-0s are a bare [0, 0]
        ft = score.get("ft") if isinstance(score, dict) else score
        if not ft:
            continue
        ht = score.get("ht") if isinstance(score, dict) else None
        row = dict.fromkeys(COLUMNS, "")
        row.update(Date=m["date"], HomeTeam=team(m["team1"]), AwayTeam=team(m["team2"]),
                   FTHG=ft[0], FTAG=ft[1], FTR=result(*ft))
        if ht:
            row.update(HTHG=ht[0], HTAG=ht[1], HTR=result(*ht))
        yield row


def fetch_matches(season):
    url = URL.format(a=season[:2], b=season[2:])
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)["matches"]


def upcoming(matches):
    """Unplayed matches as dicts with round, date, time, home and away (data/raw spelling)."""
    for m in matches:
        if not m.get("score"):
            yield {"round": m["round"], "date": m["date"], "time": m.get("time", ""),
                   "home": team(m["team1"]), "away": team(m["team2"])}


def main(args):
    season = args[0] if args else current_season()
    matches = fetch_matches(season)
    played = sorted(rows(matches), key=lambda r: (r["Date"], r["HomeTeam"]))
    if not played:
        sys.exit(f"no played matches yet for {season}")
    out = RAW_DIR / f"season-{season}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(played)
    print(f"{out.name}: {len(played)} played matches up to {played[-1]['Date']} (openfootball)")


if __name__ == "__main__":
    main(sys.argv[1:])
