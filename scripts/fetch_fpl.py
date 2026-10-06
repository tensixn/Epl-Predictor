"""This season's Premier League player stats from the Fantasy Premier League API.

Writes data/players/fpl_current.csv: one row per player on a Premier League squad now, with
minutes, starts, goals, assists, expected goals and assists, clean sheets, injury status and
current club. The Scout shows these next to last season's stats, so a shortlist reflects form
this season and where a player plays now.

The FPL API is blocked from our cloud sessions, so this runs from GitHub Actions (the daily
"refresh data" workflow). FPL has no market values; prices stay at Transfermarkt's.

    python scripts/fetch_fpl.py
"""
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "players" / "fpl_current.csv"
URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# FPL team name -> football-data.co.uk spelling used in the rest of the repo
TEAMS = {"Man Utd": "Man United", "Spurs": "Tottenham", "Coventry City": "Coventry", "Hull City": "Hull",
         "Ipswich Town": "Ipswich"}
POSITIONS = {1: "Goalkeeper", 2: "Defender", 3: "Midfield", 4: "Attack"}
STATS = ["minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded", "yellow_cards",
         "expected_goals", "expected_assists"]


def fetch():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (epl-predictor)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def table(data):
    teams = {t["id"]: TEAMS.get(t["name"], t["name"]) for t in data["teams"]}
    finished = [e["id"] for e in data["events"] if e["finished"]]
    df = pd.DataFrame(data["elements"])
    out = pd.DataFrame({
        "fpl_id": df.id, "first_name": df.first_name, "second_name": df.second_name, "web_name": df.web_name,
        "club": df.team.map(teams), "position": df.element_type.map(POSITIONS),
        **{c: pd.to_numeric(df[c], errors="coerce") for c in STATS},
        "status": df.status, "news": df.news.fillna(""),
    })
    out["gameweeks"] = max(finished, default=0)
    out["as_of"] = date.today().isoformat()
    return out.sort_values(["club", "second_name"])


def main():
    df = table(fetch())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} players, {df.club.nunique()} clubs, gameweek {df.gameweeks.iloc[0]}, "
          f"{int(df.goals_scored.sum())} goals, {int(df.starts.sum())} starts -> {OUT.name}")


if __name__ == "__main__":
    main()
