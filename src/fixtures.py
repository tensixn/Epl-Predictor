"""Upcoming Premier League fixtures, from FPL or openfootball (see scripts/fetch_current_season.py)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fetch_current_season", Path(__file__).resolve().parents[1] / "scripts" / "fetch_current_season.py")
_fcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fcs)


def next_round(season=None):
    """Return (round name, fixtures) for the first round with unplayed matches, or (None, [])."""
    todo = sorted(_fcs.upcoming(_fcs.fetch_matches(season or _fcs.current_season())),
                  key=lambda f: (f["date"], f["time"]))
    if not todo:
        return None, []
    name = todo[0]["round"]
    return name, [f for f in todo if f["round"] == name]


def season_matches(season=None):
    """Every match of the season as {date, home, away, ft}; ft is [home goals, away goals] or None if unplayed."""
    return [{"date": m["date"], "home": _fcs.team(m["team1"]), "away": _fcs.team(m["team2"]), "ft": _fcs.full_time(m)}
            for m in _fcs.fetch_matches(season or _fcs.current_season())]


ODDS_URL = "https://football-data.co.uk/fixtures.csv"  # this week's fixtures only, so E0 rows show up a few days before kickoff


def upcoming_odds():
    """{(date, home, away): (home, draw, away) decimal odds} for Premier League fixtures in football-data's weekly file."""
    import io
    import urllib.request

    import pandas as pd
    with urllib.request.urlopen(ODDS_URL, timeout=30) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()), encoding="utf-8-sig")
    out = {}
    for r in df[df.Div == "E0"].itertuples():
        for cols in (("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")):  # market average, else Bet365
            odds = tuple(getattr(r, c, float("nan")) for c in cols)
            if not pd.isna(odds).any():
                date = pd.to_datetime(r.Date, dayfirst=True).strftime("%Y-%m-%d")
                out[(date, r.HomeTeam, r.AwayTeam)] = odds
                break
    return out
