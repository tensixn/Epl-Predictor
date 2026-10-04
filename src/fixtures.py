"""Upcoming Premier League fixtures, read from openfootball (see scripts/fetch_current_season.py)."""
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
