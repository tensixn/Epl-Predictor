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
