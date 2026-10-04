import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fetch_current_season", Path(__file__).resolve().parents[1] / "scripts" / "fetch_current_season.py")
fcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fcs)


def test_rows_keep_played_matches_and_map_names():
    matches = [
        {"date": "2026-08-21", "team1": "Arsenal FC", "team2": "Coventry City FC",
         "score": {"ht": [2, 0], "ft": [3, 0]}},
        {"date": "2026-09-05", "team1": "Nottingham Forest FC", "team2": "Tottenham Hotspur FC",
         "score": [0, 0]},
        {"date": "2026-10-10", "team1": "Arsenal FC", "team2": "Leeds United FC"},
    ]
    rows = list(fcs.rows(matches))
    assert len(rows) == 2
    assert (rows[0]["HomeTeam"], rows[0]["AwayTeam"], rows[0]["FTR"], rows[0]["HTR"]) == ("Arsenal", "Coventry", "H", "H")
    assert (rows[1]["HomeTeam"], rows[1]["AwayTeam"], rows[1]["FTR"], rows[1]["HS"]) == ("Nott'm Forest", "Tottenham", "D", "")


def test_current_season():
    from datetime import date
    assert fcs.current_season(date(2026, 10, 4)) == "2627"
    assert fcs.current_season(date(2027, 3, 1)) == "2627"
