import importlib.util
import json
from pathlib import Path

import pytest

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


def test_upcoming_keeps_unplayed_matches():
    matches = [
        {"round": "Matchday 1", "date": "2026-08-21", "team1": "Arsenal FC", "team2": "Coventry City FC",
         "score": {"ft": [3, 0]}},
        {"round": "Matchday 6", "date": "2026-10-10", "time": "12:30", "team1": "Arsenal FC",
         "team2": "Leeds United FC"},
    ]
    assert list(fcs.upcoming(matches)) == [
        {"round": "Matchday 6", "date": "2026-10-10", "time": "12:30", "home": "Arsenal", "away": "Leeds"}]


# Trimmed copy of FPL's /api/fixtures/ plus the "teams" list from /api/bootstrap-static/
FPL = json.loads((Path(__file__).parent / "data" / "fpl_sample.json").read_text())
KNOWN = {"Arsenal", "Chelsea", "Coventry", "Fulham", "Leeds", "Man City", "Man United", "Tottenham"}


def test_fpl_matches_convert_to_openfootball_shape():
    matches = fcs.fpl_matches(FPL["fixtures"], FPL["teams"], "2627", known=KNOWN)
    assert len(matches) == 5  # the postponed game has no date yet
    assert matches[0] == {"round": "Matchday 1", "date": "2026-08-21", "time": "20:00", "team1": "Arsenal",
                          "team2": "Coventry", "score": {"ft": [3, 0]}}
    rows = list(fcs.rows(matches))
    assert [(r["HomeTeam"], r["AwayTeam"], r["FTHG"], r["FTAG"], r["FTR"]) for r in rows] == [
        ("Arsenal", "Coventry", 3, 0, "H"), ("Man United", "Tottenham", 1, 1, "D"), ("Man City", "Leeds", 2, 1, "H")]
    assert rows[0]["HTHG"] == ""  # FPL has no half-time scores
    assert list(fcs.upcoming(matches)) == [
        {"round": "Matchday 7", "date": "2026-10-17", "time": "12:30", "home": "Chelsea", "away": "Fulham"},
        {"round": "Matchday 7", "date": "2026-10-26", "time": "20:00", "home": "Leeds", "away": "Arsenal"}]


def test_fpl_matches_reject_unknown_team_and_other_season():
    with pytest.raises(ValueError, match="unknown FPL team"):
        fcs.fpl_matches(FPL["fixtures"], FPL["teams"], "2627", known=KNOWN - {"Leeds"})
    with pytest.raises(ValueError, match="another season"):
        fcs.fpl_matches(FPL["fixtures"], FPL["teams"], "2526", known=KNOWN)


def test_fetch_falls_back_to_openfootball_when_fpl_fails(monkeypatch):
    def down(season):
        raise OSError("403")
    monkeypatch.setattr(fcs, "fetch_fpl", down)
    monkeypatch.setattr(fcs, "fetch_openfootball", lambda season: ["openfootball"])
    assert fcs.fetch_matches_with_source(fcs.current_season()) == (["openfootball"], "openfootball")
    monkeypatch.setattr(fcs, "fetch_fpl", lambda season: ["fpl"])
    assert fcs.fetch_matches_with_source(fcs.current_season()) == (["fpl"], "FPL")
    assert fcs.fetch_matches_with_source("2425") == (["openfootball"], "openfootball")  # FPL only has this season
