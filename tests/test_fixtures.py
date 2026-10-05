import pytest

from src import fixtures

MATCHES = [
    {"round": "Matchday 5", "date": "2026-10-03", "time": "15:00", "team1": "Arsenal FC", "team2": "Coventry City FC",
     "score": {"ft": [3, 0]}},
    {"round": "Matchday 6", "date": "2026-10-11", "time": "16:30", "team1": "Chelsea FC", "team2": "Fulham FC"},
    {"round": "Matchday 6", "date": "2026-10-10", "time": "12:30", "team1": "Arsenal FC", "team2": "Leeds United FC"},
    {"round": "Matchday 7", "date": "2026-10-17", "time": "15:00", "team1": "Everton FC", "team2": "Burnley FC"},
]


@pytest.fixture
def feed(monkeypatch):
    monkeypatch.setattr(fixtures._fcs, "fetch_matches", lambda season: MATCHES)


def test_next_round_is_the_earliest_unplayed_round_in_kickoff_order(feed):
    name, games = fixtures.next_round("2627")
    assert name == "Matchday 6"
    assert [(g["home"], g["away"]) for g in games] == [("Arsenal", "Leeds"), ("Chelsea", "Fulham")]


def test_next_round_when_season_is_over(monkeypatch):
    monkeypatch.setattr(fixtures._fcs, "fetch_matches", lambda season: MATCHES[:1])
    assert fixtures.next_round("2627") == (None, [])


def test_season_matches_marks_unplayed_with_none(feed):
    out = fixtures.season_matches("2627")
    assert out[0] == {"date": "2026-10-03", "home": "Arsenal", "away": "Coventry", "ft": [3, 0]}
    assert [m["ft"] for m in out[1:]] == [None, None, None]
