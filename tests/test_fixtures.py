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


def test_odds_table_reads_premier_league_rows_with_average_else_bet365_odds():
    import pandas as pd
    df = pd.DataFrame({"Div": ["E0", "E0", "E1"], "Date": ["17/10/2026", "18/10/2026", "17/10/2026"],
                       "HomeTeam": ["Chelsea", "Leeds", "Hull"], "AwayTeam": ["Fulham", "Arsenal", "Derby"],
                       "AvgH": [1.8, None, 2.0], "AvgD": [3.6, None, 3.3], "AvgA": [4.5, None, 3.8],
                       "B365H": [1.8, 4.2, 2.0], "B365D": [3.5, 3.6, 3.3], "B365A": [4.6, 1.85, 3.8]})
    assert fixtures.odds_table(df) == {("2026-10-17", "Chelsea", "Fulham"): (1.8, 3.6, 4.5),
                                       ("2026-10-18", "Leeds", "Arsenal"): (4.2, 3.6, 1.85)}
