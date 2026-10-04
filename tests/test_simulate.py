import numpy as np

from src.simulate import current_table, simulate


def test_current_table():
    pts, gd = current_table([{"home": "A", "away": "B", "ft": [2, 0]}, {"home": "B", "away": "A", "ft": [1, 1]}])
    assert pts == {"A": 4, "B": 1} and gd == {"A": 2, "B": -2}


def test_odds_are_consistent_and_leader_wins():
    teams = list("ABCDEFGH")
    pts = {t: p for t, p in zip(teams, [60, 40, 38, 36, 30, 28, 20, 10])}
    gd = dict.fromkeys(teams, 0)
    remaining = [("A", "H"), ("H", "A"), ("B", "C"), ("C", "B")]
    probs = np.tile([0.4, 0.3, 0.3], (4, 1))
    out = simulate(pts, gd, remaining, probs, n=2000).set_index("team")
    assert out.title.sum() == 1 and out["top 4"].sum() == 4 and out.relegation.sum() == 3
    assert out.title["A"] == 1 and out.relegation["H"] == 1
