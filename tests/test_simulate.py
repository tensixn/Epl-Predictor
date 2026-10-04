import numpy as np

from src.simulate import _sample_live, _sample_static, current_table, simulate


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


def test_live_elo_with_a_flat_curve_matches_the_static_simulation():
    teams = list("ABCDEF")
    pts = dict(zip(teams, [20, 18, 15, 12, 10, 5]))
    gd = dict.fromkeys(teams, 0)
    remaining = [("A", "B"), ("C", "D"), ("E", "F"), ("B", "A")]
    probs = np.tile([0.5, 0.25, 0.25], (4, 1))
    flat = lambda diff: np.tile([0.4, 0.3, 0.3], (len(diff), 1))
    static = simulate(pts, gd, remaining, probs, n=20000).set_index("team")
    live = simulate(pts, gd, remaining, probs, n=20000, elo=dict.fromkeys(teams, 1500.0), curve=flat).set_index("team")
    assert (static["expected points"] - live["expected points"]).abs().max() < 0.15


def test_live_elo_makes_results_more_spread_out():
    n, matches = 20000, 15
    probs = np.tile([0.4, 0.3, 0.3], (matches, 1))
    home, away = np.zeros(matches, dtype=int), np.arange(1, 4).repeat(5)
    # a higher home rating means a better chance of winning, so wins and losses compound
    curve = lambda d: np.stack([1 / (1 + np.exp(-d / 100)), np.full(len(d), 0.3), 0.7 / (1 + np.exp(d / 100))], axis=1)
    static = _sample_static(np.random.default_rng(1), probs, n)
    live = _sample_live(np.random.default_rng(1), probs, n, home, away, np.full(4, 1500.0), curve)
    assert (live == 0).sum(axis=1).std() > (static == 0).sum(axis=1).std() * 1.05
