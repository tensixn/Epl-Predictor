"""Monte Carlo simulation of the rest of the season.

Remaining fixtures are sampled from the model's home/draw/away probabilities and added to the current
table. Ratings stay fixed at today's values, so the simulation ignores form changes between now and
the final day.
"""
import numpy as np
import pandas as pd

from .predict import predict


def current_table(played):
    """Points and goal difference per team from played matches ({home, away, ft})."""
    pts, gd = {}, {}
    for m in played:
        h, a = m["ft"]
        for t, gf, ga in ((m["home"], h, a), (m["away"], a, h)):
            pts[t] = pts.get(t, 0) + (3 if gf > ga else 1 if gf == ga else 0)
            gd[t] = gd.get(t, 0) + gf - ga
    return pts, gd


def simulate(pts, gd, remaining, probs, n=10000, seed=0):
    """Return a DataFrame of title / top-4 / relegation odds and expected points per team.

    `remaining` is a list of (home, away); `probs[i]` is that fixture's [home, draw, away] probabilities.
    """
    teams = sorted(pts)
    idx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs).reshape(-1, 3)
    home = np.array([idx[h] for h, _ in remaining], dtype=int)
    away = np.array([idx[a] for _, a in remaining], dtype=int)

    u = rng.random((n, len(remaining)))
    outcome = (u > probs[:, 0]).astype(int) + (u > probs[:, 0] + probs[:, 1])  # 0 home, 1 draw, 2 away
    home_pts, away_pts = np.array([3, 1, 0])[outcome], np.array([0, 1, 3])[outcome]
    total = np.tile(np.array([pts[t] for t in teams], dtype=float), (n, 1))
    for i in range(len(teams)):
        total[:, i] += home_pts[:, home == i].sum(axis=1) + away_pts[:, away == i].sum(axis=1)

    # goal difference breaks ties (current value only), then a coin flip
    score = total + np.array([gd[t] for t in teams]) / 1000 + rng.random(total.shape) * 1e-4
    rank = (-score).argsort(axis=1).argsort(axis=1)  # 0 = champions
    return pd.DataFrame({
        "team": teams,
        "points now": [pts[t] for t in teams],
        "expected points": total.mean(axis=0).round(1),
        "title": (rank == 0).mean(axis=0),
        "top 4": (rank < 4).mean(axis=0),
        "relegation": (rank >= len(teams) - 3).mean(axis=0),
    }).sort_values("expected points", ascending=False, ignore_index=True)


def simulate_season(models, cols, state, matches, model="logistic", **kw):
    """Predict every unplayed match in `matches` ({home, away, ft}) and simulate the rest of the season."""
    pts, gd = current_table([m for m in matches if m["ft"]])
    remaining = [(m["home"], m["away"]) for m in matches if not m["ft"]]
    for t in {t for f in remaining for t in f} - set(pts):  # a team that hasn't played yet
        pts[t], gd[t] = 0, 0
    probs = predict({model: models[model]}, cols, state, remaining)[model] if remaining else []
    return simulate(pts, gd, remaining, probs, **kw)
