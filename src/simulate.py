"""Monte Carlo simulation of the rest of the season.

Remaining fixtures are sampled from the model's home/draw/away probabilities and added to the current
table. With `live_elo` (the default) each simulated result also moves both teams' Elo before the next
fixture is drawn, so a team that wins its next few games becomes a better bet in later ones. The
model's probability for a fixture is then shifted by how much the Elo-only model's view of it has
changed since today. Form (last 5 / 10 matches) is still frozen at today's values.
"""
import numpy as np
import pandas as pd

from .features import ELO_HOME_ADV, ELO_K, ELO_PROMOTED
from .predict import predict

# Average Elo goal-difference multiplier, log(|gd| + 1) + 1, over decisive matches in data/raw.
# Simulated results are only H/D/A, so a win uses this instead of a sampled margin; a draw's is 1.
WIN_MULT = 1.98
ELO_GRID = np.arange(-800, 801, 10.0)


def current_table(played):
    """Points and goal difference per team from played matches ({home, away, ft})."""
    pts, gd = {}, {}
    for m in played:
        h, a = m["ft"]
        for t, gf, ga in ((m["home"], h, a), (m["away"], a, h)):
            pts[t] = pts.get(t, 0) + (3 if gf > ga else 1 if gf == ga else 0)
            gd[t] = gd.get(t, 0) + gf - ga
    return pts, gd


def _sample_static(rng, probs, n):
    u = rng.random((n, len(probs)))
    return (u > probs[:, 0]).astype(int) + (u > probs[:, 0] + probs[:, 1])  # 0 home, 1 draw, 2 away


def _sample_live(rng, probs, n, home, away, elo0, curve):
    """Draw fixtures in order, shifting each probability by the Elo drift so far and updating Elo.

    `curve(diff)` returns Elo-only-model probabilities, shape (len(diff), 3), for home-minus-away Elo.
    """
    today = curve(elo0[home] - elo0[away])  # what the Elo-only model says before any simulated result
    elo = np.tile(elo0, (n, 1))
    outcome = np.empty((n, len(probs)), dtype=int)
    for m in range(len(probs)):
        h, a = home[m], away[m]
        diff = elo[:, h] - elo[:, a]
        p = probs[m] * curve(diff) / today[m]
        p /= p.sum(axis=1, keepdims=True)
        u = rng.random(n)
        o = (u > p[:, 0]).astype(int) + (u > p[:, 0] + p[:, 1])
        outcome[:, m] = o
        expected = 1 / (1 + 10 ** ((-diff - ELO_HOME_ADV) / 400))
        delta = ELO_K * np.where(o == 1, 1.0, WIN_MULT) * (np.array([1.0, 0.5, 0.0])[o] - expected)
        elo[:, h] += delta
        elo[:, a] -= delta
    return outcome


def simulate(pts, gd, remaining, probs, n=10000, seed=0, elo=None, curve=None):
    """Return a DataFrame of title / top-4 / relegation odds and expected points per team.

    `remaining` is a list of (home, away) in date order; `probs[i]` is that fixture's [home, draw, away]
    probabilities. Pass `elo` ({team: rating}) and `curve` to update Elo as results are simulated.
    """
    teams = sorted(pts)
    idx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, dtype=float).reshape(-1, 3)
    home = np.array([idx[h] for h, _ in remaining], dtype=int)
    away = np.array([idx[a] for _, a in remaining], dtype=int)

    if elo is None:
        outcome = _sample_static(rng, probs, n)
    else:
        elo0 = np.array([elo.get(t, ELO_PROMOTED) for t in teams])
        outcome = _sample_live(rng, probs, n, home, away, elo0, curve)
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


def elo_curve(models):
    """Function mapping home-minus-away Elo to the Elo-only model's [home, draw, away] probabilities."""
    model, _ = models["elo_logistic"]
    table = model.predict_proba(ELO_GRID[:, None])
    return lambda diff: np.stack([np.interp(diff, ELO_GRID, table[:, k]) for k in range(3)], axis=1)


def simulate_season(models, cols, state, matches, model="logistic", live_elo=True, **kw):
    """Predict every unplayed match in `matches` ({date, home, away, ft}) and simulate the rest of the season."""
    pts, gd = current_table([m for m in matches if m["ft"]])
    todo = sorted((m for m in matches if not m["ft"]), key=lambda m: m["date"])
    remaining = [(m["home"], m["away"]) for m in todo]
    for t in {t for f in remaining for t in f} - set(pts):  # a team that hasn't played yet
        pts[t], gd[t] = 0, 0
    probs = predict({model: models[model]}, cols, state, remaining)[model] if remaining else []
    if live_elo and remaining:
        kw.update(elo=state.elo, curve=elo_curve(models))
    return simulate(pts, gd, remaining, probs, **kw)
