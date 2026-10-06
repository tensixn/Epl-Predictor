"""Predict win/draw/loss probabilities for upcoming fixtures.

Trains on every season in data/raw, then scores fixtures using each team's
latest Elo and form.  Usage (from the project root):

    python -m src.predict "Arsenal" "Chelsea" "Liverpool" "Man City"
"""
import sys

import pandas as pd

from .dixoncoles import Blend, DixonColes
from .evaluate import BURN_IN, make_models, select
from .features import LABELS, build_features, feature_columns, load_matches

MODELS = ("logistic", "xgboost", "elo_logistic", "blend")  # blend = logistic + Dixon-Coles


def train(names=MODELS, matches=None):
    """Fit the named models on every season; return (models, cols, state)."""
    matches = load_matches() if matches is None else matches
    feats, state = build_features(matches)
    feats = feats[feats.season != BURN_IN]
    cols = feature_columns(feats)
    y = feats.FTR.map(LABELS).values
    models = {}
    for name in names:
        if name != "blend":
            factory, use = make_models()[name]
            models[name] = (factory().fit(select(feats, use, cols), y), use)
    if "blend" in names:  # needs "logistic" in names
        dc = DixonColes(matches, matches.Date.max() + pd.Timedelta(days=1))
        models["blend"] = (Blend(models["logistic"][0], dc), "all")
    return models, cols, state


def predict(models, cols, state, fixtures, date=None):
    """Return {model name: array of [home, draw, away] probabilities per fixture}."""
    date = pd.Timestamp.today().normalize() if date is None else date
    X = pd.DataFrame([state.match_features(h, a, date) for h, a in fixtures])
    return {name: m.predict_fixtures(fixtures, select(X, use, cols)) if hasattr(m, "predict_fixtures")
            else m.predict_proba(select(X, use, cols)) for name, (m, use) in models.items()}


def explain(home, away, probs, h, a):
    """Plain-English (headline, reason) for one prediction. `probs` = (home, draw, away); `h`, `a` are team_features
    dicts. Reasons are only given when they point the favourite's way, so the text never contradicts the bar."""
    ph, pd_, pa = probs
    if abs(ph - pa) < 0.08:
        return "Too close to call", f"The teams are evenly matched, so a draw ({pd_:.0%}) is a real possibility."
    fav, opp, f, o, p = (home, away, h, a, ph) if ph > pa else (away, home, a, h, pa)
    reasons = []
    if f["elo"] - o["elo"] > 40:
        reasons.append("is the stronger side over the long run")
    if f["pts_5"] - o["pts_5"] > 0.5:
        reasons.append(f"is in better form ({f['pts_5']:.1f} v {o['pts_5']:.1f} points a game lately)")
    if fav == home:
        reasons.append("plays at home")
    return f"{fav} favoured: {p:.0%} to win", f"{fav} " + (", ".join(reasons[:-1]) + " and " + reasons[-1]
                                                          if len(reasons) > 1 else (reasons or ["has a small edge"])[0]) + "."


def main(args):
    if not args or len(args) % 2:
        sys.exit('usage: python -m src.predict HOME AWAY [HOME AWAY ...]')
    matches = load_matches()
    models, cols, state = train(matches=matches)
    last, season = matches.Date.max(), matches.season.iloc[-1]
    print(f"form and Elo as of {last:%Y-%m-%d} ({(matches.season == season).sum()} "
          f"matches played in {season[:2]}/{season[2:]})")
    fixtures = list(zip(args[::2], args[1::2]))
    for team in {t for f in fixtures for t in f} - set(state.elo):
        print(f"warning: '{team}' not in the data; treating it as a newly promoted side")

    for name, p in predict(models, cols, state, fixtures).items():
        print(f"\n{name}")
        for (h, a), (ph, pd_, pa) in zip(fixtures, p):
            print(f"  {h:>16} v {a:<16}  home {ph:5.1%}  draw {pd_:5.1%}  away {pa:5.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
