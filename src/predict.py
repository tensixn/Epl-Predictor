"""Predict win/draw/loss probabilities for upcoming fixtures.

Trains on every season in data/raw, then scores fixtures using each team's
latest Elo and form.  Usage (from the project root):

    python -m src.predict "Arsenal" "Chelsea" "Liverpool" "Man City"
"""
import sys

import pandas as pd

from .evaluate import BURN_IN, make_models, select
from .features import LABELS, build_features, feature_columns, load_matches

MODELS = ("logistic", "xgboost")


def train(names=MODELS):
    """Fit the named models on every season; return (models, cols, state)."""
    feats, state = build_features(load_matches())
    feats = feats[feats.season != BURN_IN]
    cols = feature_columns(feats)
    y = feats.FTR.map(LABELS).values
    models = {}
    for name in names:
        factory, use = make_models()[name]
        models[name] = (factory().fit(select(feats, use, cols), y), use)
    return models, cols, state


def predict(models, cols, state, fixtures, date=None):
    """Return {model name: array of [home, draw, away] probabilities per fixture}."""
    date = pd.Timestamp.today().normalize() if date is None else date
    X = pd.DataFrame([state.match_features(h, a, date) for h, a in fixtures])
    return {name: m.predict_proba(select(X, use, cols)) for name, (m, use) in models.items()}


def main(args):
    if not args or len(args) % 2:
        sys.exit('usage: python -m src.predict HOME AWAY [HOME AWAY ...]')
    models, cols, state = train()
    fixtures = list(zip(args[::2], args[1::2]))
    for team in {t for f in fixtures for t in f} - set(state.elo):
        print(f"warning: '{team}' not in the data; treating it as a newly promoted side")

    for name, p in predict(models, cols, state, fixtures).items():
        print(f"\n{name}")
        for (h, a), (ph, pd_, pa) in zip(fixtures, p):
            print(f"  {h:>16} v {a:<16}  home {ph:5.1%}  draw {pd_:5.1%}  away {pa:5.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
