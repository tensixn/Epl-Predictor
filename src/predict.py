"""Predict win/draw/loss probabilities for upcoming fixtures.

Trains on every season in data/raw, then scores fixtures using each team's
latest Elo and form.  Usage (from the project root):

    python -m src.predict "Arsenal" "Chelsea" "Liverpool" "Man City"
"""
import sys

import pandas as pd

from .evaluate import BURN_IN, make_models, select
from .features import LABELS, build_features, feature_columns, load_matches


def main(args):
    if not args or len(args) % 2:
        sys.exit('usage: python -m src.predict HOME AWAY [HOME AWAY ...]')
    feats, state = build_features(load_matches())
    feats = feats[feats.season != BURN_IN]
    cols = feature_columns(feats)
    y = feats.FTR.map(LABELS).values

    known = set(state.elo)
    today = pd.Timestamp.today().normalize()
    fixtures = list(zip(args[::2], args[1::2]))
    for team in {t for f in fixtures for t in f} - known:
        print(f"warning: '{team}' not in the data; treating it as a newly promoted side")
    X = pd.DataFrame([state.match_features(h, a, today) for h, a in fixtures])

    for name in ("logistic", "xgboost"):
        factory, use = make_models()[name]
        p = factory().fit(select(feats, use, cols), y).predict_proba(select(X, use, cols))
        print(f"\n{name}")
        for (h, a), (ph, pd_, pa) in zip(fixtures, p):
            print(f"  {h:>16} v {a:<16}  home {ph:5.1%}  draw {pd_:5.1%}  away {pa:5.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
