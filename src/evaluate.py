"""Walk-forward evaluation: for each test season, train on every earlier season.

Run from the project root:  python -m src.evaluate
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .dixoncoles import walk_forward
from .features import LABELS, build_features, feature_columns, load_matches

TEST_SEASONS = ["2324", "2425", "2526"]
BURN_IN = "0001"  # first season only seeds Elo and form, never used for training
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
ODDS_DIR = Path(__file__).resolve().parents[1] / "data" / "odds"


def brier(y, p):
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def rps(y, p):
    """Ranked probability score: rewards being 'close' (draw) when wrong."""
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(p, axis=1), np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o)[:, :2] ** 2, axis=1) / 2))


class BaseRate:
    """Always predicts the training-set home/draw/away frequencies."""

    def fit(self, X, y):
        self.p = np.bincount(y, minlength=3) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


def make_models():
    lr = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return {
        "base_rate": (BaseRate, None),
        "elo_logistic": (lr, ["diff_elo"]),
        "logistic": (lr, "all"),
        "xgboost": (lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
            objective="multi:softprob", eval_metric="mlogloss"), "all"),
    }


def bookmaker_probs(df, season):
    """Closing odds for `season` as margin-free probabilities aligned to df's rows (NaN if missing)."""
    path = ODDS_DIR / f"season-{season}.csv"
    if not path.exists():
        return np.full((len(df), 3), np.nan)
    odds = pd.read_csv(path, parse_dates=["Date"])
    merged = df[["Date", "HomeTeam", "AwayTeam"]].merge(odds, how="left", on=["Date", "HomeTeam", "AwayTeam"])
    inv = 1 / merged[["OddsH", "OddsD", "OddsA"]].values
    return inv / inv.sum(axis=1, keepdims=True)


def score(season, name, n, y, p):
    p = p / p.sum(axis=1, keepdims=True)  # xgboost's float32 output can miss 1 by a hair
    return dict(season=season, model=name, n=n, log_loss=log_loss(y, p, labels=[0, 1, 2]),
                brier=brier(y, p), rps=rps(y, p), accuracy=accuracy_score(y, p.argmax(1)))


def select(df, cols, all_cols):
    if cols is None:
        return df[all_cols[:1]].values
    return df[all_cols if cols == "all" else cols].values


def main():
    matches = load_matches()
    feats, _ = build_features(matches)
    feats = feats[feats.season != BURN_IN].reset_index(drop=True)
    cols = feature_columns(feats)
    y_all = feats.FTR.map(LABELS).values

    results, per_match, per_match_lr, per_match_blend = [], [], [], []
    for test in TEST_SEASONS:
        tr, te = feats.season < test, feats.season == test
        y, probs = y_all[te], {}
        for name, (factory, use) in make_models().items():
            model = factory().fit(select(feats[tr], use, cols), y_all[tr])
            probs[name] = model.predict_proba(select(feats[te], use, cols))
            results.append(score(test, name, int(te.sum()), y, probs[name]))
            if name in ("xgboost", "logistic"):
                out = feats.loc[te, ["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
                out[["p_home", "p_draw", "p_away"]] = probs[name]
                (per_match if name == "xgboost" else per_match_lr).append(out)
        probs["ensemble"] = (probs["logistic"] + probs["xgboost"]) / 2
        results.append(score(test, "ensemble", int(te.sum()), y, probs["ensemble"]))
        probs["dixon_coles"] = walk_forward(matches, feats[te])
        probs["blend"] = (probs["logistic"] + probs["dixon_coles"]) / 2
        for name in ("dixon_coles", "blend"):
            results.append(score(test, name, int(te.sum()), y, probs[name]))
        out = feats.loc[te, ["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
        out[["p_home", "p_draw", "p_away"]] = probs["blend"]
        per_match_blend.append(out)
        # bookmaker benchmark, on the matches that have odds; models are rescored on the same subset
        book = bookmaker_probs(feats[te], test)
        ok = ~np.isnan(book).any(axis=1)
        if ok.any():
            results.append(score(test, "bookmaker", int(ok.sum()), y[ok], book[ok]))

    res = pd.DataFrame(results)
    summary = res.groupby("model", sort=False)[["log_loss", "brier", "rps", "accuracy"]].mean()
    OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(OUT_DIR / "metrics_by_season.csv", index=False)
    pd.concat(per_match).to_csv(OUT_DIR / "xgboost_test_predictions.csv", index=False)
    pd.concat(per_match_lr).to_csv(OUT_DIR / "logistic_test_predictions.csv", index=False)
    pd.concat(per_match_blend).to_csv(OUT_DIR / "blend_test_predictions.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary.round(4).to_dict(orient="index"), indent=2))
    pd.set_option("display.width", 120)
    print(res.round(4).to_string(index=False))
    print("\nMean over test seasons (lower is better except accuracy):")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
