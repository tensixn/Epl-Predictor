"""Out-of-sample track record: log predictions before kickoff, score them once the result is in.

The log is results/predictions_log.csv, one row per fixture, written by scripts/log_predictions.py.
Only the logistic model is logged (the best in the walk-forward test).  A fixture is logged the first
time its round becomes the next one to play, and is never rewritten afterwards.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import brier
from .features import LABELS
from .fixtures import next_round, season_matches
from .predict import predict

LOG = Path(__file__).resolve().parents[1] / "results" / "predictions_log.csv"
COLUMNS = ["date", "round", "home", "away", "p_home", "p_draw", "p_away", "result"]
MODEL = "logistic"


def read_log(path=LOG):
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"result": str}, keep_default_na=False).replace({"result": {"": None}})


def log_next_round(models, cols, state, log, today=None):
    """Return `log` plus predictions for the next round's fixtures that aren't in it yet."""
    name, games = next_round()
    seen = set(zip(log.date, log.home, log.away))
    new = [g for g in games if (g["date"], g["home"], g["away"]) not in seen]
    if not new:
        return log
    p = predict({MODEL: models[MODEL]}, cols, state, [(g["home"], g["away"]) for g in new],
                date=None if today is None else pd.Timestamp(today))[MODEL]
    rows = pd.DataFrame([dict(date=g["date"], round=name, home=g["home"], away=g["away"],
                              p_home=round(a, 4), p_draw=round(b, 4), p_away=round(c, 4), result=None)
                         for g, (a, b, c) in zip(new, p)], columns=COLUMNS)
    return pd.concat([log, rows], ignore_index=True)


def score_pending(log, matches):
    """Fill `result` (H/D/A) for logged fixtures that have since been played."""
    played = {(m["date"], m["home"], m["away"]): m["ft"] for m in matches if m["ft"]}
    log = log.copy()
    for i, r in log[log.result.isna()].iterrows():
        ft = played.get((r.date, r.home, r.away))
        if ft:
            log.at[i, "result"] = "H" if ft[0] > ft[1] else "A" if ft[0] < ft[1] else "D"
    return log


def summarise(log):
    """Accuracy, log loss and Brier score over the scored rows, next to a base-rate baseline."""
    done = log[log.result.notna()]
    if done.empty:
        return None
    y = done.result.map(LABELS).values
    p = done[["p_home", "p_draw", "p_away"]].astype(float).values
    base = np.bincount(y, minlength=3) / len(y)  # base rate of these same results, so a slightly generous baseline
    return dict(n=len(done), accuracy=float((p.argmax(1) == y).mean()),
                log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()),
                brier=brier(y, p), base_rate_brier=brier(y, np.tile(base, (len(y), 1))))


def save(log, path=LOG):
    Path(path).parent.mkdir(exist_ok=True)
    log.sort_values(["date", "home"]).to_csv(path, index=False)


def update(models, cols, state):
    """Score what has been played, log the next round, write the file; return the new log."""
    log = score_pending(read_log(), season_matches())
    log = log_next_round(models, cols, state, log)
    save(log)
    return log


def calibration(preds, bins=10):
    """Predicted vs observed frequency, pooling home/draw/away probabilities into equal-width bins.

    `preds` has p_home, p_draw, p_away and FTR (H/D/A), like results/logistic_test_predictions.csv.
    """
    p = preds[["p_home", "p_draw", "p_away"]].astype(float).values.ravel()
    hit = np.eye(3)[preds.FTR.map(LABELS).values].ravel()
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bucket": bucket, "predicted": p, "observed": hit})
    return df.groupby("bucket").agg(predicted=("predicted", "mean"), observed=("observed", "mean"),
                                    n=("observed", "size")).reset_index(drop=True)
