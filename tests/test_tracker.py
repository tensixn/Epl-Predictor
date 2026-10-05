import math

import pandas as pd

from src.tracker import COLUMNS, ODDS_COLUMNS, calibration, fill_odds, score_pending, summarise, vs_bookmaker


def log_of(*rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_score_pending_fills_only_played_matches():
    log = log_of(("2026-10-10", "R6", "Arsenal", "Leeds", 0.6, 0.2, 0.2, None),
                 ("2026-10-11", "R6", "Chelsea", "Fulham", 0.5, 0.3, 0.2, None))
    out = score_pending(log, [{"date": "2026-10-10", "home": "Arsenal", "away": "Leeds", "ft": [1, 1]},
                              {"date": "2026-10-11", "home": "Chelsea", "away": "Fulham", "ft": None}])
    assert list(out.result) == ["D", None]


def test_summarise_ignores_unscored_rows():
    log = log_of(("d1", "R", "A", "B", 0.7, 0.2, 0.1, "H"), ("d2", "R", "C", "D", 0.4, 0.3, 0.3, None))
    s = summarise(log)
    assert s["n"] == 1 and s["accuracy"] == 1.0 and abs(s["log_loss"] + math.log(0.7)) < 1e-9
    assert summarise(log.iloc[1:]) is None


def test_calibration_bins_predicted_against_observed():
    preds = pd.DataFrame({"p_home": [0.95, 0.06], "p_draw": [0.03, 0.04], "p_away": [0.02, 0.9],
                          "FTR": ["H", "A"]})
    cal = calibration(preds, bins=10)
    top = cal.iloc[-1]
    assert top.n == 2 and top.observed == 1.0 and abs(top.predicted - 0.925) < 1e-9
    assert cal.n.sum() == 6


def test_fill_odds_sets_missing_unplayed_rows_once_and_scores_against_bookmaker():
    log = pd.DataFrame([("d1", "R", "A", "B", 0.5, 0.3, 0.2, "H", None, None, None),
                        ("d2", "R", "C", "D", 0.5, 0.3, 0.2, None, None, None, None)], columns=COLUMNS + ODDS_COLUMNS)
    log[ODDS_COLUMNS] = log[ODDS_COLUMNS].astype(float)
    assert vs_bookmaker(log) is None
    out = fill_odds(log, {("d1", "A", "B"): (2.0, 4.0, 4.0), ("d2", "C", "D"): (1.5, 4.0, 6.0)})
    assert out.odds_h.isna().tolist() == [True, False]  # d1 is already played, so left alone
    out.loc[0, ODDS_COLUMNS] = (2.0, 4.0, 4.0)
    vb = vs_bookmaker(out)
    assert vb["n"] == 1 and abs(vb["bookmaker"] + math.log(0.5)) < 1e-9 and abs(vb["model"] + math.log(0.5)) < 1e-9
