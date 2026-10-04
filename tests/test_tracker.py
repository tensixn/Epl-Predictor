import math

import pandas as pd

from src.tracker import COLUMNS, score_pending, summarise


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
