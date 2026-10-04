import numpy as np
import pandas as pd

from src.features import build_features


def _matches():
    rows = [
        ("2020-08-01", "A", "B", 3, 0, "H"),
        ("2020-08-08", "B", "A", 1, 1, "D"),
        ("2020-08-15", "A", "B", 0, 2, "A"),
    ]
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["HS", "AS", "HST", "AST"]:
        df[c] = 5
    df["season"] = "2021"
    return df


def test_first_match_has_no_form():
    feats, _ = build_features(_matches())
    assert np.isnan(feats.loc[0, "home_pts_5"])
    assert feats.loc[0, "home_elo"] == feats.loc[0, "away_elo"]


def test_features_only_use_earlier_matches():
    feats, _ = build_features(_matches())
    # Before match 3, A has a win and a draw: 2.0 points per game. Its own loss in match 3 must not count.
    assert feats.loc[2, "home_pts_5"] == 2.0
    assert feats.loc[2, "home_gf_5"] == 2.0


def test_elo_moves_toward_winner():
    feats, state = build_features(_matches())
    assert feats.loc[1, "away_elo"] > feats.loc[1, "home_elo"]  # A won match 1
    assert state.elo["B"] > feats.loc[2, "away_elo"]  # B won match 3
