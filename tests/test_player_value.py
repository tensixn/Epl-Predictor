import numpy as np
import pandas as pd

from src import player_value as pv


def players():
    return pv.load_players()


def test_features_never_see_a_market_value():
    X = pv.feature_matrix(players())
    assert not [c for c in X if "value" in c]


def test_features_are_unchanged_by_market_values():
    df = players()
    shuffled = df.assign(value=np.random.default_rng(0).permutation(df.value.values),
                         prev_value=df.prev_value[::-1].values)
    pd.testing.assert_frame_equal(pv.feature_matrix(df), pv.feature_matrix(shuffled))


def test_season_is_predicted_from_earlier_seasons_only():
    df = players()
    season = 2020
    base = pv.predict_season(df, season, "ridge")
    # scrambling later seasons' values must not change the estimates
    later = df.season >= season
    scrambled = df.copy()
    scrambled.loc[later, "value"] = scrambled.loc[later, "value"].sample(frac=1, random_state=1).values
    np.testing.assert_allclose(base.values, pv.predict_season(scrambled, season, "ridge").values)


def test_one_row_per_player_season():
    df = players()
    assert not df.duplicated(["player_id", "season"]).any()
    assert df.age.between(15, 45).all()
