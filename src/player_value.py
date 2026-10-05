"""Player value analyzer: what a Premier League player's season says their market value should be.

The model sees age, position, height, foot, this season's and last season's
appearances / starts / goals / assists, European games, big-5 league
experience and the club's league finish. It never sees any market value. Its
estimate (the "stats value") is compared with Transfermarkt's value from the
summer after the season: a player valued well above their stats value is
priced for something the stats don't show (reputation, potential, contract),
one well below it is cheap for what they did on the pitch.

Values are modelled relative to that summer's median EPL value, so transfer
inflation doesn't swamp the model; each estimate is scaled back to euros with
the same summer's median.

Walk-forward like the match model: each season is predicted by a model fitted
only on earlier seasons.  Run from the project root:

    python -m src.player_value
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "players" / "epl_player_seasons.csv"
OUT_DIR = ROOT / "results"
TEST_SEASONS = [2023, 2024, 2025]  # 2022/23 to 2024/25, the three latest seasons with summer values
FIRST_SCORED = 2010  # seasons from here on get out-of-sample estimates in player_values.csv
COUNT_PREFIXES = ("epl_", "all_", "prev_", "europe_", "ucl_", "career_")
POSITIONS = ["Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield",
             "Attacking Midfield", "Left Midfield", "Right Midfield", "Left Winger", "Right Winger",
             "Second Striker", "Centre-Forward"]


def load_players(path=DATA):
    return pd.read_csv(path)


def count_columns(df):
    return [c for c in df if c.startswith(COUNT_PREFIXES) and c != "prev_value"]


def feature_matrix(df):
    """Model inputs. Built only from stats and profile, never from a market value."""
    counts = count_columns(df)
    X = df[counts].clip(lower=0).astype(float)
    X = X.join(np.log1p(X).add_prefix("log_"))
    X["age"] = df.age
    X["age_sq"] = (df.age - 26) ** 2
    X["height"] = df.height
    X["left_foot"] = (df.foot == "left").astype(float)
    X["team_ppg"] = df.team_ppg
    X["team_rank"] = df.team_rank
    X["start_share"] = df.epl_starts / 38
    X["goal_involvements_per_app"] = (df.epl_goals + df.epl_assists) / df.epl_apps.clip(lower=1)
    for p in ("Goalkeeper", "Defender", "Midfield", "Attack"):
        X[f"is_{p.lower()}"] = (df.main_position == p).astype(float)
    for p in POSITIONS:
        X[f"pos_{p.lower().replace(' ', '_').replace('-', '_')}"] = (df.sub_position == p).astype(float)
    return X


def relative_target(df):
    """log(value / median value of the same summer)."""
    return np.log(df.value) - np.log(df.groupby("season").value.transform("median"))


class MedianByAgePosition:
    """Baseline: median relative value of players of the same position and age band."""

    def fit(self, df, y):
        self.table = y.groupby(self._key(df)).median()
        self.overall = y.median()
        return self

    def predict(self, df):
        return self._key(df).map(self.table).fillna(self.overall).values

    @staticmethod
    def _key(df):
        band = pd.cut(df.age, [0, 21, 24, 27, 30, 33, 60], labels=False).astype(str)
        return df.main_position.fillna("?") + "|" + band


class FeatureModel:
    def __init__(self, factory):
        self.factory = factory

    def fit(self, df, y):
        X = feature_matrix(df)
        self.fill = X.median()
        self.model = self.factory().fit(X.fillna(self.fill), y)
        return self

    def predict(self, df):
        return self.model.predict(feature_matrix(df).fillna(self.fill))


def make_models():
    return {
        "age_position": MedianByAgePosition,
        "ridge": lambda: FeatureModel(lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))),
        "xgboost": lambda: FeatureModel(lambda: XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, random_state=0)),
    }


def predict_season(df, season, name="xgboost"):
    """Relative-value estimates for `season`'s players from a model fitted on earlier seasons only."""
    known = df[df.value.notna()]
    train = known[known.season < season]
    test = df[df.season == season]
    model = make_models()[name]().fit(train, relative_target(train))
    return pd.Series(model.predict(test), index=test.index)


def score(season, name, y, pred):
    err = pred - y
    return dict(season=season, model=name, n=len(y),
                mae_log=float(np.mean(np.abs(err))),
                r2_log=float(1 - np.mean(err ** 2) / np.var(y)),
                median_pct_error=float(np.median(np.abs(np.exp(err) - 1))),
                within_25pct=float(np.mean(np.abs(np.exp(err) - 1) <= 0.25)))


def evaluate(df):
    """Walk-forward metrics for every model on TEST_SEASONS."""
    rows = []
    known = df[df.value.notna()]
    for season in TEST_SEASONS:
        test = known[known.season == season]
        y = relative_target(known)[test.index]
        for name in make_models():
            rows.append(score(season, name, y.values, predict_season(known, season, name)[test.index].values))
    return pd.DataFrame(rows)


def stats_values(df, seasons):
    """Out-of-sample stats value in euros for every player in `seasons`, next to their market value."""
    out = []
    for season in seasons:
        rel = predict_season(df, season)
        part = df.loc[rel.index]
        median = part.value.median()
        out.append(part.assign(stats_value=np.exp(rel) * median))
    res = pd.concat(out)
    res["gap"] = res.value / res.stats_value - 1  # +50% = the market prices the player 50% above their stats
    cols = ["season", "player_id", "name", "club", "age", "main_position", "sub_position", "citizenship",
            "epl_apps", "epl_starts", "epl_goals", "epl_assists", "epl_clean_sheets", "europe_apps", "team_rank",
            "value", "prev_value", "stats_value", "gap"]
    return res[cols]


def main():
    df = load_players()
    metrics = evaluate(df)
    OUT_DIR.mkdir(exist_ok=True)
    metrics.to_csv(OUT_DIR / "player_value_metrics.csv", index=False, float_format="%.4f")
    summary = metrics.groupby("model")[["mae_log", "r2_log", "median_pct_error", "within_25pct"]].mean()
    (OUT_DIR / "player_value_summary.json").write_text(json.dumps(summary.round(4).to_dict("index"), indent=2))
    print(summary.round(3).to_string())

    seasons = sorted(s for s in df.season.unique() if s >= FIRST_SCORED)
    vals = stats_values(df, seasons)
    vals.to_csv(OUT_DIR / "player_values.csv", index=False, float_format="%.4g")
    print(f"{len(vals)} player-season estimates, {seasons[0]}-{seasons[-1]} -> results/player_values.csv")


if __name__ == "__main__":
    main()
