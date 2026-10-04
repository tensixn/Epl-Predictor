"""Build pre-match features from historical EPL results.

Every feature for a match uses only matches played before it, so the model
never sees the result it is trying to predict.
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

ELO_START = 1500.0
ELO_PROMOTED = 1420.0  # teams new to the league start below average
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_SEASON_REGRESS = 0.2  # pull ratings 20% back toward the mean each summer
WINDOWS = (5, 10)
STATS = ["pts", "gf", "ga", "sf", "sa", "stf", "sta"]  # points, goals, shots, shots on target


def load_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob("season-*.csv")):
        df = pd.read_csv(path)
        df["season"] = path.stem.split("-")[1]  # e.g. "2425"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)


def _elo_expected(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_away - r_home - ELO_HOME_ADV) / 400.0))


class TeamState:
    """Running per-team history used to compute features."""

    def __init__(self):
        self.elo = {}
        self.history = defaultdict(lambda: deque(maxlen=max(WINDOWS)))
        self.last_date = {}
        self.season = None

    def new_season(self, teams):
        if self.elo:
            mean = np.mean(list(self.elo.values()))
            self.elo = {t: r + ELO_SEASON_REGRESS * (mean - r) for t, r in self.elo.items()}
        for t in teams:
            self.elo.setdefault(t, ELO_PROMOTED if self.season else ELO_START)

    def team_features(self, team, date):
        f = {"elo": self.elo.get(team, ELO_PROMOTED)}
        hist = list(self.history[team])
        for w in WINDOWS:
            recent = hist[-w:]
            for s in STATS:
                f[f"{s}_{w}"] = np.mean([m[s] for m in recent]) if recent else np.nan
        last = self.last_date.get(team)
        f["rest_days"] = min((date - last).days, 30) if last is not None else 30
        return f

    def match_features(self, home, away, date):
        h = self.team_features(home, date)
        a = self.team_features(away, date)
        row = {f"home_{k}": v for k, v in h.items()}
        row.update({f"away_{k}": v for k, v in a.items()})
        for k in h:
            row[f"diff_{k}"] = h[k] - a[k]
        row["elo_home_win_exp"] = _elo_expected(h["elo"], a["elo"])
        return row

    def update(self, m):
        home, away = m.HomeTeam, m.AwayTeam
        hg, ag = m.FTHG, m.FTAG
        # Elo update with a goal-difference multiplier
        exp = _elo_expected(self.elo[home], self.elo[away])
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mult = np.log(abs(hg - ag) + 1) + 1
        delta = ELO_K * mult * (actual - exp)
        self.elo[home] += delta
        self.elo[away] -= delta

        h_pts = 3 if hg > ag else 1 if hg == ag else 0
        a_pts = 3 if ag > hg else 1 if hg == ag else 0
        self.history[home].append(dict(pts=h_pts, gf=hg, ga=ag, sf=m.HS, sa=m.AS, stf=m.HST, sta=m.AST))
        self.history[away].append(dict(pts=a_pts, gf=ag, ga=hg, sf=m.AS, sa=m.HS, stf=m.AST, sta=m.HST))
        self.last_date[home] = self.last_date[away] = m.Date


def build_features(matches: pd.DataFrame):
    """Return (feature table, final TeamState) for all matches in order."""
    state = TeamState()
    rows = []
    for season, games in matches.groupby("season", sort=False):
        state.new_season(set(games.HomeTeam) | set(games.AwayTeam))
        state.season = season
        for m in games.itertuples(index=False):
            row = state.match_features(m.HomeTeam, m.AwayTeam, m.Date)
            row.update(Date=m.Date, season=season, HomeTeam=m.HomeTeam,
                       AwayTeam=m.AwayTeam, FTR=m.FTR)
            rows.append(row)
            state.update(m)
    return pd.DataFrame(rows), state


META_COLS = ["Date", "season", "HomeTeam", "AwayTeam", "FTR"]
LABELS = {"H": 0, "D": 1, "A": 2}


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS]
