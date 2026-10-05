"""Build data/players/epl_player_seasons.csv from the Transfermarkt datalake.

Source: github.com/salimt/football-datasets (Transfermarkt profiles, season
stats and market value history).  One row per player per Premier League
season they played in, with that season's and the previous season's stats and
the market value Transfermarkt gave them in the summer after the season.

    python scripts/build_player_data.py                # download, then build
    python scripts/build_player_data.py --raw DIR      # use CSVs already in DIR
"""
import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "players" / "epl_player_seasons.csv"
BASE = "https://github.com/salimt/football-datasets/raw/main/datalake/transfermarkt"  # redirects LFS files to the media host
FILES = {
    "player_performances.csv": "player_performances/player_performances.csv",
    "player_profiles.csv": "player_profiles/player_profiles.csv",
    "player_market_value.csv": "player_market_value/player_market_value.csv",
}
FIRST_SEASON = 2005  # 2004/05: Transfermarkt values start in late 2003
BIG5 = {"GB1", "ES1", "IT1", "L1", "FR1"}
EUROPE = {"CL", "EL", "UCOL", "UEFA"}

# Transfermarkt club name -> football-data.co.uk name used in data/raw
CLUBS = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton", "Burnley FC": "Burnley", "Chelsea FC": "Chelsea",
    "Everton FC": "Everton", "Fulham FC": "Fulham", "Liverpool FC": "Liverpool",
    "Manchester City": "Man City", "Manchester United": "Man United", "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham", "West Bromwich Albion": "West Brom", "Wolverhampton Wanderers": "Wolves",
    "Queens Park Rangers": "QPR", "Sunderland AFC": "Sunderland", "Watford FC": "Watford", "Reading FC": "Reading",
    "Portsmouth FC": "Portsmouth", "Middlesbrough FC": "Middlesbrough", "Southampton FC": "Southampton",
    "Blackpool FC": "Blackpool", "Birmingham City": "Birmingham", "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton", "Bradford City": "Bradford", "Cardiff City": "Cardiff",
    "Charlton Athletic": "Charlton", "Coventry City": "Coventry", "Derby County": "Derby",
    "Huddersfield Town": "Huddersfield", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Leicester City": "Leicester", "Luton Town": "Luton", "Newcastle United": "Newcastle",
    "Norwich City": "Norwich", "Stoke City": "Stoke", "Swansea City": "Swansea", "West Ham United": "West Ham",
    "Wigan Athletic": "Wigan",
}


def download(dest):
    for name, path in FILES.items():
        print("downloading", path)
        urllib.request.urlretrieve(f"{BASE}/{path}", dest / name)


def season_end(label):
    """'23/24' -> 2024, '99/00' -> 2000."""
    yy = int(label[3:])
    return 2000 + yy if yy < 90 else 1900 + yy


def club_table():
    """EPL points per game and final position per (club, season end year), from data/raw."""
    rows = []
    for f in sorted((ROOT / "data" / "raw").glob("season-*.csv")):
        m = pd.read_csv(f)
        end = 2000 + int(f.stem[-2:])
        if len(m) < 380:  # season in progress
            continue
        pts = {}
        for h, a, r in zip(m.HomeTeam, m.AwayTeam, m.FTR):
            pts[h] = pts.get(h, 0) + {"H": 3, "D": 1, "A": 0}[r]
            pts[a] = pts.get(a, 0) + {"H": 0, "D": 1, "A": 3}[r]
        table = sorted(pts, key=pts.get, reverse=True)
        rows += [{"club": c, "season": end, "team_ppg": pts[c] / 38, "team_rank": table.index(c) + 1} for c in pts]
    return pd.DataFrame(rows)


def stat_totals(perf, mask, prefix):
    cols = {"nb_in_group": "squad", "nb_on_pitch": "apps", "starts": "starts", "goals": "goals",
            "assists": "assists", "penalty_goals": "pens", "yellow_cards": "yellows", "clean_sheets": "clean_sheets",
            "goals_conceded": "conceded"}
    g = perf[mask].groupby(["player_id", "season"])[list(cols)].sum()
    return g.rename(columns={c: f"{prefix}{n}" for c, n in cols.items()})


def build(raw):
    perf = pd.read_csv(raw / "player_performances.csv", low_memory=False)
    perf["season"] = perf.season_name.map(season_end)
    num = ["nb_in_group", "nb_on_pitch", "subed_in", "goals", "assists", "penalty_goals", "yellow_cards",
           "clean_sheets", "goals_conceded"]
    perf[num] = perf[num].fillna(0)
    # minutes_played is unusable (missing for over half the rows, garbled above 999), so starts stand in
    perf["starts"] = (perf.nb_on_pitch - perf.subed_in).clip(lower=0)
    epl_ids = perf.loc[perf.competition_id == "GB1", "player_id"].unique()
    perf = perf[perf.player_id.isin(epl_ids)]

    epl = stat_totals(perf, perf.competition_id == "GB1", "epl_")
    epl = epl[epl.epl_apps > 0]
    allc = stat_totals(perf, perf.nb_on_pitch >= 0, "all_")[["all_apps", "all_starts", "all_goals", "all_assists"]]
    eur = perf[perf.competition_id.isin(EUROPE)].groupby(["player_id", "season"]).nb_on_pitch.sum().rename("europe_apps")
    ucl = perf[perf.competition_id == "CL"].groupby(["player_id", "season"]).nb_on_pitch.sum().rename("ucl_apps")
    big5 = perf[perf.competition_id.isin(BIG5)].groupby(["player_id", "season"])[["nb_on_pitch", "starts"]].sum()

    prev = allc.join(big5.rename(columns={"nb_on_pitch": "big5_apps", "starts": "big5_starts"}), how="outer").fillna(0)
    prev = prev.add_prefix("prev_").reset_index()
    prev["season"] += 1

    # career big-5 league starts before the season (experience)
    car = big5.starts.rename("career_big5_starts").reset_index().sort_values("season")
    car["career_big5_starts"] = car.groupby("player_id").career_big5_starts.cumsum()
    car["season"] += 1  # cumulative up to and including the previous season

    df = epl.join(allc).join(eur).join(ucl).reset_index()
    df = df.merge(prev, on=["player_id", "season"], how="left")
    df = pd.merge_asof(df.sort_values("season"), car.sort_values("season"), on="season", by="player_id")
    df[[c for c in df if c.startswith(("prev_", "europe", "ucl", "career"))]] = \
        df[[c for c in df if c.startswith(("prev_", "europe", "ucl", "career"))]].fillna(0)

    # club = the EPL side the player made most league appearances for
    e = perf[perf.competition_id == "GB1"].sort_values("nb_on_pitch")
    club = e.groupby(["player_id", "season"]).team_name.last().map(CLUBS).rename("club").reset_index()
    df = df.merge(club, on=["player_id", "season"]).merge(club_table(), on=["club", "season"], how="left")

    prof = pd.read_csv(raw / "player_profiles.csv", low_memory=False)
    prof = prof[prof.player_id.isin(epl_ids)]
    prof["name"] = prof.player_name.str.replace(r"\s*\(\d+\)$", "", regex=True).fillna(
        prof.player_slug.str.replace("-", " ").str.title())
    prof["sub_position"] = prof.position.str.split(" - ").str[1].fillna(prof.main_position)
    prof["height"] = prof.height.replace(0, np.nan)
    prof["dob"] = pd.to_datetime(prof.date_of_birth, errors="coerce")
    df = df.merge(prof[["player_id", "name", "dob", "main_position", "sub_position", "height", "foot", "citizenship"]],
                  on="player_id", how="left")
    df["age"] = (pd.to_datetime(df.season.astype(str) + "-07-01") - df.dob).dt.days / 365.25

    # target: last Transfermarkt value dated 1 Jan - 31 Aug after the season ends;
    # prev_value: the last value before the season started (for reference only, not a model feature)
    mv = pd.read_csv(raw / "player_market_value.csv")
    mv = mv[mv.player_id.isin(epl_ids) & (mv.value > 0)]
    mv["date"] = pd.to_datetime(mv.date_unix)
    mv = mv.sort_values("date")

    def value_at(when, earliest):
        q = df[["player_id"]].assign(date=when).reset_index().sort_values("date")
        r = pd.merge_asof(q, mv[["player_id", "date", "value"]].rename(columns={"date": "vdate"}),
                          left_on="date", right_on="vdate", by="player_id")
        r.loc[r.vdate < earliest.reindex(r["index"]).values, "value"] = np.nan
        return r.set_index("index").value.sort_index()

    df = df.reset_index(drop=True)
    end = pd.to_datetime(df.season.astype(str) + "-08-31")
    df["value"] = value_at(end, pd.to_datetime(df.season.astype(str) + "-01-01"))
    start = pd.to_datetime((df.season - 1).astype(str) + "-08-31")
    df["prev_value"] = value_at(start, pd.to_datetime((df.season - 2).astype(str) + "-09-01"))

    # the newest season in the datalake is only a few matchdays old, and has no summer value yet
    df = df[(df.season >= FIRST_SEASON) & (df.season < df.season.max()) & df.dob.notna() & df.club.notna()]
    df = df.drop(columns="dob").sort_values(["season", "club", "name"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, help="folder that already holds the Transfermarkt CSVs")
    args = ap.parse_args()
    if args.raw:
        df = build(args.raw)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            download(Path(tmp))
            df = build(Path(tmp))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4g")
    print(f"{len(df)} player-seasons, {df.value.notna().sum()} with a value, seasons "
          f"{df.season.min()}-{df.season.max()} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
