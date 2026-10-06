"""Fetch the 2025/26 season from the dcaribou/transfermarkt-datasets snapshot, in the datalake's format.

The salimt datalake that build_player_data.py reads stops in September 2025. dcaribou's dataset has
the whole 2025/26 season (appearances to 28 June 2026, market values to 12 June 2026) but is hosted
on a Cloudflare R2 bucket our cloud sessions can't reach, so this runs from GitHub Actions
(.github/workflows/player-snapshot.yml) and commits small CSVs to data/players/snapshot/:

    player_performances.csv   one row per player, competition and club for 2025/26 (datalake columns)
    player_market_value.csv   values dated from 1 Sep 2025 for those players
    player_profiles.csv       profiles for those players

Only players with a 2025/26 Premier League appearance are kept. build_player_data.py uses these in
place of the datalake's 2025/26 rows. The snapshot is frozen (updates paused in July 2026), so this
only needs to run again if it resumes.

    python scripts/fetch_player_snapshot.py
"""
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "players" / "snapshot"
BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["games", "appearances", "game_lineups", "game_events", "player_valuations", "players", "clubs"]
SEASON = 2025  # dcaribou labels a season by its start year: 2025 = 2025/26
LABEL = "25/26"


def load(tmp):
    out = {}
    for t in TABLES:
        path = tmp / f"{t}.csv.gz"
        print("downloading", t, flush=True)
        urllib.request.urlretrieve(f"{BASE}/{t}.csv.gz", path)
        out[t] = pd.read_csv(path, low_memory=False)
        print(f"  {len(out[t])} rows: {list(out[t].columns)}", flush=True)
    return out


def performances(d):
    games = d["games"][d["games"].season == SEASON]
    app = d["appearances"].merge(games[["game_id", "home_club_id", "away_club_id", "home_club_goals",
                                        "away_club_goals"]], on="game_id")
    app = app.rename(columns={"player_club_id": "team_id"})
    epl_ids = app.loc[app.competition_id == "GB1", "player_id"].unique()
    app = app[app.player_id.isin(epl_ids)].copy()

    lineups = d["game_lineups"][d["game_lineups"].game_id.isin(games.game_id)]
    started = set(zip(*lineups[lineups.type == "starting_lineup"][["game_id", "player_id"]].values.T))
    app["started"] = [(g, p) in started for g, p in zip(app.game_id, app.player_id)]

    ev = d["game_events"]
    pens = ev[(ev.type == "Goals") & ev.description.fillna("").str.contains("enalty")]
    pens = pens.groupby(["game_id", "player_id"]).size().rename("penalty_goals")
    app = app.merge(pens, on=["game_id", "player_id"], how="left")

    # goalkeepers' clean sheets and goals conceded, as the datalake counts them (keepers only)
    keepers = set(d["players"].loc[d["players"].position == "Goalkeeper", "player_id"])
    home = app.team_id == app.home_club_id
    against = app.away_club_goals.where(home, app.home_club_goals)
    gk = app.player_id.isin(keepers)
    app["goals_conceded"] = against.where(gk, 0)
    app["clean_sheets"] = ((against == 0) & gk).astype(int)

    g = app.groupby(["player_id", "competition_id", "team_id"])
    perf = pd.DataFrame({
        "nb_on_pitch": g.size(),
        "goals": g.goals.sum(), "assists": g.assists.sum(),
        "subed_in": g.size() - g.started.sum(),
        "yellow_cards": g.yellow_cards.sum(),
        "penalty_goals": g.penalty_goals.sum(),
        "minutes_played": g.minutes_played.sum(),
        "goals_conceded": g.goals_conceded.sum(), "clean_sheets": g.clean_sheets.sum(),
    }).reset_index()
    perf["nb_in_group"] = perf.nb_on_pitch  # per-competition squad counts aren't in the snapshot
    clubs = d["clubs"].set_index("club_id").name
    perf["team_name"] = perf.team_id.map(clubs)
    perf["season_name"] = LABEL
    return perf, epl_ids


def profiles(d, ids):
    p = d["players"][d["players"].player_id.isin(ids)]
    return pd.DataFrame({
        "player_id": p.player_id, "player_name": p.name,
        "player_slug": p.url.str.extract(r"transfermarkt\.[^/]+/([^/]+)/", expand=False),
        "date_of_birth": pd.to_datetime(p.date_of_birth, errors="coerce").dt.date,
        "position": p.position + " - " + p.sub_position.fillna(p.position),
        "main_position": p.position, "height": p.height_in_cm, "foot": p.foot,
        "citizenship": p.country_of_citizenship,
    })


def values(d, ids):
    v = d["player_valuations"]
    v = v[v.player_id.isin(ids) & (pd.to_datetime(v.date) >= "2025-09-01")]
    return pd.DataFrame({"player_id": v.player_id, "date_unix": pd.to_datetime(v.date).dt.date,
                         "value": v.market_value_in_eur})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d = load(Path(tmp))
    perf, ids = performances(d)
    OUT.mkdir(parents=True, exist_ok=True)
    perf.to_csv(OUT / "player_performances.csv", index=False)
    profiles(d, ids).to_csv(OUT / "player_profiles.csv", index=False)
    vals = values(d, ids)
    vals.to_csv(OUT / "player_market_value.csv", index=False)
    epl = perf[perf.competition_id == "GB1"]
    print(f"2025/26 EPL: {epl.player_id.nunique()} players, {epl.nb_on_pitch.sum()} appearances, "
          f"{(epl.nb_on_pitch - epl.subed_in).sum()} starts, {epl.goals.sum()} goals; "
          f"{len(vals)} values, latest {vals.date_unix.max()}")


if __name__ == "__main__":
    main()
