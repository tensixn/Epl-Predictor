"""Download bookmaker closing odds for the walk-forward test seasons to data/odds/.

Odds only exist for past matches, so they are a benchmark for src/evaluate.py, not a model feature.

    python scripts/fetch_odds.py
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "odds"
URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
SEASONS = ["2324", "2425", "2526"]
# market average closing odds, falling back to Bet365 closing odds
ODDS = [("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        with urllib.request.urlopen(URL.format(season=season), timeout=30) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()), encoding="utf-8-sig")
        cols = next(c for c in ODDS if set(c) <= set(df.columns))
        out = df[["Date", "HomeTeam", "AwayTeam", *cols]].dropna()
        out.columns = ["Date", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]
        out["Date"] = pd.to_datetime(out["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")
        out.to_csv(OUT_DIR / f"season-{season}.csv", index=False)
        print(f"season-{season}.csv: {len(out)} matches ({cols[0]})")


if __name__ == "__main__":
    main()
