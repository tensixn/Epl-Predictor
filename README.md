# EPL match outcome predictor

Machine learning model that predicts home win / draw / away win probabilities for Premier League matches.
It is trained on 26 seasons of results and tested walk-forward on the last three seasons, with no future data leaking into features.

**Live app: https://epl-predictor-tension.streamlit.app/**

## Project layout
```
data/raw/            season CSVs (2000/01 to 2025/26, plus 2026/27 so far)
scripts/             download_data.sh, fetch_current_season.py (season in progress), fetch_odds.py, log_predictions.py
src/features.py      Elo ratings and rolling form, built only from past matches
src/evaluate.py      walk-forward evaluation of the models
src/dixoncoles.py    Dixon-Coles goals model and the blend with logistic
src/predict.py       probabilities for any fixture
src/fixtures.py      next round and full season of fixtures from openfootball
src/simulate.py      Monte Carlo of the rest of the season (title, top 4, relegation odds), with Elo updated as results are simulated
src/tracker.py       logs predictions before kickoff and scores them afterwards
.github/workflows/   ci.yml runs pytest on every push; refresh.yml updates results and the log daily
app.py               Streamlit page on top of src/predict.py
tests/               leakage, Elo, fetch, simulation and tracker checks
results/             metrics, per-match test predictions (xgboost, logistic and blend; the blend's feeds the calibration chart), predictions_log.csv (live track record)
data/odds/           closing odds for the three test seasons, used only as a benchmark
```

## Data
`data/raw/season-XXYY.csv`: 26 seasons (2000/01 to 2025/26, 9,880 matches) from the
[datasets/football-datasets](https://github.com/datasets/football-datasets) mirror of
football-data.co.uk (the same source as datahub.io). Kaggle isn't reachable from the
build environment, so this replaces the Kaggle set from the reel. The columns are
results plus shots, shots on target, corners, fouls and cards. data/raw has no odds,
lineups or injuries.

The mirror only adds a season once it has finished, so the season in progress
(`season-2627.csv`) comes from [openfootball](https://github.com/openfootball/football.json),
which is updated after each matchday. It has scores and half-time scores only, so
points and goals form uses this season's matches while shots form falls back to each
team's latest matches that have shot data. Checked against the mirror on 2024/25,
all 380 openfootball results match. Refresh it with `python scripts/fetch_current_season.py`.

## Features (`src/features.py`)
Everything is computed from matches *before* kickoff:
- Elo rating per team (home advantage, goal-difference multiplier, 20% regression to the mean each summer, promoted teams start at 1420)
- Rolling averages over the last 5 and 10 matches: points, goals for/against, shots for/against, shots on target for/against
- Rest days, plus home−away differences of all of the above

## Models (`src/evaluate.py`)
- `base_rate`: always predicts historical H/D/A frequencies
- `elo_logistic`: multinomial logistic regression on Elo difference only
- `logistic`: logistic regression on all features
- `xgboost`: gradient-boosted trees on all features
- `dixon_coles`: Poisson goals model with attack/defence ratings, home advantage, the low-score correction and time decay (8-month half-life, refit every 4 weeks). Scores only, no form or shots
- `blend`: average of `logistic` and `dixon_coles`. This is the app's default and the model in the live track record

Walk-forward test: each of 2023/24, 2024/25 and 2025/26 is predicted by models trained only on earlier seasons.

| model | log loss | Brier | RPS | accuracy |
|---|---|---|---|---|
| base_rate | 1.0745 | 0.6507 | 0.2329 | 43.2% |
| elo_logistic | 0.9852 | 0.5879 | 0.2019 | 53.4% |
| logistic | 0.9791 | 0.5839 | 0.1993 | 53.2% |
| xgboost | 0.9877 | 0.5880 | 0.2010 | 53.3% |
| ensemble (logistic + xgboost average) | 0.9815 | 0.5847 | 0.1997 | 53.3% |
| dixon_coles | 0.9787 | 0.5832 | 0.2000 | 52.5% |
| blend (logistic + dixon_coles) | **0.9747** | **0.5807** | **0.1984** | 53.3% |
| bookmaker closing odds | **0.9597** | **0.5699** | **0.1938** | 55.0% |

The bookmaker row is market-average closing odds with the margin removed (`data/odds/`, from football-data.co.uk). It beats every model here, so that is the realistic ceiling. Averaging logistic and xgboost did not beat logistic alone, but averaging logistic with Dixon-Coles did. The decay rate was picked on these same three seasons, so the true gain is probably a little under the 0.004 shown. Odds can't be a model feature here: they only exist for past matches, and the fixture feed (openfootball) carries none.

## Run
```
pip install -r requirements.txt
python -m src.evaluate                       # metrics -> results/
python -m src.predict Arsenal Chelsea        # HOME AWAY pairs
pytest                                       # run the tests
streamlit run app.py                         # web page with team pickers
./scripts/download_data.sh                   # refresh data/raw (all seasons)
python scripts/fetch_current_season.py       # refresh just the current season
```
Team names follow football-data.co.uk spelling ("Man City", "Man United", "Nott'm Forest", "Spurs" is "Tottenham").

## Web app
`app.py` is a Streamlit page with four tabs: predictions for the next round of fixtures, a picker for any
home and away team, title / top 4 / relegation odds for the season, and the live track record. Each prediction
is a home / draw / away probability bar. The theme is in `.streamlit/config.toml`. The app trains its models from
`data/raw` when it starts (about 10 seconds, cached after that), so there is no separate training step. To host it on Streamlit Community Cloud, sign in at
share.streamlit.io with GitHub, choose this repo, branch `main` and main file path `app.py`.

## Ideas for next steps
- xG was tried (Understat, 2014/15 on, rolling 5/10-match xG for/against): it did not help. Logistic log loss went from 0.9791 to 0.9847 with all xG columns, and 0.9793 with only the 10-match differences, so it was dropped
