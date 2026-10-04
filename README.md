# EPL match outcome predictor

Machine learning model that predicts home win / draw / away win probabilities for Premier League matches.
It is trained on 26 seasons of results and tested walk-forward on the last three seasons, with no future data leaking into features.

## Project layout
```
data/raw/            season CSVs (2000/01 to 2025/26)
scripts/             download_data.sh to refresh the data
src/features.py      Elo ratings and rolling form, built only from past matches
src/evaluate.py      walk-forward evaluation of four models
src/predict.py       probabilities for any fixture
app.py               Streamlit page on top of src/predict.py
tests/               leakage and Elo sanity checks
results/             metrics and per-match test predictions
```

## Data
`data/raw/season-XXYY.csv`: 26 seasons (2000/01 to 2025/26, 9,880 matches) from the
[datasets/football-datasets](https://github.com/datasets/football-datasets) mirror of
football-data.co.uk (the same source as datahub.io). Kaggle isn't reachable from the
build environment, so this replaces the Kaggle set from the reel. The columns are
results plus shots, shots on target, corners, fouls and cards. There are no odds,
lineups or injuries. The 2026/27 season isn't in the mirror yet.

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

Walk-forward test: each of 2023/24, 2024/25 and 2025/26 is predicted by models trained only on earlier seasons.

| model | log loss | Brier | RPS | accuracy |
|---|---|---|---|---|
| base_rate | 1.0745 | 0.6507 | 0.2329 | 43.2% |
| elo_logistic | 0.9852 | 0.5879 | 0.2019 | 53.4% |
| logistic | **0.9791** | **0.5839** | **0.1993** | 53.2% |
| xgboost | 0.9879 | 0.5882 | 0.2012 | 53.5% |

Bookmaker closing odds usually score about 0.96 to 0.97 log loss on the EPL, so that is the realistic ceiling.

## Run
```
pip install -r requirements.txt
python -m src.evaluate                       # metrics -> results/
python -m src.predict Arsenal Chelsea        # HOME AWAY pairs
pytest                                       # run the tests
streamlit run app.py                         # web page with team pickers
./scripts/download_data.sh                   # refresh data/raw
```
Team names follow football-data.co.uk spelling ("Man City", "Man United", "Nott'm Forest", "Spurs" is "Tottenham").

## Web app
`app.py` is a Streamlit page: pick a home and away team and it shows the win, draw and loss
probabilities. It trains both models from `data/raw` when it starts (about 10 seconds, cached after that),
so there is no separate training step. To host it on Streamlit Community Cloud, sign in at
share.streamlit.io with GitHub, choose this repo, branch `main` and main file path `app.py`.

## Ideas for next steps
- Add bookmaker odds or xG as features
- Pull in current-season results so predictions use live form
