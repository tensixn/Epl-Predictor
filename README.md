# Football predictor

Two tools on one page: a match predictor and a player value analyzer.
The match predictor is a machine learning model that predicts home win / draw / away win probabilities for Premier League matches.
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
src/player_value.py  player value analyzer: stats-based value vs Transfermarkt market value
scripts/build_player_data.py  builds data/players/ from the Transfermarkt datalake
.github/workflows/   ci.yml runs pytest on every push; refresh.yml updates results and the log daily
app.py               Streamlit page on top of src/predict.py
tests/               leakage, Elo, fetch, simulation and tracker checks
results/             metrics, per-match test predictions (xgboost, logistic and blend; the blend's feeds the calibration chart), predictions_log.csv (live track record)
data/players/        one row per EPL player-season (2004/05 to 2025/26): stats, profile, summer market value
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
`app.py` is a Streamlit page with two sections, switched at the top. **Match predictor** has four tabs: predictions
for the next round of fixtures (kick-off times are shown in the viewer's own time zone), a picker for any home and
away team, title / top 4 / relegation odds for the season, and the live track record. Each prediction is a home /
draw / away probability bar. **Player values** compares each player's Transfermarkt price with what their season says
they're worth, by season, club and position, with a plain-language verdict from Bargain to Overpriced. The look
(a dark "matchday broadcast" style) is the CSS block at the top of `app.py` plus the theme in `.streamlit/config.toml`.
The app trains its models from `data/raw` when it starts (about 10 seconds, cached after that), so there is no
separate training step. To host it on Streamlit Community Cloud, sign in at share.streamlit.io with GitHub, choose
this repo, branch `main` and main file path `app.py`.

## Player value analyzer (`src/player_value.py`)
The second idea from the reel: what should a player be worth, judging only by their season?
For every Premier League player-season since 2004/05, an XGBoost model estimates the player's
Transfermarkt value in the summer after the season from age, position, height, foot, league and
all-competition appearances, starts, goals and assists (this season and last), European games,
big-5 league experience and the club's league finish. It never sees a market value, so the gap
between the market value and this "stats value" shows who the market prices above or below what
they did on the pitch. Values are modelled relative to that summer's median EPL value so transfer
inflation doesn't dominate.

Data: [salimt/football-datasets](https://github.com/salimt/football-datasets), a Transfermarkt
datalake on GitHub, for 2004/05 to 2024/25 (its values stop in September 2025), plus 2025/26 from the
[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) snapshot
(appearances to June 2026, values to 12 June 2026). That snapshot sits on a bucket the cloud build
environment can't reach, so `scripts/fetch_player_snapshot.py` runs from GitHub Actions (the "player
snapshot" workflow, run by hand) and commits the 2025/26 rows to `data/players/snapshot/`. The snapshot's
updates are paused, so 2025/26 with summer 2026 values is the latest season. Transfermarkt itself, FBref and
Kaggle are blocked from the build environment. The datalake's minutes column is missing for over half the
rows, so starts stand in. Rebuild with `python scripts/build_player_data.py`.

Walk-forward test, each of 2023/24, 2024/25 and 2025/26 predicted by models trained only on earlier seasons:

| model | median error | within 25% | variance explained (log value) |
|---|---|---|---|
| median by age band and position | 57% | 21% | 24% |
| xgboost | **35%** | **37%** | **75%** |

A typical estimate is about a third off the market value. 2025/26 is the hardest of
the three (39%). Its rows come from the other source, which leaves out a few minor competitions (youth
leagues) the datalake counts, so that may explain some of it. Adding the player's previous market
value as a feature cuts that to about 21%, but then the model mostly repeats last year's price, so it
is left out on purpose. `results/player_values.csv` holds out-of-sample stats values for 2009/10 to
2025/26, which the app's Player values tab reads.

```
python -m src.player_value                   # metrics + player_values.csv -> results/
```

## AI player scout (`src/scout.py`)
The third idea from the reel: who should a club sign? The app's Scout section takes a club, a
budget and an age limit, and works in three steps:

1. **Squad needs.** For each position (goalkeeper, centre-back, full-back, defensive, central and
   attacking midfield, winger, striker) it takes the starts-weighted stats value of the players who
   played there and compares it with the average of that season's top six. Lowest first.
2. **Shortlist.** Players at other Premier League clubs in the chosen positions, priced within the
   budget, no older than the limit, with 15+ starts and a higher stats value than the club's current
   players there.
3. **Scouting report.** Claude (`claude-opus-5-5`) gets both tables and picks up to three signings
   from the shortlist only, backing each with the numbers it was given.

Steps 1 and 2 need nothing extra. The report needs an Anthropic API key: on Streamlit Community
Cloud add `ANTHROPIC_API_KEY = "..."` under the app's Settings → Secrets; locally put the same line
in `.streamlit/secrets.toml` (git-ignored) or set the environment variable. Each report is one API
call, and the app caches it so the same club, budget and shortlist don't call twice.

It uses the player value data, so it scouts on 2025/26 stats and summer 2026 prices, and only
Premier League players; some have moved since.

### This season's form
FPL is the only reachable source for 2026/27 player stats, so `scripts/fetch_fpl.py` pulls the
Fantasy Premier League API from GitHub Actions (the daily refresh) into `data/players/fpl_current.csv`:
minutes, starts, goals, assists, expected goals and assists, availability and current club for every
Premier League squad player. `src/form.py` matches those players to ours by name (FPL often uses full
legal names, e.g. "Bruno Borges Fernandes"), and only takes unambiguous matches: 403 of the 537
2025/26 players match; most of the rest have left the league. The Scout adds the "now" columns to the
shortlist and Claude's prompt, and leaves out players who have since joined the club you're scouting
for. FPL has no market values, so prices stay at summer 2026.

## Ideas for next steps
- xG was tried (Understat, 2014/15 on, rolling 5/10-match xG for/against): it did not help. Logistic log loss went from 0.9791 to 0.9847 with all xG columns, and 0.9793 with only the 10-match differences, so it was dropped
- Injuries were tried (Transfermarkt injury dates, 2008/09 on): the share of a club's previous-season starters out on match day, as home, away and difference columns. It did not help, within ±0.001 log loss on 2023/24 and 2024/25 for logistic and xgboost, so it was dropped. The data also ends in December 2025, so it couldn't feed live predictions anyway
- Retuning the blend weight is not worth it: 50/50 is already the best, and 0.4 to 0.6 are within 0.0002 of each other
