"""Streamlit page for the EPL predictor.

Run locally from the project root:  streamlit run app.py
"""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.features import load_matches
from src.fixtures import next_round, season_matches
from src.predict import predict, train
from src.simulate import simulate_season
from src.tracker import calibration, read_log, summarise

st.set_page_config(page_title="EPL match predictor", page_icon=str(Path(__file__).parent / "assets" / "favicon.png"),
                   layout="centered")

st.markdown("""
<style>
:root { --home:#3ddc97; --draw:#8b7a91; --away:#d6336c; --ink:#1d0a22; --rule:rgba(245,240,247,.16); }
[data-testid="stHeader"] { background:transparent; }
.stApp { background-image: radial-gradient(60rem 28rem at 12% -8%, rgba(132,72,150,.20), transparent 70%); }
h1 { letter-spacing:-.025em; text-wrap:balance; }
h2, h3 { letter-spacing:-.015em; font-weight:600; text-wrap:balance; }
[data-testid="stCaptionContainer"] { text-wrap:pretty; }
.pbar { display:flex; height:34px; border-radius:8px; overflow:hidden; font-weight:600; font-size:.9rem; }
.pbar span { display:flex; align-items:center; justify-content:center; min-width:0; white-space:nowrap; }
.pbar .h { background:var(--home); color:var(--ink); }
.pbar .d { background:var(--draw); color:var(--ink); }
.pbar .a { background:var(--away); color:#fff; }
.legend { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.2rem 1rem; font-size:.85rem; margin:.4rem 0 .8rem; }
.legend i { display:inline-block; width:.7rem; height:.7rem; border-radius:3px; margin-right:.4rem; }
.legend .h i { background:var(--home); } .legend .d i { background:var(--draw); } .legend .a i { background:var(--away); }
.pbar, .legend, .fx { font-variant-numeric: tabular-nums; }
.day { margin:1.8rem 0 .3rem; padding-bottom:.35rem; font-weight:600; border-bottom:1px solid var(--rule); }
.fx { display:flex; justify-content:space-between; align-items:baseline; margin:1rem 0 .4rem; font-weight:500; }
.fx small { font-weight:400; opacity:.75; }
::selection { background:var(--home); color:var(--ink); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training on every season in data/raw...")
def load():
    matches = load_matches()
    models, cols, state = train(matches=matches)
    latest = matches[matches.season == matches.season.max()]
    current = sorted(set(latest.HomeTeam) | set(latest.AwayTeam))
    return models, cols, state, current


@st.cache_data(ttl=3600, show_spinner="Fetching upcoming fixtures...")
def fixtures():
    return next_round()


@st.cache_data(ttl=3600, show_spinner="Simulating the rest of the season...")
def season_odds(model):
    return simulate_season(models, cols, state, season_matches(), model=model)


def bar(p_home, p_draw, p_away, home, away):
    """Stacked home / draw / away bar. The legend names each colour, so colour isn't the only cue."""
    ps = (p_home, p_draw, p_away)
    tight = [f" {p:.0%}" if p < 0.08 else "" for p in ps]  # no label fits inside a thin segment
    spans = "".join(f'<span class="{c}" style="flex:{p:.4f}">{p:.0%}</span>' if p >= 0.08
                    else f'<span class="{c}" style="flex:{p:.4f}"></span>' for c, p in zip("hda", ps))
    return (f'<div class="pbar" role="img" aria-label="{escape(home)} {p_home:.0%}, draw {p_draw:.0%}, '
            f'{escape(away)} {p_away:.0%}">{spans}</div>'
            f'<div class="legend"><span class="h"><i></i>{escape(home)}{tight[0]}</span>'
            f'<span class="d"><i></i>Draw{tight[1]}</span>'
            f'<span class="a"><i></i>{escape(away)}{tight[2]}</span></div>')


models, cols, state, current = load()
form_to = max(state.last_date.values())

st.title("EPL match predictor")
st.caption(f"Elo ratings and form run to {form_to:%d %b %Y}, the last match in the data.")

model = st.radio("Model", ["blend", "logistic", "xgboost"], horizontal=True,
                 format_func={"blend": "Blend", "logistic": "Logistic regression", "xgboost": "XGBoost"}.get,
                 help="The blend averages logistic regression with a Dixon-Coles goals model. It scored best in the "
                      "walk-forward test (log loss 0.975 vs 0.979 for logistic alone, 0.988 for XGBoost).")

tab_next, tab_pick, tab_season, tab_track = st.tabs(
    ["Upcoming fixtures", "Pick a match", "Season odds", "Track record"])

with tab_next:
    try:
        round_name, games = fixtures()
    except Exception as e:  # network or an unfamiliar team name; the other tab still works
        round_name, games = None, []
        st.warning(f"Couldn't load fixtures ({type(e).__name__}). Use the other tab.")
    if round_name:
        st.subheader(round_name)
        for team in {t for g in games for t in (g["home"], g["away"])} - set(state.elo):
            st.info(f"{team} isn't in the data yet; treated as a newly promoted side.")
        probs = predict({model: models[model]}, cols, state, [(g["home"], g["away"]) for g in games])[model]
        day = None
        for g, (ph, pd_, pa) in zip(games, probs):
            if g["date"] != day:
                day = g["date"]
                st.markdown(f'<div class="day">{pd.Timestamp(day):%A %d %B}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="fx"><span>{escape(g["home"])} v {escape(g["away"])}</span>'
                        f'<small>{escape(g["time"])}</small></div>' + bar(ph, pd_, pa, g["home"], g["away"]),
                        unsafe_allow_html=True)
    elif round_name is None and not games:
        st.info("No upcoming fixtures found.")

with tab_pick:
    all_teams = st.checkbox("Show every team since 2000/01", value=False)
    teams = sorted(state.elo) if all_teams else current
    left, right = st.columns(2)
    home = left.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away = right.selectbox("Away team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        p_home, p_draw, p_away = predict({model: models[model]}, cols, state, [(home, away)])[model][0]
        st.markdown(bar(p_home, p_draw, p_away, home, away), unsafe_allow_html=True)
        with st.expander("Team form behind this prediction"):
            rows = []
            for t in (home, away):
                f = state.team_features(t, form_to)
                rows.append({"team": t, "Elo": round(f["elo"]), "points per game (last 5)": f["pts_5"],
                             "goals for (last 5)": f["gf_5"], "goals against (last 5)": f["ga_5"]})
            st.dataframe(rows, hide_index=True)

with tab_season:
    st.caption("10,000 simulations of the remaining fixtures. Each simulated result moves both teams' Elo before "
               "the next game is drawn, so a hot streak carries on. Form (last 5 / 10 matches) stays at today's values.")
    try:
        odds = season_odds(model)
        odds[["title", "top 4", "relegation"]] *= 100
        chance = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.1f%%")
        st.dataframe(odds, hide_index=True, width="stretch", height="content", column_config={
            "team": "Team",
            "points now": st.column_config.NumberColumn("Points", width="small"),
            "expected points": st.column_config.NumberColumn("Projected", format="%.1f", width="small"),
            "title": chance("Title"), "top 4": chance("Top 4"), "relegation": chance("Relegation")})
    except Exception as e:
        st.warning(f"Couldn't simulate the season ({type(e).__name__}).")

with tab_track:
    log = read_log()
    s = summarise(log)
    st.caption("Predictions are logged before kickoff and scored once the result is in, so these numbers are "
               "genuinely out of sample. Blend model only.")
    if s is None:
        st.info(f"{len(log)} predictions logged so far. None have been played yet.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predictions scored", s["n"])
        m2.metric("Accuracy", f"{s['accuracy']:.1%}")
        m3.metric("Brier score", f"{s['brier']:.3f}", f"{s['brier'] - s['base_rate_brier']:+.3f} vs base rate",
                  delta_color="inverse", help="Lower is better. The base rate always predicts the home/draw/away "
                                              "frequencies of these same results.")
    if len(log):
        shown = log.sort_values("date", ascending=False).assign(
            result=lambda d: d.result.map({"H": "Home win", "D": "Draw", "A": "Away win"}).fillna("Not played yet"),
            **{c: lambda d, c=c: (d[c].astype(float) * 100).round() for c in ("p_home", "p_draw", "p_away")})
        st.dataframe(shown[["date", "home", "away", "p_home", "p_draw", "p_away", "result"]], hide_index=True,
                     width="stretch", column_config={
                         "date": "Date", "home": "Home", "away": "Away", "result": "Result",
                         "p_home": st.column_config.NumberColumn("Home win", format="%d%%"),
                         "p_draw": st.column_config.NumberColumn("Draw", format="%d%%"),
                         "p_away": st.column_config.NumberColumn("Away win", format="%d%%")})

    with st.expander("Backtest: 2023/24 to 2025/26, trained only on earlier seasons"):
        res = Path("results")
        if (res / "metrics_by_season.csv").exists():
            m = pd.read_csv(res / "metrics_by_season.csv", dtype={"season": str})
            m = m[m.model.isin(["base_rate", "logistic", "xgboost", "blend", "bookmaker"])]
            st.caption("Log loss by season (lower is better). The bookmaker row uses closing odds.")
            st.dataframe(m.pivot(index="model", columns="season", values="log_loss").round(4)
                         .assign(mean=lambda d: d.mean(axis=1).round(4)), width="stretch")
        if (res / "blend_test_predictions.csv").exists():
            cal = calibration(pd.read_csv(res / "blend_test_predictions.csv"))
            st.caption("Calibration: when the model says 30%, does it happen about 30% of the time? "
                       "Closer to the diagonal is better.")
            st.line_chart(cal.rename(columns={"observed": "observed frequency"})
                          .assign(perfect=cal.predicted).set_index("predicted")[["observed frequency", "perfect"]])

st.caption("Data: football-data.co.uk and openfootball. These are model probabilities, not betting tips.")
