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

st.set_page_config(page_title="EPL match predictor", page_icon="⚽", layout="centered")

st.markdown("""
<style>
.pbar { display:flex; height:34px; border-radius:8px; overflow:hidden; font-weight:700; font-size:.9rem; }
.pbar span { display:flex; align-items:center; justify-content:center; min-width:0; white-space:nowrap; }
.pbar .h { background:#00ff85; color:#1d0a22; }
.pbar .d { background:#8b7a91; color:#1d0a22; }
.pbar .a { background:#e90052; color:#fff; }
.legend { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.2rem 1rem; font-size:.85rem; margin:.4rem 0 .8rem; }
.legend i { display:inline-block; width:.7rem; height:.7rem; border-radius:3px; margin-right:.4rem; }
.legend .h i { background:#00ff85; } .legend .d i { background:#8b7a91; } .legend .a i { background:#e90052; }
.fx { margin:.9rem 0 .2rem; font-weight:600; }
.fx small { font-weight:400; opacity:.7; }
@media (prefers-reduced-motion: no-preference) { .pbar span { transition: flex-basis .3s ease; } }
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
    """Stacked home / draw / away bar; labels sit below it too so colour isn't the only cue."""
    ps = (p_home, p_draw, p_away)
    spans = "".join(f'<span class="{c}" style="flex:{p:.4f}">{p:.0%}</span>' if p >= 0.08
                    else f'<span class="{c}" style="flex:{p:.4f}"></span>' for c, p in zip("hda", ps))
    return (f'<div class="pbar" role="img" aria-label="{escape(home)} {p_home:.0%}, draw {p_draw:.0%}, '
            f'{escape(away)} {p_away:.0%}">{spans}</div>'
            f'<div class="legend"><span class="h"><i></i>{escape(home)} {p_home:.1%}</span>'
            f'<span class="d"><i></i>Draw {p_draw:.1%}</span>'
            f'<span class="a"><i></i>{escape(away)} {p_away:.1%}</span></div>')


models, cols, state, current = load()
form_to = max(state.last_date.values())

st.title("⚽ EPL match predictor")
st.caption(f"Elo ratings and form run to {form_to:%d %b %Y}, the last match in the data.")

model = st.radio("Model", ["logistic", "xgboost"], horizontal=True,
                 help="Logistic regression scored best in the walk-forward test (log loss 0.979 vs 0.988).")

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
        for g, (ph, pd_, pa) in zip(games, probs):
            when = pd.Timestamp(g["date"]).strftime("%a %d %b") + (f", {g['time']}" if g["time"] else "")
            st.markdown(f'<div class="fx">{escape(g["home"])} v {escape(g["away"])} <small>{when}</small></div>'
                        + bar(ph, pd_, pa, g["home"], g["away"]), unsafe_allow_html=True)
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
        st.dataframe(odds, hide_index=True, width="stretch", column_config={
            "points now": st.column_config.NumberColumn("points now"),
            "expected points": st.column_config.NumberColumn("expected points", format="%.1f"),
            "title": st.column_config.ProgressColumn("title", min_value=0, max_value=1, format="percent"),
            "top 4": st.column_config.ProgressColumn("top 4", min_value=0, max_value=1, format="percent"),
            "relegation": st.column_config.ProgressColumn("relegation", min_value=0, max_value=1, format="percent")})
    except Exception as e:
        st.warning(f"Couldn't simulate the season ({type(e).__name__}).")

with tab_track:
    log = read_log()
    s = summarise(log)
    st.caption("Predictions are logged before kickoff and scored once the result is in, so these numbers are "
               "genuinely out of sample. Logistic model only.")
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
        st.dataframe(log.sort_values("date", ascending=False), hide_index=True, width="stretch")

    with st.expander("Backtest: 2023/24 to 2025/26, trained only on earlier seasons"):
        res = Path("results")
        if (res / "metrics_by_season.csv").exists():
            m = pd.read_csv(res / "metrics_by_season.csv", dtype={"season": str})
            m = m[m.model.isin(["base_rate", "logistic", "xgboost", "bookmaker"])]
            st.caption("Log loss by season (lower is better). The bookmaker row uses closing odds.")
            st.dataframe(m.pivot(index="model", columns="season", values="log_loss").round(4)
                         .assign(mean=lambda d: d.mean(axis=1).round(4)), width="stretch")
        if (res / "logistic_test_predictions.csv").exists():
            cal = calibration(pd.read_csv(res / "logistic_test_predictions.csv"))
            st.caption("Calibration: when the model says 30%, does it happen about 30% of the time? "
                       "Closer to the diagonal is better.")
            st.line_chart(cal.rename(columns={"observed": "observed frequency"})
                          .assign(perfect=cal.predicted).set_index("predicted")[["observed frequency", "perfect"]])
