"""Streamlit page for the EPL predictor.

Run locally from the project root:  streamlit run app.py
"""
import streamlit as st

from src.features import load_matches
from src.predict import predict, train

st.set_page_config(page_title="EPL match predictor", page_icon="⚽")


@st.cache_resource(show_spinner="Training on every season in data/raw...")
def load():
    models, cols, state = train()
    matches = load_matches()
    latest = matches[matches.season == matches.season.max()]
    current = sorted(set(latest.HomeTeam) | set(latest.AwayTeam))
    return models, cols, state, current


models, cols, state, current = load()
form_to = max(state.last_date.values())

st.title("EPL match predictor")
st.caption(f"Elo ratings and form run to {form_to:%d %b %Y}, the last match in the data.")

all_teams = st.checkbox("Show every team since 2000/01", value=False)
teams = sorted(state.elo) if all_teams else current

left, right = st.columns(2)
home = left.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
away = right.selectbox("Away team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

model = st.radio("Model", ["logistic", "xgboost"], horizontal=True,
                 help="Logistic regression scored best in the walk-forward test (log loss 0.979 vs 0.988).")
p_home, p_draw, p_away = predict({model: models[model]}, cols, state, [(home, away)])[model][0]

c1, c2, c3 = st.columns(3)
c1.metric(f"{home} win", f"{p_home:.1%}")
c2.metric("Draw", f"{p_draw:.1%}")
c3.metric(f"{away} win", f"{p_away:.1%}")

with st.expander("Team form behind this prediction"):
    rows = []
    for t in (home, away):
        f = state.team_features(t, form_to)
        rows.append({"team": t, "Elo": round(f["elo"]), "points per game (last 5)": f["pts_5"],
                     "goals for (last 5)": f["gf_5"], "goals against (last 5)": f["ga_5"]})
    st.dataframe(rows, hide_index=True)
