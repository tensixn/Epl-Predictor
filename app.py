"""Streamlit page for the EPL predictor.

Run locally from the project root:  streamlit run app.py
"""
import json
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud pulls each new commit into the running app but keeps the src modules it already imported,
# so a merge that adds a function this page imports fails with ImportError until the app is rebooted.
# Dropping them makes every run import the code that is on disk now.
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]

from src.features import load_matches
from src.fixtures import next_round, season_matches
from src.predict import predict, train
from src.simulate import simulate_season
from src.tracker import calibration, read_log, summarise, vs_bookmaker

st.set_page_config(page_title="Football predictor", page_icon=str(Path(__file__).parent / "assets" / "favicon.png"),
                   layout="centered")

st.markdown("""
<style>
:root { --home:#3ddc97; --draw:#8b7a91; --away:#7aa7ff; --ink:#1d0a22; --rule:rgba(245,240,247,.16); }
html { scrollbar-color:#43294a transparent; }
[data-testid="stHeader"] { background:transparent; }
.stApp { background-image: radial-gradient(60rem 28rem at 12% -8%, rgba(132,72,150,.20), transparent 70%); }
h1 { letter-spacing:-.025em; text-wrap:balance; }
h2, h3 { letter-spacing:-.015em; font-weight:600; text-wrap:balance; }
[data-testid="stCaptionContainer"] { text-wrap:pretty; }
.pbar { display:flex; height:34px; border-radius:8px; overflow:hidden; font-weight:500; font-size:.9rem; }
.pbar span { display:flex; align-items:center; justify-content:center; min-width:0; white-space:nowrap; }
.pbar .fav { font-weight:700; }
.pbar .h { background:var(--home); color:var(--ink); }
.pbar .d { background:var(--draw); color:var(--ink); }
.pbar .a { background:var(--away); color:var(--ink); }
.thin { font-size:.8rem; opacity:.8; text-align:right; margin-top:.25rem; }
.legend { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.2rem 1rem; font-size:.85rem; margin:.4rem 0 .8rem; }
.legend.key { justify-content:flex-start; gap:.2rem 1.4rem; margin:.2rem 0 0; }
.legend i { display:inline-block; width:.7rem; height:.7rem; border-radius:3px; margin-right:.4rem; }
.legend .h i { background:var(--home); } .legend .d i { background:var(--draw); } .legend .a i { background:var(--away); }
.pbar, .legend, .fx { font-variant-numeric: tabular-nums; }
.day { margin:1.8rem 0 .3rem; padding-bottom:.35rem; font-weight:600; border-bottom:1px solid var(--rule); }
.fx { display:grid; grid-template-columns:1fr auto 1fr; gap:0 .6rem; align-items:baseline; margin:1.1rem 0 .4rem; font-weight:500; }
.fx span:last-child { text-align:right; }
.fx small { font-weight:400; opacity:.75; }
::selection { background:var(--home); color:var(--ink); }
:focus-visible { outline:2px solid var(--home); outline-offset:2px; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-top:3.2rem; }
  [data-testid="stMainBlockContainer"] h1 { font-size:2.1rem; }
  [role="tablist"] { gap:.75rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading predictions (about 10 seconds the first time)...")
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


@st.cache_data
def player_values():
    """Out-of-sample stats values from `python -m src.player_value`, or None if they haven't been built."""
    try:
        return pd.read_csv(Path(__file__).parent / "results" / "player_values.csv")
    except OSError:
        return None


@st.cache_data
def backtest_accuracy():
    """(model, bookmaker) share of matches called right in the walk-forward test, or None."""
    try:
        s = json.loads((Path(__file__).parent / "results" / "summary.json").read_text())
        return s["blend"]["accuracy"], s["bookmaker"]["accuracy"]
    except (OSError, KeyError, ValueError):
        return None


def bar(p_home, p_draw, p_away, home, away, legend=True):
    """Stacked home / draw / away bar, the favourite's label in bold. With `legend`, names each colour
    underneath (so colour isn't the only cue); without it the caller shows one shared key, and any segment
    too thin for its label is listed below the bar instead."""
    ps = (p_home, p_draw, p_away)
    fav = max(range(3), key=ps.__getitem__)
    spans = "".join(f'<span class="{c}{" fav" if i == fav else ""}" style="flex:{p:.4f}">{p:.0%}</span>' if p >= 0.08
                    else f'<span class="{c}" style="flex:{p:.4f}"></span>' for i, (c, p) in enumerate(zip("hda", ps)))
    html = (f'<div class="pbar" role="img" aria-label="{escape(home)} {p_home:.0%}, draw {p_draw:.0%}, '
            f'{escape(away)} {p_away:.0%}">{spans}</div>')
    if legend:
        tight = [f" {p:.0%}" if p < 0.08 else "" for p in ps]  # no label fits inside a thin segment
        return html + (f'<div class="legend"><span class="h"><i></i>{escape(home)}{tight[0]}</span>'
                       f'<span class="d"><i></i>Draw{tight[1]}</span>'
                       f'<span class="a"><i></i>{escape(away)}{tight[2]}</span></div>')
    thin = [f"{n} {p:.0%}" for n, p in zip(("Home win", "Draw", "Away win"), ps) if p < 0.08]
    return html + (f'<div class="thin">{" · ".join(thin)}</div>' if thin else "")


models, cols, state, current = load()
form_to = max(state.last_date.values())

st.title("Football predictor")
section = st.radio("Section", ["Match predictor", "Player values"], horizontal=True, label_visibility="collapsed")

if section == "Match predictor":
    st.caption("The chance of a home win, a draw or an away win for each Premier League match, from a model trained on "
               f"26 seasons of results. Based on matches up to {form_to:%d %b %Y}. Probabilities, not tips.")

    model = st.session_state.get("model", "blend")  # the picker is in "Model settings" at the bottom

    tab_next, tab_pick, tab_season, tab_track = st.tabs(
        ["Fixtures", "Pick a match", "Season odds", "Track record"])

    with tab_next:
        try:
            round_name, games = fixtures()
        except Exception as e:  # network or an unfamiliar team name; the other tab still works
            round_name, games = None, []
            st.warning("Couldn't load the fixtures right now. Try the Pick a match tab, or reload in a minute.")
        if round_name:
            st.subheader(round_name)
            acc = backtest_accuracy()
            if acc:
                st.caption(f"In a test on the last three seasons the model picked the right result {acc[0]:.0%} of the "
                           f"time; the bookmakers managed {acc[1]:.0%}. Track record shows how it is doing on live "
                           "matches. Kick-off times are UK time.")
            st.markdown('<div class="legend key"><span class="h"><i></i>Home win</span><span class="d"><i></i>Draw</span>'
                        '<span class="a"><i></i>Away win</span></div>', unsafe_allow_html=True)
            for team in {t for g in games for t in (g["home"], g["away"])} - set(state.elo):
                st.info(f"{team} isn't in the data yet; treated as a newly promoted side.")
            probs = predict({model: models[model]}, cols, state, [(g["home"], g["away"]) for g in games])[model]
            day = None
            for g, (ph, pd_, pa) in zip(games, probs):
                if g["date"] != day:
                    day = g["date"]
                    st.markdown(f'<div class="day">{pd.Timestamp(day):%A %d %B}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="fx"><span>{escape(g["home"])}</span><small>{escape(g["time"])}</small>'
                            f'<span>{escape(g["away"])}</span></div>' + bar(ph, pd_, pa, g["home"], g["away"], legend=False),
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
                    rows.append({"team": t, "strength rating (Elo)": round(f["elo"]), "points per game (last 5)": f["pts_5"],
                                 "goals for (last 5)": f["gf_5"], "goals against (last 5)": f["ga_5"]})
                st.dataframe(rows, hide_index=True)

    with tab_season:
        st.caption("10,000 simulated finishes to the season. Each simulated result moves both teams' strength ratings "
                   "before the next game is drawn, so a hot streak carries on. Recent form stays at today's values.")
        try:
            odds = season_odds(model)
            odds[["title", "top 4", "relegation"]] *= 100
            chance = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.1f%%",
                                                                   width="small")
            # the chances come before "Points" so a phone's first screen shows them without sideways scrolling
            st.dataframe(odds, hide_index=True, width="stretch", height="content",
                         column_order=["team", "expected points", "title", "top 4", "relegation", "points now"],
                         column_config={
                "team": "Team",
                "points now": st.column_config.NumberColumn("Points", width="small"),
                "expected points": st.column_config.NumberColumn("Projected", format="%.1f", width="small"),
                "title": chance("Title"), "top 4": chance("Top 4"), "relegation": chance("Relegation")})
        except Exception as e:
            st.warning("Couldn't simulate the season right now. Try reloading in a minute.")

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
        vb = vs_bookmaker(log)
        if vb:
            st.caption(f"Against the bookmakers over {vb['n']} scored games with odds (log loss, lower is better): "
                       f"model {vb['model']:.3f}, bookmakers {vb['bookmaker']:.3f}. Odds are the market average "
                       "when they were first published, so they are a bit less sharp than closing odds.")
        if len(log):
            shown = log.sort_values("date", ascending=False).assign(
                date=lambda d: pd.to_datetime(d.date).dt.strftime("%d %b"),
                result=lambda d: d.result.map({"H": "Home win", "D": "Draw", "A": "Away win"}).fillna("Not played yet"),
                **{c: lambda d, c=c: (d[c].astype(float) * 100).round() for c in ("p_home", "p_draw", "p_away")})
            st.dataframe(shown[["date", "home", "away", "p_home", "p_draw", "p_away", "result"]], hide_index=True,
                         width="stretch", column_config={
                             "date": "Date", "home": "Home", "away": "Away", "result": "Result",
                             "p_home": st.column_config.NumberColumn("Home %", format="%d%%", width="small"),
                             "p_draw": st.column_config.NumberColumn("Draw %", format="%d%%", width="small"),
                             "p_away": st.column_config.NumberColumn("Away %", format="%d%%", width="small")})

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

    with st.expander("Model settings (advanced)"):
        st.radio("Model", ["blend", "logistic", "xgboost"], horizontal=True, key="model",
                 format_func={"blend": "Blend", "logistic": "Logistic regression", "xgboost": "XGBoost"}.get,
                 help="The blend averages logistic regression with a Dixon-Coles goals model. It scored best in the "
                      "walk-forward test (log loss 0.975 vs 0.979 for logistic alone, 0.988 for XGBoost).")
        st.caption("Changes Fixtures, Pick a match and Season odds. Track record always shows the blend.")

else:
    pv = player_values()
    if pv is None:
        st.info("Player values haven't been built yet. Run python -m src.player_value.")
    else:
        st.caption("Is a player's price tag fair? We compare each Premier League player's Transfermarkt price with what "
                   "their season says they're worth: age, position, games, goals, assists and how their club finished. "
                   "It's an estimate, not a transfer fee.")
        label = lambda s: f"{s - 1}/{s % 100:02d}"
        seasons = sorted(pv.season.unique(), reverse=True)
        c1, c2 = st.columns(2)
        season = c1.selectbox("Season", seasons, format_func=label)
        clubs = sorted(pv[pv.season == season].club.unique())
        club = c2.selectbox("Club", ["All clubs"] + clubs)
        c3, c4 = st.columns(2)
        group = c3.selectbox("Position", ["All positions", "Goalkeeper", "Defender", "Midfield", "Attack"])
        min_starts = c4.slider("Minimum games started", 0, 38, 10)
        view = pv[(pv.season == season) & (pv.epl_starts >= min_starts) & pv.value.notna()]
        if club != "All clubs":
            view = view[view.club == club]
        if group != "All positions":
            view = view[view.main_position == group]
        order = st.radio("Show first", ["Most overpriced", "Biggest bargains", "Most expensive"], horizontal=True)
        by, asc = {"Most overpriced": ("gap", False), "Biggest bargains": ("gap", True),
                   "Most expensive": ("value", False)}[order]
        view = view.sort_values(by, ascending=asc)
        verdict = pd.cut(view.gap, [-9, -.5, -.2, .2, .5, 1e9],
                         labels=["Bargain", "Good value", "Fair price", "Pricey", "Overpriced"]).astype(str)
        st.dataframe(view.assign(value=view.value / 1e6, stats_value=view.stats_value / 1e6, verdict=verdict),
                     hide_index=True, width="stretch",
                     column_order=["name", "club", "sub_position", "age", "value", "stats_value", "verdict",
                                   "epl_starts", "epl_goals", "epl_assists"],
                     column_config={
                         "name": "Player", "club": "Club", "sub_position": "Position", "verdict": "Verdict",
                         "age": st.column_config.NumberColumn("Age", format="%d", width="small"),
                         "value": st.column_config.NumberColumn("Price €m", format="%.1f", width="small",
                                                                help="Transfermarkt market value"),
                         "stats_value": st.column_config.NumberColumn("Stats say €m", format="%.1f", width="small",
                                                                      help="What their season suggests they're worth"),
                         "epl_starts": st.column_config.NumberColumn("Starts", width="small"),
                         "epl_goals": st.column_config.NumberColumn("Goals", width="small"),
                         "epl_assists": st.column_config.NumberColumn("Assists", width="small")})
        st.caption(f"{len(view)} players. Overpriced or pricey usually means the market is paying for youth, potential "
                   "or reputation the numbers can't see. Bargain or good value often means age or a short contract.")

        with st.expander("One player's history"):
            names = view.drop_duplicates("player_id")
            if len(names):
                pick = st.selectbox("Player", names.player_id, format_func=dict(zip(names.player_id, names.name)).get)
                hist = pv[(pv.player_id == pick) & pv.value.notna()].sort_values("season")
                st.line_chart(hist.assign(season=hist.season.map(label), **{
                    "Market value (€m)": hist.value / 1e6, "Stats value (€m)": hist.stats_value / 1e6})
                    .set_index("season")[["Market value (€m)", "Stats value (€m)"]])

        with st.expander("How accurate is the stats value?"):
            path = Path(__file__).parent / "results" / "player_value_metrics.csv"
            if path.exists():
                m = pd.read_csv(path)
                m = m.groupby("model")[["median_pct_error", "within_25pct", "r2_log"]].mean()
                st.caption("Tested on 2022/23 to 2024/25 with models trained only on earlier seasons. The app uses "
                           "XGBoost. Median error is how far a typical estimate is from the market value.")
                st.dataframe((m * 100).round(0).rename(
                    index={"age_position": "Age and position only", "ridge": "Ridge regression", "xgboost": "XGBoost"},
                    columns={"median_pct_error": "Median error %", "within_25pct": "Within 25% of market %",
                             "r2_log": "Variance explained %"}), width="stretch")

st.caption("Data: football-data.co.uk, openfootball and Transfermarkt (via salimt/football-datasets). "
           "These are model probabilities, not betting tips.")
