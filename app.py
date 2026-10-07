"""Streamlit page for the EPL predictor.

Run locally from the project root:  streamlit run app.py
"""
import json
import os
import sys
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# Streamlit Cloud pulls each new commit into the running app but keeps the src modules it already imported,
# so a merge that adds a function this page imports fails with ImportError until the app is rebooted.
# Dropping them makes every run import the code that is on disk now.
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]

from src.features import load_matches
from src.fixtures import next_round, season_matches
from src.predict import explain, predict, train
from src.form import load_fpl
from src.scout import ROLES, scouting_report, shortlist, squad_needs
from src.simulate import simulate_season
from src.tracker import calibration, read_log, summarise, vs_bookmaker

st.set_page_config(page_title="Football Predictor", page_icon=str(Path(__file__).parent / "assets" / "favicon.png"),
                   layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&display=swap');
:root { --home:#c8ff3d; --draw:#7f8da3; --away:#4cc3ff; --ink:#07100a; --panel:#101722; --rule:rgba(238,242,247,.14);
        --display:'Barlow Condensed','Arial Narrow',sans-serif; --ease:cubic-bezier(.16,1,.3,1); }
html { scrollbar-color:#2a3a52 transparent; }
[data-testid="stHeader"] { background:transparent; }
.stApp { background-image: radial-gradient(50rem 22rem at 50% -10%, rgba(200,255,61,.07), transparent 70%); }
h1, h1 span { font-family:var(--display) !important; font-weight:800 !important; text-transform:uppercase; }
h1 { font-size:3.6rem; line-height:.95; letter-spacing:-.01em; text-wrap:balance; }
h2, h3, h2 span, h3 span { font-family:var(--display) !important; font-weight:700 !important; text-transform:uppercase; }
h2, h3 { letter-spacing:.015em; text-wrap:balance; }
[data-testid="stCaptionContainer"] { text-wrap:pretty; }
[data-testid="stRadioOption"] { border:1px solid var(--rule); border-radius:6px; padding:.3rem .95rem; margin-right:.5rem;
                                transition:border-color .2s, background-color .2s; }
[data-testid="stRadioOption"] > div > div:first-child { display:none; }
[data-testid="stRadioOption"] p, [role="tab"] p { font-family:var(--display) !important; font-weight:700;
                                font-size:1.1rem; letter-spacing:.05em; text-transform:uppercase; }
[data-testid="stRadioOption"]:hover { border-color:var(--home); }
[data-testid="stRadioOption"][data-selected="true"] { background:var(--home); border-color:var(--home); }
[data-testid="stRadioOption"][data-selected="true"] p { color:var(--ink); }
.card { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:.95rem 1.1rem 1.05rem;
        margin:.7rem 0; transition:border-color .25s; animation:rise .6s var(--ease) both; animation-delay:calc(var(--i,0) * 45ms); }
.card:hover { border-color:rgba(200,255,61,.5); }
.pbar { display:flex; gap:2px; height:38px; border-radius:6px; overflow:hidden; font-family:var(--display); font-weight:700;
        font-size:1.2rem; letter-spacing:.02em; animation:sweep .9s var(--ease) both; animation-delay:calc(var(--i,0) * 45ms + 150ms); }
.pbar span { display:flex; align-items:center; justify-content:center; min-width:0; white-space:nowrap; }
.pbar .fav { font-weight:800; }
.pbar .h { background:var(--home); color:var(--ink); }
.pbar .d { background:var(--draw); color:var(--ink); }
.pbar .a { background:var(--away); color:var(--ink); }
.verdict { font-family:var(--display); font-weight:700; font-size:1.2rem; letter-spacing:.03em; text-transform:uppercase;
           margin-top:.7rem; }
.why { font-size:.9rem; opacity:.8; margin-top:.15rem; }
.thin { font-size:.8rem; opacity:.8; text-align:right; margin-top:.25rem; }
.legend { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.2rem 1rem; font-size:.85rem; margin:.4rem 0 .8rem; }
.legend.key { justify-content:flex-start; gap:.2rem 1.4rem; margin:.2rem 0 0; }
.legend i { display:inline-block; width:.7rem; height:.7rem; border-radius:3px; margin-right:.4rem; }
.legend .h i { background:var(--home); } .legend .d i { background:var(--draw); } .legend .a i { background:var(--away); }
.pbar, .legend, .fx { font-variant-numeric: tabular-nums; }
.day { margin:2rem 0 .2rem; padding-bottom:.35rem; font-family:var(--display); font-weight:700; font-size:1.25rem;
       letter-spacing:.06em; text-transform:uppercase; color:var(--home); border-bottom:1px solid var(--rule); }
.fx { display:grid; grid-template-columns:1fr auto 1fr; gap:0 .9rem; align-items:center; margin:0 0 .75rem;
      font-family:var(--display); font-weight:700; font-size:1.65rem; line-height:1; text-transform:uppercase; }
.fx span:first-child { text-align:right; }
.fx small { font-family:inherit; font-weight:600; font-size:.95rem; letter-spacing:.06em; padding:.25rem .6rem;
            border:1px solid var(--rule); border-radius:4px; opacity:.9; }
::selection { background:var(--home); color:var(--ink); }
:focus-visible { outline:2px solid var(--home); outline-offset:2px; }
@keyframes rise { from { opacity:0; transform:translateY(12px); } }
@keyframes sweep { from { clip-path:inset(0 100% 0 0 round 6px); } to { clip-path:inset(0 0 0 0 round 6px); } }
@media (prefers-reduced-motion: reduce) { .card, .pbar { animation:none; } }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-top:3.2rem; }
  [data-testid="stMainBlockContainer"] h1 { font-size:2.6rem; }
  .fx { font-size:1.25rem; gap:0 .5rem; }
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


def anthropic_key():
    """The Anthropic API key from Streamlit secrets or the environment, or None."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except FileNotFoundError:  # no secrets.toml
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_data(show_spinner=False)
def report(key, club, season_label, budget, needs_json, picks_json, form_label):
    """Claude's write-up, cached so the same inputs don't call the API twice."""
    import anthropic
    from io import StringIO
    needs, picks = pd.read_json(StringIO(needs_json)), pd.read_json(StringIO(picks_json))
    try:
        return scouting_report(anthropic.Anthropic(api_key=key), club, season_label, budget, needs, picks,
                               form_label)
    except anthropic.AuthenticationError:
        return "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in the app's secrets."
    except anthropic.RateLimitError:
        return "Too many requests right now. Try again in a minute."
    except anthropic.APIError as e:
        return f"Couldn't reach Claude ({e.__class__.__name__}). Try again later."


@st.cache_data(ttl=3600)
def fpl_form():
    """This season's FPL player stats from scripts/fetch_fpl.py (refreshed daily), or None."""
    return load_fpl()


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


def why(home, away, probs):
    """The verdict and a plain-English reason for one prediction."""
    head, reason = explain(home, away, probs, state.team_features(home, form_to), state.team_features(away, form_to))
    return f'<div class="verdict">{escape(head)}</div><div class="why">{escape(reason)}</div>'


def kickoff(g, tz):
    """Kick-off as a timestamp in the viewer's time zone; the feed's times are UK time."""
    return pd.Timestamp(f'{g["date"]} {g["time"]}', tz="Europe/London").tz_convert(tz)


models, cols, state, current = load()
form_to = max(state.last_date.values())

st.title("Football predictor")
# A widget's state is dropped on runs where it isn't drawn, so re-assign it to survive a trip to Player values.
st.session_state.model = st.session_state.get("model", "blend")
section = st.radio("Section", ["Match predictor", "Player values", "Scout"], horizontal=True,
                   label_visibility="collapsed")

if section == "Match predictor":
    st.caption("The chance of a home win, a draw or an away win for each Premier League match, from a model trained on "
               f"26 seasons of results. Based on matches up to {form_to:%d %b %Y}. Probabilities, not tips: "
               "the bar splits 100% between home win, draw and away win, and the line under it says why.")

    model = st.session_state.model  # the picker is in "Model settings" at the bottom

    tab_next, tab_pick, tab_season, tab_track = st.tabs(
        ["Fixtures", "Pick a match", "Season odds", "Track record"])

    with tab_next:
        try:
            round_name, games = fixtures()
        except Exception as e:  # network or an unfamiliar team name; the other tab still works
            round_name, games = None, []
            st.warning("Couldn't load the fixtures right now. Try the Pick a match tab, or reload in a minute.")
        if round_name:
            try:  # the browser's zone; None until the page reports it, or if the name isn't one we know
                tz = ZoneInfo(st.context.timezone).key
            except (TypeError, ValueError, LookupError):
                tz = "Europe/London"
            st.subheader(round_name)
            acc = backtest_accuracy()
            if acc:
                st.caption(f"In a test on the last three seasons the model picked the right result {acc[0]:.0%} of the "
                           f"time; the bookmakers managed {acc[1]:.0%}. Track record shows how it is doing on live "
                           f"matches. Kick-off times are in your time zone ({tz}).")
            st.markdown('<div class="legend key"><span class="h"><i></i>Home win</span><span class="d"><i></i>Draw</span>'
                        '<span class="a"><i></i>Away win</span></div>', unsafe_allow_html=True)
            for team in {t for g in games for t in (g["home"], g["away"])} - set(state.elo):
                st.info(f"{team} isn't in the data yet; treated as a newly promoted side.")
            probs = predict({model: models[model]}, cols, state, [(g["home"], g["away"]) for g in games])[model]
            day = None
            for i, (g, (ph, pd_, pa)) in enumerate(zip(games, probs)):
                ko = kickoff(g, tz)
                if ko.date() != day:
                    day = ko.date()
                    st.markdown(f'<div class="day">{ko:%A %d %B}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card" style="--i:{i}"><div class="fx"><span>{escape(g["home"])}</span>'
                            f'<small>{ko:%H:%M}</small><span>{escape(g["away"])}</span></div>'
                            + bar(ph, pd_, pa, g["home"], g["away"], legend=False)
                            + why(g["home"], g["away"], (ph, pd_, pa)) + '</div>',
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
            st.markdown(bar(p_home, p_draw, p_away, home, away) + why(home, away, (p_home, p_draw, p_away)),
                        unsafe_allow_html=True)
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

    with st.expander("What I tried that didn't help"):
        st.markdown("**Expected goals (xG).** I added rolling 5 and 10-match xG for and against from Understat "
                    "(2014/15 on). Log loss, lower is better: 0.9791 without xG, 0.9847 with all the xG columns, "
                    "0.9793 with only the 10-match differences. It never beat the model without it, so I dropped it.")
        st.markdown("**Betting against the bookmakers.** Over the 1,140 backtest matches I staked one unit whenever the "
                    "model's chance times the bookmakers' average closing odds was above a cutoff. The model never "
                    "beat the market, and the bigger its claimed edge, the more it lost:")
        st.dataframe(pd.DataFrame({
            "Rule": ["Bet on every outcome (the bookmaker margin)", "Favourite only", "Model edge over 5%",
                     "Model edge over 20%"],
            "Bets": [3420, 1140, 845, 309], "Return on stake": ["-6.3%", "-4.9%", "-11.0%", "-22.3%"]}),
            hide_index=True, width="stretch")
        st.caption("Where the model disagrees most with the market is mostly where the model is wrong. "
                   "Not betting advice.")

    with st.expander("Model settings (advanced)"):
        st.radio("Model", ["blend", "logistic", "xgboost"], horizontal=True, key="model",
                 format_func={"blend": "Blend", "logistic": "Logistic regression", "xgboost": "XGBoost"}.get,
                 help="The blend averages logistic regression with a Dixon-Coles goals model. It scored best in the "
                      "walk-forward test (log loss 0.975 vs 0.979 for logistic alone, 0.988 for XGBoost).")
        st.caption("Changes Fixtures, Pick a match and Season odds. Track record always shows the blend.")

elif section == "Player values":
    pv = player_values()
    if pv is None:
        st.info("Player values haven't been built yet. Run python -m src.player_value.")
    else:
        label = lambda s: f"{s - 1}/{s % 100:02d}"
        st.caption("Is a player's price tag fair? We compare each Premier League player's Transfermarkt price with what "
                   "their season says they're worth: age, position, games, goals, assists and how their club finished. "
                   f"It's an estimate, not a transfer fee. Covers {label(pv.season.min())} to {label(pv.season.max())}, "
                   "with prices from the summer after each season.")
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
                    "Price (€m)": hist.value / 1e6, "Stats say (€m)": hist.stats_value / 1e6})
                    .set_index("season")[["Price (€m)", "Stats say (€m)"]])

        with st.expander("How accurate are the estimates?"):
            path = Path(__file__).parent / "results" / "player_value_metrics.csv"
            if path.exists():
                m = pd.read_csv(path).groupby("model")[["median_pct_error", "within_25pct"]].mean()
                st.caption("Tested on 2022/23 to 2024/25, using only earlier seasons to make each estimate. "
                           "How close are the estimates to the real Transfermarkt price?")
                st.dataframe((m.loc[["age_position", "xgboost"]] * 100).round(0).rename(
                    index={"age_position": "Just age and position", "xgboost": "Our model"},
                    columns={"median_pct_error": "Typical miss %", "within_25pct": "Players within 25% of price"}),
                    width="stretch")

else:
    pv = player_values()
    if pv is None:
        st.info("Player values haven't been built yet. Run python -m src.player_value.")
    else:
        season = int(pv.season.max())
        label = f"{season - 1}/{season % 100:02d}"
        fpl = fpl_form()
        form_label = None
        if fpl is not None:
            form_label = (f"{season}/{(season + 1) % 100:02d} up to gameweek {fpl.gameweeks.iloc[0]} "
                          f"(Fantasy Premier League, {pd.Timestamp(fpl.as_of.iloc[0]):%d %b})")
        st.caption(f"Pick a club, a budget and an age limit. The scout finds the club's weakest positions, shortlists "
                   f"Premier League players who'd be an upgrade there, and Claude writes up the best signings. "
                   f"Based on {label} stats and summer {season} prices, the latest our data has"
                   + (f", plus this season's form: {form_label}." if form_label else "."))
        c1, c2, c3 = st.columns(3)
        # this season's clubs when we have them, so promoted clubs are in and relegated ones out
        clubs = fpl.club.dropna().unique() if fpl is not None else pv[pv.season == season].club.unique()
        club = c1.selectbox("Club", sorted(clubs))
        budget = c2.slider("Budget (€m)", 10, 200, 60, step=5) * 1e6
        max_age = c3.slider("Oldest age", 18, 34, 28)

        needs = squad_needs(pv, club, season, fpl)
        st.subheader("Where they're weakest")
        if club not in set(pv[pv.season == season].club):
            st.caption(f"{club} weren't in the Premier League in {label}, so this counts only their players who were, "
                       "at other clubs. Positions with nobody show as zero.")
        st.dataframe(needs.assign(strength=needs.strength / 1e6, top6=needs.top6 / 1e6, vs_top6=needs.vs_top6 * 100),
                     hide_index=True, width="stretch",
                     column_order=["role", "regulars", "starts", "age", "strength", "top6", "vs_top6"],
                     column_config={
                         "role": "Position", "regulars": "Who played there",
                         "starts": st.column_config.NumberColumn("Starts", width="small"),
                         "age": st.column_config.NumberColumn("Avg age", format="%.0f", width="small"),
                         "strength": st.column_config.NumberColumn("Stats value €m", format="%.0f", width="small",
                                                                   help="Starts-weighted stats value of who played there"),
                         "top6": st.column_config.NumberColumn("Top six €m", format="%.0f", width="small",
                                                               help="The same for the average top-six club"),
                         "vs_top6": st.column_config.ProgressColumn("vs top six", format="%.0f%%", min_value=0,
                                                                    max_value=200)})

        # default to the two weakest positions where someone fits the budget and age, so the list isn't empty
        fits = [r for r in needs.role if not shortlist(pv, club, season, [r], budget, max_age, fpl=fpl).empty]
        roles = st.multiselect("Positions to strengthen", list(ROLES), default=(fits or list(needs.role))[:2])
        picks = shortlist(pv, club, season, roles, budget, max_age, fpl=fpl)
        st.subheader("Shortlist")
        if picks.empty:
            st.info("Nobody fits. Try a bigger budget, a higher age limit or other positions.")
        else:
            st.dataframe(picks.assign(value=picks.value / 1e6, stats_value=picks.stats_value / 1e6),
                         hide_index=True, width="stretch",
                         column_order=["name", "club", "now_club", "sub_position", "age", "value", "stats_value",
                                       "epl_starts", "epl_goals", "epl_assists", "now_starts", "now_goals",
                                       "now_assists", "now_news"],
                         column_config={
                             "name": "Player", "club": "Club", "sub_position": "Position",
                             "age": st.column_config.NumberColumn("Age", format="%d", width="small"),
                             "value": st.column_config.NumberColumn("Price €m", format="%.0f", width="small"),
                             "stats_value": st.column_config.NumberColumn("Stats say €m", format="%.0f", width="small"),
                             "epl_starts": st.column_config.NumberColumn("Starts", width="small"),
                             "epl_goals": st.column_config.NumberColumn("Goals", width="small"),
                             "epl_assists": st.column_config.NumberColumn("Assists", width="small"),
                             "now_club": "Now at",
                             "now_starts": st.column_config.NumberColumn("Starts now", width="small",
                                                                         help="This season so far"),
                             "now_goals": st.column_config.NumberColumn("Goals now", width="small"),
                             "now_assists": st.column_config.NumberColumn("Assists now", width="small"),
                             "now_news": "Fitness news"})
            st.caption(f"Players at other Premier League clubs in {label}, within budget and age, whose stats value "
                       "beats the club's players in that position. Best first. Starts, goals and assists are last "
                       "season's; the \"now\" columns are this season so far. A blank \"Now at\" means he isn't in a "
                       "Premier League squad this season, or we couldn't match his name.")

            st.subheader("Scouting report")
            key = anthropic_key()
            if key is None:
                st.info("Add an ANTHROPIC_API_KEY to the app's secrets to have Claude write up the best signings.")
            elif st.button("Write the scouting report", type="primary"):
                with st.spinner("Claude is reading the shortlist..."):
                    st.markdown(report(key, club, label, budget, needs.to_json(), picks.to_json(), form_label))

st.caption("Data: football-data.co.uk, openfootball, Transfermarkt (via salimt/football-datasets and "
           "dcaribou/transfermarkt-datasets) and Fantasy Premier League. "
           "These are model probabilities, not betting tips.")
