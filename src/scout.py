"""AI player scout: who should a Premier League club sign, given its squad, a budget and an age limit?

Works on the player value analyzer's output (results/player_values.csv), so every player has a
Transfermarkt price and a "stats value" (what their season says they're worth, from a model that
never sees a price).

1. squad_needs ranks a club's positions by how strong its regulars are there, against the average
   of that season's top six. Strength is the starts-weighted stats value of the players in the role.
2. shortlist finds players at other clubs who fit the budget and age limit and are stronger than the
   club's current regulars in the weakest roles.
3. scouting_report hands both tables to Claude, which picks signings from the shortlist only and
   explains each pick with the numbers it was given.

Steps 1 and 2 are plain pandas, so the app shows them without an API key; only step 3 needs one.
"""
import pandas as pd

from src.form import with_form

MODEL = "claude-opus-5-5"

ROLES = {
    "Goalkeeper": ["Goalkeeper"],
    "Centre-back": ["Centre-Back"],
    "Full-back": ["Left-Back", "Right-Back"],
    "Defensive midfield": ["Defensive Midfield"],
    "Central midfield": ["Central Midfield", "Left Midfield", "Right Midfield"],
    "Attacking midfield": ["Attacking Midfield"],
    "Winger": ["Left Winger", "Right Winger"],
    "Striker": ["Centre-Forward", "Second Striker"],
}
ROLE_OF = {pos: role for role, positions in ROLES.items() for pos in positions}


def with_roles(pv):
    return pv.assign(role=pv.sub_position.map(ROLE_OF))


def role_strength(players):
    """Starts-weighted stats value per club and role, with the role's regulars and their age."""
    p = players[players.epl_starts > 0].assign(w=lambda d: d.epl_starts * d.stats_value)
    g = p.groupby(["club", "role"])
    return pd.DataFrame({"strength": g.w.sum() / g.epl_starts.sum(),
                         "starts": g.epl_starts.sum(),
                         "age": (p.age * p.epl_starts).groupby([p.club, p.role]).sum() / g.epl_starts.sum()})


def squad(pv, club, season, fpl=None):
    """The club's players in `season`. A club that wasn't in the Premier League that season (promoted since) has
    none, so with `fpl` it gets the players on its squad now who played in the league that season, elsewhere."""
    players = with_roles(pv[pv.season == season])
    mine = players[players.club == club]
    if mine.empty and fpl is not None:
        mine = with_form(players, fpl)
        mine = mine.loc[mine.now_club == club, players.columns].assign(club=club)
    return mine


def squad_needs(pv, club, season, fpl=None):
    """The club's roles, weakest first: its strength there against the top six's average."""
    players = with_roles(pv[pv.season == season])
    strength = role_strength(players)
    top6 = players.groupby("club").team_rank.first().nsmallest(6).index
    benchmark = strength.loc[strength.index.get_level_values("club").isin(top6)].groupby("role").strength.mean()
    own = squad(pv, club, season, fpl)
    mine = (role_strength(own).droplevel("club") if len(own) else pd.DataFrame(columns=["strength", "starts", "age"])
            ).reindex(list(ROLES))
    regulars = (own[own.epl_starts > 0].sort_values("epl_starts", ascending=False)
                .groupby("role").name.apply(lambda s: ", ".join(s.head(3))))
    out = pd.DataFrame({"role": list(ROLES), "regulars": regulars.reindex(list(ROLES)).fillna("").values,
                        "starts": mine.starts.fillna(0).astype(int).values, "age": mine.age.values,
                        "strength": mine.strength.fillna(0).values,
                        "top6": benchmark.reindex(list(ROLES)).values})
    out["vs_top6"] = out.strength / out.top6
    return out.sort_values("vs_top6").reset_index(drop=True)


def shortlist(pv, club, season, roles, budget, max_age, min_starts=15, per_role=5, fpl=None):
    """Players at other clubs, priced within budget and no older than max_age, who outscore the club's
    current regulars in each role (for a promoted club, the players on its squad now who played in the league
    that season). Best stats value first. With `fpl` (src.form.load_fpl()), adds this
    season's form and current club, and leaves out players who have since joined `club`."""
    players = with_roles(pv[(pv.season == season) & pv.value.notna()])
    own = squad(pv, club, season, fpl)
    current = role_strength(own).strength if len(own) else pd.Series(dtype=float)
    pool = players[(players.club != club) & players.role.isin(roles) & (players.value <= budget)
                   & (players.age <= max_age) & (players.epl_starts >= min_starts)]
    bar = pd.Series([current.get((club, r), 0.0) for r in pool.role], index=pool.index, dtype=float)
    pool = pool[pool.stats_value > bar]
    cols = ["name", "club", "role", "sub_position", "age", "value", "stats_value", "gap",
            "epl_starts", "epl_goals", "epl_assists", "epl_clean_sheets"]
    if fpl is not None:
        pool = with_form(pool, fpl)
        pool = pool[pool.now_club != club]
        cols += ["now_club", "now_starts", "now_minutes", "now_goals", "now_assists", "now_xg", "now_xa",
                 "now_status", "now_news"]
    pool = pool.sort_values("stats_value", ascending=False).groupby("role").head(per_role)
    return pool[cols].reset_index(drop=True)


SYSTEM = """You are a football recruitment analyst writing a short scouting note for a Premier League club.
You get two tables: the club's squad needs by role, and a shortlist of affordable players from other Premier League clubs.
Recommend up to three signings, all from the shortlist; never name a player who isn't on it.
For each pick, say which need it fills and back it with the numbers given (starts, goals, assists, age, price, stats value).
"Price" is the Transfermarkt market value, not a transfer fee. "Stats value" is what the player's season says they're worth;
a stats value above the price suggests a bargain. Keep the picks within the total budget if you can, and say so if you can't.
If the shortlist has now_ columns, they are this season so far: now_club is where the player is now (blank: not on a
Premier League squad now, so he left the league or we couldn't match him), now_status is FPL's availability
(a = available, d = doubtful, i = injured, s = suspended, u = unavailable) with now_news explaining it.
Weigh this season's form and availability as well as last season's numbers, and say when they disagree.
Close with one line on the biggest risk. Plain English, no hype, under 250 words, Markdown with a short heading per pick."""


def build_prompt(club, season_label, budget, needs, picks, form_label=None):
    money = lambda v: f"{v / 1e6:.1f}"
    needs_csv = needs.assign(strength=needs.strength.map(money), top6=needs.top6.map(money),
                             vs_top6=needs.vs_top6.round(2), age=needs.age.round(1)).to_csv(index=False)
    picks_csv = picks.assign(value=picks.value.map(money), stats_value=picks.stats_value.map(money),
                             gap=picks.gap.round(2), age=picks.age.round(1)).to_csv(index=False)
    form = f"This season's form (now_ columns): {form_label}\n" if form_label else ""
    return (f"Club: {club}\nSeason the stats are from: {season_label}\n{form}Total budget: €{budget / 1e6:.0f}m\n\n"
            f"Squad needs, weakest role first (strength and top6 in €m, vs_top6 = club strength / top-six average):\n"
            f"{needs_csv}\nShortlist (value = price in €m, stats_value in €m, gap = price / stats value - 1):\n"
            f"{picks_csv}")


def scouting_report(client, club, season_label, budget, needs, picks, form_label=None, model=MODEL):
    """Claude's recommendation as Markdown. `client` is an anthropic.Anthropic()."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(club, season_label, budget, needs, picks, form_label)}],
    )
    if response.stop_reason == "refusal":
        return "The scout declined to write a report for this request."
    return "".join(b.text for b in response.content if b.type == "text")
