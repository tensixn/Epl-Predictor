from types import SimpleNamespace

import pandas as pd

from src import scout

SEASON = 2026  # 2025/26


def values():
    return pd.read_csv(scout.__file__.replace("src/scout.py", "results/player_values.csv"))


def test_every_position_has_a_role():
    pv = values()
    assert pv.sub_position.map(scout.ROLE_OF).notna().all()


def test_every_club_is_in_the_latest_season():
    assert values().query("season == @SEASON").club.nunique() == 20


def test_needs_cover_every_role_weakest_first():
    needs = scout.squad_needs(values(), "Arsenal", SEASON)
    assert list(sorted(needs.role)) == sorted(scout.ROLES)
    assert needs.vs_top6.is_monotonic_increasing


def test_shortlist_respects_the_filters():
    pv = values()
    picks = scout.shortlist(pv, "Man United", SEASON, ["Defensive midfield", "Centre-back"], 50e6, 27)
    assert len(picks)
    assert (picks.value <= 50e6).all() and (picks.age <= 27).all()
    assert (picks.club != "Man United").all() and set(picks.role) <= {"Defensive midfield", "Centre-back"}
    current = scout.role_strength(scout.with_roles(pv[pv.season == SEASON])).strength
    assert all(row.stats_value > current[("Man United", row.role)] for row in picks.itertuples())


def test_shortlist_can_be_empty():
    assert scout.shortlist(values(), "Arsenal", SEASON, ["Striker"], 1e6, 18).empty


class FakeClient:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[
            SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="## Sign him")])


def test_report_sends_only_the_shortlist_and_returns_text():
    pv = values()
    needs = scout.squad_needs(pv, "Chelsea", SEASON)
    picks = scout.shortlist(pv, "Chelsea", SEASON, list(needs.role[:2]), 60e6, 28)
    client = FakeClient()
    assert scout.scouting_report(client, "Chelsea", "2024/25", 60e6, needs, picks) == "## Sign him"
    prompt = client.calls[0]["messages"][0]["content"]
    assert all(name in prompt for name in picks.name)
    assert client.calls[0]["model"] == scout.MODEL


def test_report_handles_a_refusal():
    pv = values()
    needs = scout.squad_needs(pv, "Chelsea", SEASON)
    picks = scout.shortlist(pv, "Chelsea", SEASON, list(needs.role[:2]), 60e6, 28)
    assert "declined" in scout.scouting_report(FakeClient("refusal"), "Chelsea", "2024/25", 60e6, needs, picks)


def test_shortlist_with_form_drops_players_who_since_joined_the_club():
    from src.form import load_fpl, with_form
    pv, fpl = values(), load_fpl()
    roles = list(scout.ROLES)
    picks = scout.shortlist(pv, "Man United", SEASON, roles, 200e6, 34, per_role=50, fpl=fpl)
    assert "now_starts" in picks and (picks.now_club != "Man United").all()
    plain = with_form(scout.shortlist(pv, "Man United", SEASON, roles, 200e6, 34, per_role=50), fpl)
    joined = plain[plain.now_club == "Man United"]
    assert len(joined)  # e.g. Youri Tielemans and Carlos Baleba play for Man United now
    assert set(picks.name) == set(plain.name) - set(joined.name)


def test_a_promoted_club_is_judged_on_its_current_squad():
    from src.form import load_fpl
    pv, fpl = values(), load_fpl()
    assert "Hull" not in set(pv[pv.season == SEASON].club)
    needs = scout.squad_needs(pv, "Hull", SEASON, fpl)
    assert sorted(needs.role) == sorted(scout.ROLES)
    own = scout.squad(pv, "Hull", SEASON, fpl)
    assert len(own) and (own.club == "Hull").all()
    assert set(needs.regulars.str.split(", ").explode()) - {""} <= set(own.name)
    picks = scout.shortlist(pv, "Hull", SEASON, list(needs.role[:2]), 30e6, 28, fpl=fpl)
    assert len(picks) and (picks.now_club != "Hull").all()
