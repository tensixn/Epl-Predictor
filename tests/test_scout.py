from types import SimpleNamespace

import pandas as pd

from src import scout

SEASON = 2025


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
    picks = scout.shortlist(pv, "Man United", SEASON, ["Striker", "Winger"], 40e6, 26)
    assert len(picks)
    assert (picks.value <= 40e6).all() and (picks.age <= 26).all()
    assert (picks.club != "Man United").all() and set(picks.role) <= {"Striker", "Winger"}
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
