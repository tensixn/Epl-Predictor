from datetime import date

from src.quota import DailyCap


def test_cap_stops_at_the_limit():
    cap = DailyCap(2, today=lambda: date(2026, 10, 7))
    assert cap.take() and cap.take()
    assert not cap.take() and cap.left() == 0


def test_cap_resets_the_next_day():
    day = [date(2026, 10, 7)]
    cap = DailyCap(1, today=lambda: day[0])
    assert cap.take() and not cap.take()
    day[0] = date(2026, 10, 8)
    assert cap.left() == 1 and cap.take()
