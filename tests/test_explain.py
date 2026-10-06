from src.predict import explain

T = lambda elo, pts: {"elo": elo, "pts_5": pts}


def test_close_game():
    assert explain("A", "B", (.36, .30, .34), T(1500, 1), T(1500, 1))[0] == "Too close to call"


def test_favourite_and_reasons_agree_with_probs():
    head, why = explain("A", "B", (.2, .25, .55), T(1500, 1), T(1600, 2))
    assert head == "B favoured: 55% to win" and "stronger" in why and "better form" in why and "home" not in why


def test_small_edge_fallback():
    assert explain("A", "B", (.3, .3, .4), T(1500, 1), T(1510, 1))[1] == "B has a small edge."
