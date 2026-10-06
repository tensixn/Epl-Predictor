import pandas as pd

from src import form


def fpl():
    return pd.DataFrame({
        "first_name": ["Bruno", "Bruno", "Ezri", "Felipe", "Pedro", "Kaoru"],
        "second_name": ["Borges Fernandes", "Guimarães Rodriguez Moura", "Konsa Ngoyo", "Rodrigues da Silva",
                        "Porro Sauceda", "Mitoma"],
        "web_name": ["B.Fernandes", "Bruno G.", "Konsa", "Morato", "Pedro Porro", "Mitoma"],
        "club": ["Man United", "Arsenal", "Arsenal", "Nott'm Forest", "Tottenham", "Brighton"],
        "minutes": [450, 400, 90, 0, 300, 200], "starts": [5, 4, 1, 0, 3, 2], "goals_scored": [3, 0, 0, 0, 1, 1],
        "assists": [1, 2, 0, 0, 0, 0], "expected_goals": [2.5, .3, 0, 0, .4, .9], "expected_assists": [1, 1, 0, 0, 0, 0],
        "status": ["a"] * 6, "news": [""] * 6,
    })


def ours():
    return pd.DataFrame({"name": ["Bruno Fernandes", "Bruno Guimarães", "Ezri Konsa", "Jota Silva", "Pedro Porro",
                                  "Kaoru Mitoma", "Jarrod Bowen"],
                         "club": ["Man United", "Newcastle", "Aston Villa", "Nott'm Forest", "Tottenham", "Brighton",
                                  "West Ham"]})


def test_matches_across_spellings_and_moves():
    m = form.match(ours(), fpl())
    assert m.tolist()[:3] == [0, 1, 2]  # surname in club, every word in one full name (twice, after moves)
    assert m[4] == 4 and m[5] == 5


def test_no_guess_when_unsure():
    m = form.match(ours(), fpl())
    assert pd.isna(m[3])  # Jota Silva is not Felipe Rodrigues da Silva
    assert pd.isna(m[6])  # not in FPL: left the league


def test_with_form_adds_now_columns():
    out = form.with_form(ours(), fpl())
    assert out.loc[1, "now_club"] == "Arsenal" and out.loc[0, "now_goals"] == 3
    assert out.now_club.isna().sum() == 2
