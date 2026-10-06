"""This season's form: Fantasy Premier League stats (scripts/fetch_fpl.py) matched to our players.

FPL and Transfermarkt spell names differently ("Bruno Borges Fernandes" vs "Bruno Fernandes"), so a
player is matched by full name, then by first initial and surname, or FPL short name, within last season's club, then by
short name alone for one-name players, then by every word of the name appearing in one FPL player's
full name. Only unambiguous matches count.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

FPL = Path(__file__).resolve().parents[1] / "data" / "players" / "fpl_current.csv"
# FPL column -> name in our tables ("now_" = this season so far)
COLUMNS = {"club": "now_club", "minutes": "now_minutes", "starts": "now_starts", "goals_scored": "now_goals",
           "assists": "now_assists", "expected_goals": "now_xg", "expected_assists": "now_xa",
           "status": "now_status", "news": "now_news"}


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).split()


def load_fpl(path=FPL):
    try:
        return pd.read_csv(path)
    except OSError:
        return None


def _unique(keys):
    """key -> fpl row index, for keys that point at exactly one FPL player."""
    s = pd.Series(keys.index, index=keys.values)
    counts = s.index.value_counts()
    return s[s.index.isin(counts[counts == 1].index)]


def match(players, fpl):
    """FPL row index for each of `players` (name, club = last season's club), NaN when there's no clear match."""
    full = _unique(fpl.apply(lambda r: " ".join(norm(f"{r.first_name} {r.second_name}")), axis=1))
    last = _unique(fpl.apply(lambda r: (r.club, (norm(r.first_name) or [" "])[0][0],
                                        (norm(r.second_name) or [""])[-1]), axis=1))
    short = _unique(fpl.apply(lambda r: (r.club, " ".join(norm(r.web_name))), axis=1))
    mono = _unique(fpl.web_name.map(lambda n: " ".join(norm(n))))
    fullwords = {i: set(norm(f"{r.first_name} {r.second_name}")) for i, r in fpl.iterrows()}

    def one(name, club):
        words = norm(name)
        if not words:
            return None
        for key, table in ((" ".join(words), full), ((club, words[0][0], words[-1]), last),
                           ((club, " ".join(words)), short), ((club, words[-1]), short)):
            if key in table.index:
                return table[key]
        if len(words) == 1 and words[0] in mono.index:
            return mono[words[0]]
        # every word of the name appears in one FPL player's full name ("Ezri Konsa" -> "Ezri Konsa Ngoyo")
        hits = [i for i, w in fullwords.items() if set(words) <= w]
        return hits[0] if len(hits) == 1 else None

    found = pd.Series([one(n, c) for n, c in zip(players.name, players.club)], index=players.index, dtype="float")
    return found.mask(found.duplicated(keep=False))  # two of our players on one FPL player: trust neither


def with_form(players, fpl):
    """`players` plus this season's FPL columns (empty where unmatched, i.e. no longer in the league)."""
    idx = match(players, fpl)
    got = fpl[list(COLUMNS)].rename(columns=COLUMNS).reindex(idx.values).set_axis(players.index)
    return players.join(got)
