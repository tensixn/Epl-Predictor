"""Dixon-Coles goals model, and its blend with the logistic model.

Each team gets an attack and a defence rating; goals are Poisson with a home advantage and the
low-score correction (rho) from Dixon & Coles (1997). Older matches count less (exponential decay).
It sees only scores, not form or shots, so averaging it with the logistic model helps a little.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

XI = 0.003      # decay per day (about an 8-month half-life); best of 0.0008-0.005 in the walk-forward test
RIDGE = 0.1     # pins the ratings' mean at zero
REFIT_DAYS = 28
MAX_GOALS = 10


class DixonColes:
    def __init__(self, matches, asof):
        """Fit on matches (Date, HomeTeam, AwayTeam, FTHG, FTAG) played before `asof`."""
        tr = matches[matches.Date < asof]
        teams = sorted(set(tr.HomeTeam) | set(tr.AwayTeam))
        self.ix, n = {t: i for i, t in enumerate(teams)}, len(teams)
        h, a = tr.HomeTeam.map(self.ix).values, tr.AwayTeam.map(self.ix).values
        gh, ga = tr.FTHG.values, tr.FTAG.values
        w = np.exp(-XI * (asof - tr.Date).dt.days.values)

        def nll(p):  # weighted Poisson negative log-likelihood and its gradient
            lh, la = p[2 * n] + p[h] - p[n + a], p[a] - p[n + h]
            mh, ma = np.exp(lh), np.exp(la)
            rh, ra = w * (gh - mh), w * (ga - ma)
            g = np.zeros_like(p)
            np.add.at(g, h, -rh); np.add.at(g, a, -ra)
            np.add.at(g, n + a, rh); np.add.at(g, n + h, ra)
            g[2 * n] = -rh.sum()
            g[:2 * n] += RIDGE * p[:2 * n]
            return -(w * (gh * lh - mh + ga * la - ma)).sum() + RIDGE / 2 * (p[:2 * n] ** 2).sum(), g

        p0 = np.zeros(2 * n + 1)
        p0[2 * n] = 0.25
        p = minimize(nll, p0, jac=True, method="L-BFGS-B").x
        self.att, self.dfn, self.home = p[:n], p[n:2 * n], p[2 * n]
        # teams with no history (promoted sides) get the 25th-percentile rating
        self.unknown = (np.percentile(self.att, 25), np.percentile(self.dfn, 25))
        self.rho = self._fit_rho(w, h, a, gh, ga)

    def _fit_rho(self, w, h, a, gh, ga):
        mh, ma = np.exp(self.home + self.att[h] - self.dfn[a]), np.exp(self.att[a] - self.dfn[h])

        def loglik(rho):
            t = np.ones(len(gh))
            for (x, y), v in {(0, 0): 1 - mh * ma * rho, (0, 1): 1 + mh * rho, (1, 0): 1 + ma * rho, (1, 1): 1 - rho}.items():
                m = (gh == x) & (ga == y)
                t[m] = v[m] if isinstance(v, np.ndarray) else v
            return (w * np.log(np.clip(t, 1e-9, None))).sum()

        grid = np.linspace(-0.25, 0.1, 36)
        return grid[np.argmax([loglik(r) for r in grid])]

    def _rating(self, team):
        i = self.ix.get(team)
        return (self.att[i], self.dfn[i]) if i is not None else self.unknown

    def predict(self, fixtures):
        """[home, draw, away] probabilities for each (home, away) pair."""
        out = []
        for home, away in fixtures:
            (ah, dh), (aa, da) = self._rating(home), self._rating(away)
            mh, ma = np.exp(self.home + ah - da), np.exp(aa - dh)
            P = np.outer(poisson.pmf(range(MAX_GOALS), mh), poisson.pmf(range(MAX_GOALS), ma))
            r = self.rho
            P[0, 0] *= 1 - mh * ma * r
            P[0, 1] *= 1 + mh * r
            P[1, 0] *= 1 + ma * r
            P[1, 1] *= 1 - r
            P /= P.sum()
            out.append([np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()])
        return np.array(out)


class Blend:
    """Average of a fitted logistic pipeline and a DixonColes. Used through predict.predict (needs fixtures)."""

    def __init__(self, logistic, dc):
        self.logistic, self.dc = logistic, dc

    def predict_fixtures(self, fixtures, X):
        return (self.logistic.predict_proba(X) + self.dc.predict(fixtures)) / 2


def walk_forward(matches, games):
    """Dixon-Coles probabilities for `games` (Date, HomeTeam, AwayTeam), refit every REFIT_DAYS on earlier matches only."""
    out = np.empty((len(games), 3))
    start = games.Date.min()
    block = ((games.Date - start).dt.days // REFIT_DAYS).values
    for b in np.unique(block):
        rows = np.where(block == b)[0]
        model = DixonColes(matches, start + pd.Timedelta(days=int(b) * REFIT_DAYS))
        out[rows] = model.predict(zip(games.HomeTeam.values[rows], games.AwayTeam.values[rows]))
    return out
