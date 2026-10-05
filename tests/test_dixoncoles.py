import numpy as np
import pandas as pd

from src.dixoncoles import DixonColes


def season():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(60):
        for h, a, mh, ma in (("Strong", "Weak", 2.5, 0.5), ("Weak", "Strong", 0.7, 1.8)):
            rows.append(dict(Date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), HomeTeam=h, AwayTeam=a,
                             FTHG=rng.poisson(mh), FTAG=rng.poisson(ma)))
    return pd.DataFrame(rows)


def test_probabilities_sum_to_one_and_favour_the_strong_side():
    m = DixonColes(season(), pd.Timestamp("2026-04-01"))
    p = m.predict([("Strong", "Weak"), ("Weak", "Strong"), ("Newcomer", "Weak")])
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 0] > 0.6 and p[1, 2] > p[1, 0]
    assert np.isfinite(p).all()  # a team with no history still gets a prediction
