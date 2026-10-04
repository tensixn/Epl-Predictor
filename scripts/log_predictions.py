"""Score logged predictions against real results, then log the next round.

Run from the project root after scripts/fetch_current_season.py:

    python scripts/log_predictions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import train  # noqa: E402
from src.tracker import summarise, update  # noqa: E402

models, cols, state = train()
log = update(models, cols, state)
s = summarise(log)
print(f"{len(log)} predictions logged, {s['n'] if s else 0} scored")
if s:
    print(f"accuracy {s['accuracy']:.1%}, log loss {s['log_loss']:.3f}, Brier {s['brier']:.3f}")
