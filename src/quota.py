"""A site-wide daily limit, so visitors to the public app can't run up the Anthropic bill.

The app keeps one DailyCap for the whole server (st.cache_resource), so every visitor shares it. It
lives in memory: a reboot of the app resets the count, which at worst allows one extra day's worth.
"""
import threading
from datetime import datetime, timezone


def utc_today():
    return datetime.now(timezone.utc).date()


class DailyCap:
    def __init__(self, limit, today=utc_today):
        self.limit, self.today = limit, today
        self.day, self.used = today(), 0
        self.lock = threading.Lock()

    def _roll(self):
        if self.today() != self.day:
            self.day, self.used = self.today(), 0

    def left(self):
        with self.lock:
            self._roll()
            return self.limit - self.used

    def take(self):
        """Use one for today; False if none are left."""
        with self.lock:
            self._roll()
            if self.used >= self.limit:
                return False
            self.used += 1
            return True
