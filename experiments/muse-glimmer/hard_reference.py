"""Simple list-based semantic oracle; intentionally not an O(1) implementation."""

import math


class ExpiringLRU:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("capacity")
        self.capacity, self.rows, self.time = capacity, [], None

    def numeric(self, value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("number")

    def validate_now(self, now):
        self.numeric(now)
        if self.time is not None and now < self.time:
            raise ValueError("backwards")

    def purge(self, now):
        self.rows = [r for r in self.rows if r[2] is None or r[2] > now]
        self.time = now

    def put(self, key, value, *, now, ttl=None):
        self.put_many([(key, value, ttl)], now=now)

    def put_many(self, entries, *, now):
        self.validate_now(now)
        pending = []
        try:
            iterator = iter(entries)
        except TypeError:
            raise ValueError("entries") from None
        for row in iterator:
            try:
                key, value, ttl = row
            except (TypeError, ValueError):
                raise ValueError("triple") from None
            if not isinstance(key, str):
                raise ValueError("key")  # noqa: TRY004 - the prompt requires ValueError.
            deadline = None
            if ttl is not None:
                self.numeric(ttl)
                deadline = now + ttl
                if not math.isfinite(deadline):
                    raise ValueError("deadline")
            pending.append((key, value, deadline, ttl))
        self.purge(now)
        for key, value, deadline, ttl in pending:
            self.rows = [r for r in self.rows if r[0] != key]
            if ttl != 0:
                self.rows.append((key, value, deadline))
        self.rows = self.rows[-self.capacity :] if self.capacity else []

    def get(self, key, default=None, *, now):
        self.validate_now(now)
        if not isinstance(key, str):
            raise ValueError("key")  # noqa: TRY004 - the prompt requires ValueError.
        self.purge(now)
        found = next((r for r in self.rows if r[0] == key), None)
        if found is None:
            return default
        self.rows.remove(found)
        self.rows.append(found)
        return found[1]

    def snapshot(self, *, now):
        self.validate_now(now)
        self.purge(now)
        return [(k, v) for k, v, _ in self.rows]
