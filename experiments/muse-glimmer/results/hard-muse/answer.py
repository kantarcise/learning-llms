class ExpiringLRU:
    def __init__(self, capacity):
        if type(capacity) is not int:
            raise ValueError
        if capacity < 0:
            raise ValueError
        self._capacity = capacity
        self._data = {}  # key -> [value, deadline or None]
        self._order = []  # list of keys LRU -> MRU, maintained manually
        # alternative use dict for O(1) move, but list scan for move is ok for spec? need O(1)
        # Use dict of node links
        self._prev = {}
        self._next = {}
        self._head = None
        self._tail = None
        self._last_time = None

    # internal helpers
    def _validate_time(self, now):
        if type(now) is bool:
            raise ValueError
        if not isinstance(now, (int, float)):
            raise ValueError
        if not (now >= 0 and now == float(now) and not isinstance(now, bool)):
            raise ValueError
        if not (now == now and not isinstance(now, bool)):
            raise ValueError
        return float(now)

    def _validate_ttl(self, ttl):
        if ttl is None:
            return None
        if type(ttl) is bool:
            raise ValueError
        if not isinstance(ttl, (int, float)):
            raise ValueError
        if not (ttl >= 0 and ttl == float(ttl)):
            raise ValueError
        return float(ttl)

    def _validate_key(self, key):
        if not isinstance(key, str):
            raise ValueError
        return key

    def _check_time_monotonic(self, now):
        if self._last_time is not None and now < self._last_time - 1e-12:
            raise ValueError
        self._last_time = float(now)

    def _purge(self, now):
        # remove expired nodes
        # iterate over copy of keys
        to_remove = []
        for k, (v, dl) in self._data.items():
            if dl is not None and dl <= now + 1e-12:
                to_remove.append(k)
        for k in to_remove:
            self._remove_node(k)
            del self._data[k]

    def _add_node_to_tail(self, key):
        self._prev[key] = None
        self._next[key] = None
        if self._tail is None:
            self._head = self._tail = key
        else:
            self._prev[key] = self._tail
            self._next[self._tail] = key
            self._tail = key

    def _remove_node(self, key):
        p = self._prev.get(key)
        n = self._next.get(key)
        if p is not None:
            self._next[p] = n
        else:
            self._head = n
        if n is not None:
            self._prev[n] = p
        else:
            self._tail = p
        self._prev.pop(key, None)
        self._next.pop(key, None)

    def _move_to_tail(self, key):
        if self._tail is key:
            return
        self._remove_node(key)
        self._add_node_to_tail(key)

    def _evict_to_capacity(self):
        while len(self._data) > self._capacity:
            if self._head is None:
                break
            k = self._head
            self._remove_node(k)
            del self._data[k]

    # public API
    def put(self, key, value, *, now, ttl=None):
        now = self._validate_time(now)
        self._check_time_monotonic(now)
        key = self._validate_key(key)
        ttl_val = self._validate_ttl(ttl)
        self._purge(now)
        if self._capacity == 0:
            return
        if ttl_val is None:
            deadline = None
        else:
            deadline = now + ttl_val
            if not (deadline == float(deadline)):
                raise ValueError
            if ttl_val == 0:
                # remove if present
                if key in self._data:
                    self._remove_node(key)
                    del self._data[key]
                return
        if key in self._data:
            self._data[key] = [value, deadline]
            self._move_to_tail(key)
        else:
            self._data[key] = [value, deadline]
            self._add_node_to_tail(key)
        self._evict_to_capacity()

    def put_many(self, entries, *, now):
        now = self._validate_time(now)
        self._check_time_monotonic(now)
        # validate and collect
        collected = []
        try:
            for item in entries:
                if not (isinstance(item, tuple) or isinstance(item, list)) or len(item) != 3:
                    raise ValueError
                k, v, ttl = item
                key = self._validate_key(k)
                ttl_val = self._validate_ttl(ttl)
                collected.append((key, v, ttl_val))
        except Exception:
            # if validation fails, state unchanged
            raise
        # now apply atomically
        self._purge(now)
        if self._capacity == 0:
            return
        for key, value, ttl_val in collected:
            if ttl_val is None:
                deadline = None
            else:
                deadline = now + ttl_val
                if not (deadline == float(deadline)):
                    raise ValueError
                if ttl_val == 0:
                    if key in self._data:
                        self._remove_node(key)
                        del self._data[key]
                    continue
            if key in self._data:
                self._data[key] = [value, deadline]
                self._move_to_tail(key)
            else:
                self._data[key] = [value, deadline]
                self._add_node_to_tail(key)
        self._evict_to_capacity()

    def get(self, key, default=None, *, now):
        now = self._validate_time(now)
        self._check_time_monotonic(now)
        key = self._validate_key(key)
        self._purge(now)
        if key in self._data:
            value, deadline = self._data[key]
            self._move_to_tail(key)
            return value
        return default

    def snapshot(self, *, now):
        now = self._validate_time(now)
        self._check_time_monotonic(now)
        self._purge(now)
        # collect in order
        res = []
        cur = self._head
        while cur is not None:
            v, dl = self._data.get(cur, [None, None])
            res.append((cur, v))
            cur = self._next.get(cur)
        return res
