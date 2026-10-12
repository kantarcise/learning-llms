"""Held-out contract checks; execute model answers only inside a sandbox."""

import random
import runpy
import sys
import unittest


class CacheTests(unittest.TestCase):
    def test_capacity_validation(self):
        for capacity in [-1, True, False, 1.0, None, "2"]:
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                Cache(capacity)

    def test_identity_none_and_empty_key(self):
        cache = Cache(3)
        value = object()
        cache.put("", value, now=0)
        cache.put("none", None, now=0)
        self.assertIs(cache.get("", now=0), value)
        self.assertIsNone(cache.get("none", "missing", now=0))
        self.assertEqual(cache.get("absent", "missing", now=0), "missing")

    def test_lru_hits_misses_and_snapshot(self):
        cache = Cache(2)
        cache.put("a", 1, now=0)
        cache.put("b", 2, now=0)
        self.assertEqual(cache.snapshot(now=0), [("a", 1), ("b", 2)])
        cache.get("missing", now=0)
        cache.put("c", 3, now=0)
        self.assertEqual(cache.snapshot(now=0), [("b", 2), ("c", 3)])
        cache.get("b", now=0)
        cache.put("d", 4, now=0)
        self.assertEqual(cache.snapshot(now=0), [("b", 2), ("d", 4)])

    def test_expiry_boundary(self):
        cache = Cache(3)
        cache.put("a", 1, now=0, ttl=2)
        cache.put("b", 2, now=0)
        self.assertEqual(cache.get("a", now=1.5), 1)
        self.assertEqual(cache.get("a", "expired", now=2), "expired")
        self.assertEqual(cache.snapshot(now=2), [("b", 2)])

    def test_expired_entries_purged_before_eviction(self):
        cache = Cache(2)
        cache.put("a", 1, now=0)
        cache.put("b", 2, now=0, ttl=1)
        cache.put("c", 3, now=1)
        self.assertEqual(cache.snapshot(now=1), [("a", 1), ("c", 3)])

    def test_replacement_resets_expiry_and_order(self):
        cache = Cache(2)
        cache.put("a", 1, now=0, ttl=1)
        cache.put("b", 2, now=0)
        cache.put("a", 3, now=0.5)
        self.assertEqual(cache.snapshot(now=1), [("b", 2), ("a", 3)])
        cache.put("a", 4, now=1, ttl=1)
        self.assertEqual(cache.snapshot(now=2), [("b", 2)])

    def test_zero_ttl_deletes_without_eviction(self):
        cache = Cache(2)
        cache.put_many([("a", 1, None), ("b", 2, None)], now=0)
        cache.put("missing", 0, now=0, ttl=0)
        cache.put("a", 9, now=0, ttl=0)
        self.assertEqual(cache.snapshot(now=0), [("b", 2)])

    def test_capacity_zero_still_validates_and_advances(self):
        cache = Cache(0)
        cache.put("a", object(), now=2)
        self.assertEqual(cache.snapshot(now=2), [])
        with self.assertRaises(ValueError):
            cache.snapshot(now=1)
        with self.assertRaises(ValueError):
            cache.put("a", 1, now=3, ttl=True)
        self.assertEqual(cache.snapshot(now=2), [])

    def test_all_calls_validate_time_without_mutation(self):
        for method in ["put", "put_many", "get", "snapshot"]:
            for bad in [-1, True, None, "3", float("nan"), float("inf")]:
                with self.subTest(method=method, bad=bad):
                    cache = Cache(1)
                    cache.put("a", 1, now=1, ttl=1)
                    args = {"put": ("b", 2), "put_many": ([],), "get": ("a",)}
                    with self.assertRaises(ValueError):
                        getattr(cache, method)(*args.get(method, ()), now=bad)
                    self.assertEqual(cache.snapshot(now=1), [("a", 1)])

    def test_backward_time_in_every_method(self):
        for method in ["put", "put_many", "get", "snapshot"]:
            with self.subTest(method=method):
                cache = Cache(1)
                cache.put("a", 1, now=2)
                args = {"put": ("b", 2), "put_many": ([],), "get": ("a",)}
                with self.assertRaises(ValueError):
                    getattr(cache, method)(*args.get(method, ()), now=1)
                self.assertEqual(cache.snapshot(now=2), [("a", 1)])

    def test_invalid_keys_and_ttl_are_atomic(self):
        for key, ttl in [
            (1, None),
            ([], None),
            ("a", True),
            ("a", -1),
            ("a", "3"),
            ("a", float("nan")),
            ("a", float("inf")),
        ]:
            with self.subTest(key=key, ttl=ttl):
                cache = Cache(1)
                cache.put("old", 1, now=1, ttl=1)
                with self.assertRaises(ValueError):
                    cache.put(key, 2, now=100, ttl=ttl)
                self.assertEqual(cache.snapshot(now=1), [("old", 1)])
        cache = Cache(1)
        cache.put("a", 1, now=0)
        with self.assertRaises(ValueError):
            cache.get([], now=100)
        self.assertEqual(cache.snapshot(now=0), [("a", 1)])

    def test_deadline_overflow_is_atomic(self):
        cache = Cache(1)
        cache.put("a", 1, now=0)
        with self.assertRaises(ValueError):
            cache.put("b", 2, now=1e308, ttl=1e308)
        self.assertEqual(cache.snapshot(now=0), [("a", 1)])

    def test_batch_eviction_happens_once(self):
        cache = Cache(2)
        cache.put_many([("a", 1, None), ("b", 2, None)], now=0)
        # Temporary writes and deletes must not cause premature eviction.
        cache.put_many([("c", 3, None), ("c", 4, 0), ("d", 5, None)], now=0)
        self.assertEqual(cache.snapshot(now=0), [("b", 2), ("d", 5)])
        cache = Cache(2)
        cache.put_many([("a", 1, None), ("b", 2, None)], now=0)
        cache.put_many([("c", 3, None), ("c", 4, 0)], now=0)
        self.assertEqual(cache.snapshot(now=0), [("a", 1), ("b", 2)])

    def test_duplicate_batch_keys_and_inputs(self):
        cache = Cache(2)
        value = object()
        entries = [["a", 1, None], ["b", 2, None], ["a", value, None]]
        original = [row.copy() for row in entries]
        cache.put_many(iter(entries), now=0)
        result = cache.snapshot(now=0)
        self.assertEqual([key for key, _ in result], ["b", "a"])
        self.assertIs(result[-1][1], value)
        self.assertEqual(entries, original)

    def test_malformed_batch_rolls_back_everything(self):
        for bad in [("bad",), ("bad", 2, None, 4), 3, None, ([], 2, None)]:
            with self.subTest(bad=bad):
                cache = Cache(2)
                cache.put_many([("a", 1, 1), ("b", 2, None)], now=0)
                with self.assertRaises(ValueError):
                    cache.put_many(iter([("c", 3, None), bad]), now=5)
                self.assertEqual(cache.snapshot(now=0), [("a", 1), ("b", 2)])

    def test_iterator_exception_preserved_and_atomic(self):
        cache = Cache(2)
        cache.put_many([("a", 1, 1), ("b", 2, None)], now=0)
        error = RuntimeError("input failure")

        def broken():
            yield ("c", 3, None)
            raise error

        with self.assertRaises(RuntimeError) as caught:
            cache.put_many(broken(), now=5)
        self.assertIs(caught.exception, error)
        self.assertEqual(cache.snapshot(now=0), [("a", 1), ("b", 2)])

    def test_empty_batch_and_miss_advance_time(self):
        for method, args in [("put_many", (iter([]),)), ("get", ("missing",))]:
            with self.subTest(method=method):
                cache = Cache(1)
                cache.put("a", 1, now=0, ttl=1)
                getattr(cache, method)(*args, now=2)
                with self.assertRaises(ValueError):
                    cache.snapshot(now=1)
                self.assertEqual(cache.snapshot(now=2), [])

    def test_seeded_valid_operations_against_list_reference(self):
        rng = random.Random(73)
        for capacity in range(4):
            cache, rows, now = Cache(capacity), [], 0
            for step in range(200):
                now += rng.choice([0, 0.5, 1])
                rows = [row for row in rows if row[2] is None or row[2] > now]
                if rng.randrange(2):
                    entries = [
                        (rng.choice("abcd"), step + i, rng.choice([None, 0, 1, 3]))
                        for i in range(rng.randrange(5))
                    ]
                    cache.put_many(iter(entries), now=now)
                    for key, value, ttl in entries:
                        rows = [row for row in rows if row[0] != key]
                        if ttl != 0:
                            rows.append(
                                (key, value, None if ttl is None else now + ttl)
                            )
                    rows = rows[-capacity:] if capacity else []
                else:
                    key = rng.choice("abcde")
                    found = next((row for row in rows if row[0] == key), None)
                    actual = cache.get(key, "missing", now=now)
                    self.assertEqual(actual, found[1] if found else "missing")
                    if found:
                        rows.remove(found)
                        rows.append(found)
                self.assertEqual(cache.snapshot(now=now), [(k, v) for k, v, _ in rows])


if __name__ == "__main__":
    Cache = runpy.run_path(sys.argv.pop(1))["ExpiringLRU"]
    unittest.main()
