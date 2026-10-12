import unittest

from chunks import chunks


class ChunkTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunks([], 2), [])

    def test_exact_multiple(self):
        self.assertEqual(chunks([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_remainder(self):
        self.assertEqual(chunks([1, 2, 3], 2), [[1, 2], [3]])

    def test_generator(self):
        self.assertEqual(chunks(iter([1, 2, 3]), 2), [[1, 2], [3]])

    def test_invalid_sizes(self):
        for size in [True, False, 0, -1, 1.5, "2", None]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                chunks(iter([1, 2]), size)

    def test_oversized_chunk(self):
        self.assertEqual(chunks([1], 5), [[1]])

    def test_no_mutation(self):
        items = [1, 2, 3]
        chunks(items, 2)
        self.assertEqual(items, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
