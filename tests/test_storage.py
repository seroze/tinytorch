import unittest
from tinytorch.storage import Storage


class TestStorageInit(unittest.TestCase):

    def test_size_stored(self):
        s = Storage(5)
        self.assertEqual(s.size, 5)

    def test_zero_initialized(self):
        s = Storage(4)
        for i in range(4):
            self.assertEqual(s[i], 0.0)

    def test_len_matches_size(self):
        s = Storage(7)
        self.assertEqual(len(s), 7)

    def test_zero_size(self):
        s = Storage(0)
        self.assertEqual(s.size, 0)
        self.assertEqual(len(s), 0)


class TestStorageGetSet(unittest.TestCase):

    def test_set_and_get(self):
        s = Storage(3)
        s[0] = 1.5
        s[2] = -3.14
        self.assertAlmostEqual(s[0], 1.5)
        self.assertAlmostEqual(s[2], -3.14)

    def test_overwrite(self):
        s = Storage(3)
        s[1] = 42.0
        s[1] = -1.0
        self.assertEqual(s[1], -1.0)

    def test_default_dtype_double_precision(self):
        s = Storage(1)
        s[0] = 1.23456789012345
        self.assertAlmostEqual(s[0], 1.23456789012345, places=10)

    def test_custom_dtype_float(self):
        s = Storage(1, dtype='f')
        s[0] = 3.14
        self.assertAlmostEqual(s[0], 3.14, places=5)

    def test_all_indices_independent(self):
        s = Storage(5)
        for i in range(5):
            s[i] = float(i * 10)
        for i in range(5):
            self.assertAlmostEqual(s[i], float(i * 10))

    def test_negative_values(self):
        s = Storage(2)
        s[0] = -0.5
        s[1] = -1000.0
        self.assertAlmostEqual(s[0], -0.5)
        self.assertAlmostEqual(s[1], -1000.0)


if __name__ == '__main__':
    unittest.main()
