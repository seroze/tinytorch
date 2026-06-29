import unittest
from tinytorch.dataloader import Dataset, DataLoader


X5 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
y5 = [0.0, 1.0, 0.0, 1.0, 0.0]


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.ds = Dataset(X5, y5)

    def test_len(self):
        self.assertEqual(len(self.ds), 5)

    def test_X_shape(self):
        self.assertEqual(self.ds.X.data.shape, (5, 2))

    def test_y_shape(self):
        self.assertEqual(self.ds.y.data.shape, (5,))

    def test_getitem_X_shape(self):
        x, y = self.ds[0]
        self.assertEqual(x.data.shape, (2,))

    def test_getitem_y_shape(self):
        x, y = self.ds[0]
        self.assertEqual(y.data.shape, (1,))

    def test_getitem_X_values(self):
        x, y = self.ds[0]
        self.assertAlmostEqual(x.data._storage[0], 1.0)
        self.assertAlmostEqual(x.data._storage[1], 2.0)

    def test_getitem_y_value(self):
        x, y = self.ds[0]
        self.assertAlmostEqual(y.data._storage[0], 0.0)

    def test_getitem_last(self):
        x, y = self.ds[4]
        self.assertAlmostEqual(x.data._storage[0], 9.0)
        self.assertAlmostEqual(y.data._storage[0], 0.0)

    def test_getitem_returns_tensors(self):
        from tinytorch.tensor import Tensor
        x, y = self.ds[0]
        self.assertIsInstance(x, Tensor)
        self.assertIsInstance(y, Tensor)

    def test_all_samples_accessible(self):
        for i in range(len(self.ds)):
            x, y = self.ds[i]
            self.assertEqual(x.data.shape, (2,))


class TestDataLoaderLen(unittest.TestCase):

    def test_exact_division(self):
        ds = Dataset([[1.0]] * 6, [0.0] * 6)
        self.assertEqual(len(DataLoader(ds, batch_size=2)), 3)

    def test_partial_last_batch(self):
        ds = Dataset([[1.0]] * 5, [0.0] * 5)
        self.assertEqual(len(DataLoader(ds, batch_size=2)), 3)

    def test_batch_size_larger_than_dataset(self):
        ds = Dataset([[1.0]] * 3, [0.0] * 3)
        self.assertEqual(len(DataLoader(ds, batch_size=10)), 1)

    def test_batch_size_one(self):
        ds = Dataset([[1.0]] * 5, [0.0] * 5)
        self.assertEqual(len(DataLoader(ds, batch_size=1)), 5)


class TestDataLoaderIter(unittest.TestCase):

    def setUp(self):
        self.ds = Dataset(X5, y5)

    def test_yields_correct_number_of_batches(self):
        loader = DataLoader(self.ds, batch_size=2)
        batches = list(loader)
        self.assertEqual(len(batches), 3)

    def test_full_batch_shapes(self):
        loader = DataLoader(self.ds, batch_size=2)
        xb, yb = next(iter(loader))
        self.assertEqual(xb.data.shape, (2, 2))
        self.assertEqual(yb.data.shape, (2,))

    def test_last_partial_batch_shape(self):
        loader = DataLoader(self.ds, batch_size=2)
        batches = list(loader)
        xb, yb = batches[-1]
        self.assertEqual(xb.data.shape, (1, 2))
        self.assertEqual(yb.data.shape, (1,))

    def test_all_samples_covered_no_shuffle(self):
        loader = DataLoader(self.ds, batch_size=2, shuffle=False)
        seen = []
        for xb, yb in loader:
            for i in range(yb.data.size):
                seen.append(yb.data._storage[i])
        self.assertEqual(len(seen), 5)
        self.assertAlmostEqual(sum(seen), sum(y5))

    def test_no_shuffle_preserves_order(self):
        loader = DataLoader(self.ds, batch_size=5, shuffle=False)
        xb, yb = next(iter(loader))
        for i in range(5):
            self.assertAlmostEqual(xb.data[i, 0], X5[i][0])

    def test_shuffle_covers_all_samples(self):
        loader = DataLoader(self.ds, batch_size=2, shuffle=True)
        seen_y = []
        for xb, yb in loader:
            for i in range(yb.data.size):
                seen_y.append(yb.data._storage[i])
        self.assertEqual(len(seen_y), 5)
        self.assertAlmostEqual(sum(seen_y), sum(y5))

    def test_can_iterate_multiple_times(self):
        loader = DataLoader(self.ds, batch_size=2)
        first  = [yb.data.size for _, yb in loader]
        second = [yb.data.size for _, yb in loader]
        self.assertEqual(first, second)

    def test_batch_size_one(self):
        loader = DataLoader(self.ds, batch_size=1)
        for xb, yb in loader:
            self.assertEqual(xb.data.shape, (1, 2))
            self.assertEqual(yb.data.shape, (1,))

    def test_batch_size_equals_dataset(self):
        loader = DataLoader(self.ds, batch_size=5)
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        xb, yb = batches[0]
        self.assertEqual(xb.data.shape, (5, 2))


if __name__ == '__main__':
    unittest.main()
