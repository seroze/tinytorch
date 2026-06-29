import unittest
import math
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor
from tinytorch.activations import ReLU, Sigmoid, Tanh, GELU, Softmax


def tensor(values, shape=None):
    if shape is None:
        shape = (len(values),)
    return Tensor(NDArray(shape, data=list(map(float, values))))


def tensor2d(rows):
    flat = [float(v) for row in rows for v in row]
    shape = (len(rows), len(rows[0]))
    return Tensor(NDArray(shape, data=flat))


def get(t, *idx):
    return t.data[idx]


class TestReLU(unittest.TestCase):

    def setUp(self):
        self.relu = ReLU()

    def test_positive_unchanged(self):
        x = tensor([1.0, 2.0, 3.0])
        y = self.relu(x)
        self.assertAlmostEqual(get(y, 0), 1.0)
        self.assertAlmostEqual(get(y, 1), 2.0)
        self.assertAlmostEqual(get(y, 2), 3.0)

    def test_negative_zeroed(self):
        x = tensor([-1.0, -2.0, -0.001])
        y = self.relu(x)
        self.assertAlmostEqual(get(y, 0), 0.0)
        self.assertAlmostEqual(get(y, 1), 0.0)
        self.assertAlmostEqual(get(y, 2), 0.0)

    def test_zero_stays_zero(self):
        x = tensor([0.0])
        y = self.relu(x)
        self.assertAlmostEqual(get(y, 0), 0.0)

    def test_mixed(self):
        x = tensor([-3.0, 0.0, 5.0])
        y = self.relu(x)
        self.assertAlmostEqual(get(y, 0), 0.0)
        self.assertAlmostEqual(get(y, 1), 0.0)
        self.assertAlmostEqual(get(y, 2), 5.0)

    def test_returns_tensor(self):
        x = tensor([1.0])
        self.assertIsInstance(self.relu(x), Tensor)

    def test_does_not_mutate_input(self):
        x = tensor([-1.0, 2.0])
        _ = self.relu(x)
        self.assertAlmostEqual(get(x, 0), -1.0)


class TestSigmoid(unittest.TestCase):

    def setUp(self):
        self.sig = Sigmoid()

    def test_zero_gives_half(self):
        x = tensor([0.0])
        y = self.sig(x)
        self.assertAlmostEqual(get(y, 0), 0.5)

    def test_output_range(self):
        x = tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        y = self.sig(x)
        for i in range(5):
            v = y.data[i,]
            self.assertGreater(v, 0.0)
            self.assertLess(v, 1.0)

    def test_monotone_increasing(self):
        x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = self.sig(x)
        for i in range(4):
            self.assertLess(y.data[i,], y.data[i + 1,])

    def test_large_positive_near_one(self):
        x = tensor([100.0])
        y = self.sig(x)
        self.assertAlmostEqual(get(y, 0), 1.0, places=5)

    def test_large_negative_near_zero(self):
        x = tensor([-100.0])
        y = self.sig(x)
        self.assertAlmostEqual(get(y, 0), 0.0, places=5)

    def test_numerical_stability_no_nan(self):
        # Would overflow without the clip + piecewise formula
        x = tensor([600.0, -600.0])
        y = self.sig(x)
        self.assertFalse(math.isnan(get(y, 0)))
        self.assertFalse(math.isnan(get(y, 1)))

    def test_symmetry(self):
        # sigmoid(-x) = 1 - sigmoid(x)
        x = tensor([2.0])
        xneg = tensor([-2.0])
        self.assertAlmostEqual(get(self.sig(x), 0) + get(self.sig(xneg), 0), 1.0, places=10)

    def test_returns_tensor(self):
        self.assertIsInstance(self.sig(tensor([0.0])), Tensor)


class TestTanh(unittest.TestCase):

    def setUp(self):
        self.tanh = Tanh()

    def test_zero(self):
        x = tensor([0.0])
        self.assertAlmostEqual(get(self.tanh(x), 0), 0.0)

    def test_positive(self):
        x = tensor([1.0])
        self.assertAlmostEqual(get(self.tanh(x), 0), math.tanh(1.0))

    def test_negative(self):
        x = tensor([-1.0])
        self.assertAlmostEqual(get(self.tanh(x), 0), math.tanh(-1.0))

    def test_output_range(self):
        x = tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        y = self.tanh(x)
        for i in range(5):
            v = y.data[i,]
            self.assertGreater(v, -1.0)
            self.assertLess(v, 1.0)

    def test_saturation(self):
        x = tensor([100.0, -100.0])
        y = self.tanh(x)
        self.assertAlmostEqual(get(y, 0), 1.0, places=5)
        self.assertAlmostEqual(get(y, 1), -1.0, places=5)

    def test_odd_function(self):
        # tanh(-x) = -tanh(x)
        x = tensor([2.0])
        xneg = tensor([-2.0])
        self.assertAlmostEqual(get(self.tanh(x), 0), -get(self.tanh(xneg), 0), places=10)

    def test_returns_tensor(self):
        self.assertIsInstance(self.tanh(tensor([0.0])), Tensor)


class TestGELU(unittest.TestCase):

    def setUp(self):
        self.gelu = GELU()

    def test_zero(self):
        # gelu(0) = 0 * sigmoid(0) = 0
        x = tensor([0.0])
        self.assertAlmostEqual(get(self.gelu(x), 0), 0.0)

    def test_positive_output_positive(self):
        x = tensor([1.0, 2.0, 3.0])
        y = self.gelu(x)
        for i in range(3):
            self.assertGreater(y.data[i,], 0.0)

    def test_small_negative_slightly_negative(self):
        # For small negative x, gelu is slightly negative (smooth gate)
        x = tensor([-0.1])
        y = self.gelu(x)
        self.assertLess(get(y, 0), 0.0)

    def test_large_negative_near_zero(self):
        # sigmoid(1.702 * -10) ≈ 0, so gelu ≈ 0
        x = tensor([-10.0])
        y = self.gelu(x)
        self.assertAlmostEqual(get(y, 0), 0.0, places=3)

    def test_large_positive_near_identity(self):
        # sigmoid(1.702 * 10) ≈ 1, so gelu(x) ≈ x
        x = tensor([10.0])
        y = self.gelu(x)
        self.assertAlmostEqual(get(y, 0), 10.0, places=3)

    def test_formula_matches(self):
        # gelu(x) = x * sigmoid(1.702 * x)
        sig = Sigmoid()
        for xval in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            x = tensor([xval])
            expected = xval * get(sig(tensor([1.702 * xval])), 0)
            self.assertAlmostEqual(get(self.gelu(x), 0), expected, places=8)

    def test_returns_tensor(self):
        self.assertIsInstance(self.gelu(tensor([1.0])), Tensor)


class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.softmax = Softmax()

    def _sum_close(self, t, n, expected=1.0):
        total = sum(t.data[i,] for i in range(n))
        self.assertAlmostEqual(total, expected, places=10)

    def test_sums_to_one_1d(self):
        x = tensor([1.0, 2.0, 3.0])
        y = self.softmax(x)
        self._sum_close(y, 3)

    def test_all_positive_1d(self):
        x = tensor([-1.0, 0.0, 1.0])
        y = self.softmax(x)
        for i in range(3):
            self.assertGreater(y.data[i,], 0.0)

    def test_largest_gets_most_mass(self):
        x = tensor([1.0, 2.0, 10.0])
        y = self.softmax(x)
        self.assertGreater(y.data[2,], y.data[1,])
        self.assertGreater(y.data[1,], y.data[0,])

    def test_uniform_input_uniform_output(self):
        x = tensor([1.0, 1.0, 1.0])
        y = self.softmax(x)
        for i in range(3):
            self.assertAlmostEqual(y.data[i,], 1.0 / 3.0, places=10)

    def test_numerical_stability_large_values(self):
        # Without max subtraction, exp(1000) overflows to inf
        x = tensor([1000.0, 1001.0, 1002.0])
        y = self.softmax(x)
        self._sum_close(y, 3)
        for i in range(3):
            self.assertFalse(math.isnan(y.data[i,]))

    def test_2d_rows_sum_to_one(self):
        x = tensor2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = self.softmax(x, dim=-1)
        for i in range(2):
            row_sum = sum(y.data[i, j] for j in range(3))
            self.assertAlmostEqual(row_sum, 1.0, places=10)

    def test_2d_all_positive(self):
        x = tensor2d([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]])
        y = self.softmax(x)
        for i in range(2):
            for j in range(3):
                self.assertGreater(y.data[i, j], 0.0)

    def test_2d_rows_independent(self):
        # Each row is its own distribution; multiplying a row by a constant shouldn't affect other rows
        x = tensor2d([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
        y = self.softmax(x)
        # Second row should be uniform
        for j in range(3):
            self.assertAlmostEqual(y.data[1, j], 1.0 / 3.0, places=10)

    def test_known_values_1d(self):
        # softmax([1,2,3]): e^1,e^2,e^3 / (e^1+e^2+e^3)
        x = tensor([1.0, 2.0, 3.0])
        y = self.softmax(x)
        denom = math.exp(1) + math.exp(2) + math.exp(3)
        self.assertAlmostEqual(y.data[0,], math.exp(1) / denom, places=10)
        self.assertAlmostEqual(y.data[1,], math.exp(2) / denom, places=10)
        self.assertAlmostEqual(y.data[2,], math.exp(3) / denom, places=10)

    def test_returns_tensor(self):
        self.assertIsInstance(self.softmax(tensor([1.0, 2.0])), Tensor)

    def test_dim0_columns_sum_to_one(self):
        x = tensor2d([[1.0, 2.0], [3.0, 4.0]])
        y = self.softmax(x, dim=0)
        for j in range(2):
            col_sum = sum(y.data[i, j] for i in range(2))
            self.assertAlmostEqual(col_sum, 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
