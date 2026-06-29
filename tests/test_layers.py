import unittest
import math
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor
from tinytorch.activations import ReLU, Softmax
from tinytorch.layers import Linear, Sequential


def tensor1d(values):
    return Tensor(NDArray((len(values),), data=list(map(float, values))))

def tensor2d(rows):
    flat = [float(v) for row in rows for v in row]
    return Tensor(NDArray((len(rows), len(rows[0])), data=flat))


class TestLinear(unittest.TestCase):

    def test_output_shape_1d(self):
        layer = Linear(3, 5)
        x = tensor1d([1.0, 2.0, 3.0])
        y = layer(x)
        self.assertEqual(y.data.shape, (5,))

    def test_output_shape_2d(self):
        layer = Linear(3, 5)
        x = tensor2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = layer(x)
        self.assertEqual(y.data.shape, (2, 5))

    def test_zero_weight_output_equals_bias(self):
        layer = Linear(3, 2)
        layer.weight.data.fill(0.0)
        layer.bias.data[0,] = 1.0
        layer.bias.data[1,] = 2.0
        x = tensor1d([5.0, 6.0, 7.0])
        y = layer(x)
        self.assertAlmostEqual(y.data[0,], 1.0)
        self.assertAlmostEqual(y.data[1,], 2.0)

    def test_identity_weight_no_bias(self):
        # W = I, b = 0  →  output == input
        layer = Linear(2, 2)
        layer.weight.data.fill(0.0)
        layer.weight.data[0, 0] = 1.0
        layer.weight.data[1, 1] = 1.0
        layer.bias.data.fill(0.0)
        x = tensor1d([3.0, 7.0])
        y = layer(x)
        self.assertAlmostEqual(y.data[0,], 3.0)
        self.assertAlmostEqual(y.data[1,], 7.0)

    def test_known_matmul(self):
        # W = [[1,2],[3,4]], b = [0,0], x = [1,1]
        # output = W @ x + b = [3, 7]
        layer = Linear(2, 2)
        layer.weight.data[0, 0] = 1.0; layer.weight.data[0, 1] = 2.0
        layer.weight.data[1, 0] = 3.0; layer.weight.data[1, 1] = 4.0
        layer.bias.data.fill(0.0)
        x = tensor1d([1.0, 1.0])
        y = layer(x)
        self.assertAlmostEqual(y.data[0,], 3.0)
        self.assertAlmostEqual(y.data[1,], 7.0)

    def test_weight_shape(self):
        layer = Linear(4, 8)
        self.assertEqual(layer.weight.data.shape, (8, 4))
        self.assertEqual(layer.bias.data.shape, (8,))

    def test_xavier_bound(self):
        # All weights should be within [-bound, bound]
        in_f, out_f = 4, 8
        bound = math.sqrt(6.0 / (in_f + out_f))
        layer = Linear(in_f, out_f)
        for i in range(layer.weight.data.size):
            w = layer.weight.data._storage[i]
            self.assertGreaterEqual(w, -bound)
            self.assertLessEqual(w, bound)

    def test_parameters_returns_weight_and_bias(self):
        layer = Linear(3, 5)
        params = layer.parameters()
        self.assertEqual(len(params), 2)
        self.assertIs(params[0], layer.weight)
        self.assertIs(params[1], layer.bias)

    def test_requires_grad_set(self):
        layer = Linear(3, 5)
        self.assertTrue(layer.weight.requires_grad)
        self.assertTrue(layer.bias.requires_grad)

    def test_batch_matches_single(self):
        # Batched forward of two samples should match running each individually
        layer = Linear(3, 4)
        x1 = tensor1d([1.0, 2.0, 3.0])
        x2 = tensor1d([4.0, 5.0, 6.0])
        y1 = layer(x1)
        y2 = layer(x2)
        xb = tensor2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        yb = layer(xb)
        for j in range(4):
            self.assertAlmostEqual(yb.data[0, j], y1.data[j,], places=10)
            self.assertAlmostEqual(yb.data[1, j], y2.data[j,], places=10)


class TestSequential(unittest.TestCase):

    def test_single_layer_passthrough(self):
        model = Sequential(Linear(3, 3))
        x = tensor1d([1.0, 2.0, 3.0])
        y = model(x)
        self.assertEqual(y.data.shape, (3,))

    def test_two_linear_layers(self):
        model = Sequential(Linear(4, 8), Linear(8, 2))
        x = tensor1d([1.0, 2.0, 3.0, 4.0])
        y = model(x)
        self.assertEqual(y.data.shape, (2,))

    def test_linear_relu_linear(self):
        model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
        x = tensor1d([1.0, -1.0, 0.5])
        y = model(x)
        self.assertEqual(y.data.shape, (2,))

    def test_mlp_softmax_sums_to_one(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3), Softmax())
        x = tensor1d([1.0, 2.0, 3.0, 4.0])
        y = model(x)
        total = sum(y.data[i,] for i in range(3))
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_parameters_collects_all(self):
        model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
        # ReLU has no parameters, so total = 2 layers * 2 params each = 4
        self.assertEqual(len(model.parameters()), 4)

    def test_empty_sequential(self):
        model = Sequential()
        x = tensor1d([1.0, 2.0])
        y = model(x)
        self.assertIs(y, x)


if __name__ == '__main__':
    unittest.main()
