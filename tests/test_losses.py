import unittest
import math
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor
from tinytorch.losses import log_softmax, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss


def t1(vals):
    return Tensor(NDArray((len(vals),), data=list(map(float, vals))))

def t2(rows):
    flat = [float(v) for row in rows for v in row]
    return Tensor(NDArray((len(rows), len(rows[0])), data=flat))


class TestLogSoftmax(unittest.TestCase):

    def test_1d_sums_to_one(self):
        x = t1([1.0, 2.0, 3.0])
        lp = log_softmax(x)
        total = sum(math.exp(lp.data[i]) for i in range(3))
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_1d_output_shape(self):
        x = t1([1.0, 2.0, 3.0])
        self.assertEqual(log_softmax(x).data.shape, (3,))

    def test_1d_all_negative(self):
        # log-probabilities must all be <= 0
        x = t1([1.0, 2.0, 3.0])
        lp = log_softmax(x)
        for i in range(3):
            self.assertLessEqual(lp.data[i], 0.0)

    def test_1d_numerical_stability(self):
        # Would overflow without max subtraction
        x = t1([1000.0, 1001.0, 1002.0])
        lp = log_softmax(x)
        total = sum(math.exp(lp.data[i]) for i in range(3))
        self.assertAlmostEqual(total, 1.0, places=8)
        for i in range(3):
            self.assertFalse(math.isnan(lp.data[i]))

    def test_1d_uniform_gives_equal_log_probs(self):
        x = t1([1.0, 1.0, 1.0])
        lp = log_softmax(x)
        expected = math.log(1.0 / 3.0)
        for i in range(3):
            self.assertAlmostEqual(lp.data[i], expected, places=10)

    def test_2d_each_row_sums_to_one(self):
        x = t2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        lp = log_softmax(x)
        for i in range(2):
            total = sum(math.exp(lp.data[i, j]) for j in range(3))
            self.assertAlmostEqual(total, 1.0, places=10)

    def test_2d_output_shape(self):
        x = t2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(log_softmax(x).data.shape, (2, 3))

    def test_2d_numerical_stability(self):
        x = t2([[1000.0, 1001.0, 1002.0], [500.0, 501.0, 502.0]])
        lp = log_softmax(x)
        for i in range(2):
            total = sum(math.exp(lp.data[i, j]) for j in range(3))
            self.assertAlmostEqual(total, 1.0, places=8)


class TestMSELoss(unittest.TestCase):

    def setUp(self):
        self.mse = MSELoss()

    def test_perfect_prediction_zero_loss(self):
        p = t1([1.0, 2.0, 3.0])
        t = t1([1.0, 2.0, 3.0])
        loss = self.mse(p, t)
        self.assertAlmostEqual(loss.data[0], 0.0)

    def test_known_value(self):
        # errors: [1, 2, 3] → squared: [1, 4, 9] → mean: 14/3
        p = t1([2.0, 4.0, 6.0])
        t = t1([1.0, 2.0, 3.0])
        loss = self.mse(p, t)
        self.assertAlmostEqual(loss.data[0], 14.0 / 3.0, places=10)

    def test_symmetric(self):
        p = t1([1.0, 2.0])
        t = t1([3.0, 4.0])
        loss_pt = self.mse(p, t)
        loss_tp = self.mse(t, p)
        self.assertAlmostEqual(loss_pt.data[0], loss_tp.data[0], places=10)

    def test_scalar_output(self):
        p = t1([1.0, 2.0, 3.0])
        t = t1([1.0, 2.0, 3.0])
        loss = self.mse(p, t)
        self.assertEqual(loss.data.shape, (1,))

    def test_large_error_dominates(self):
        # MSE squares errors: a 10x larger error gets 100x the penalty
        small = self.mse(t1([1.0]), t1([0.0]))   # error=1, loss=1
        large = self.mse(t1([10.0]), t1([0.0]))  # error=10, loss=100
        self.assertAlmostEqual(large.data[0] / small.data[0], 100.0, places=8)

    def test_always_non_negative(self):
        p = t1([-5.0, 3.0, -1.0])
        t = t1([2.0, -1.0, 4.0])
        loss = self.mse(p, t)
        self.assertGreaterEqual(loss.data[0], 0.0)


class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.ce = CrossEntropyLoss()

    def test_scalar_output(self):
        logits = t2([[1.0, 2.0, 3.0]])
        targets = t1([2])
        loss = self.ce(logits, targets)
        self.assertEqual(loss.data.shape, (1,))

    def test_confident_correct_prediction_low_loss(self):
        # Very high logit for correct class → loss near 0
        logits = t2([[0.0, 0.0, 100.0]])
        targets = t1([2])
        loss = self.ce(logits, targets)
        self.assertAlmostEqual(loss.data[0], 0.0, places=3)

    def test_confident_wrong_prediction_high_loss(self):
        # Very high logit for wrong class → high loss
        logits = t2([[100.0, 0.0, 0.0]])
        targets = t1([2])
        loss = self.ce(logits, targets)
        self.assertGreater(loss.data[0], 90.0)

    def test_uniform_logits_loss(self):
        # Uniform logits → probability 1/3 → loss = log(3)
        logits = t2([[1.0, 1.0, 1.0]])
        targets = t1([0])
        loss = self.ce(logits, targets)
        self.assertAlmostEqual(loss.data[0], math.log(3), places=8)

    def test_batch_averages_samples(self):
        # Two identical samples → same loss as one sample
        logits = t2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        targets = t1([2, 2])
        loss_batch = self.ce(logits, targets)
        loss_single = self.ce(t2([[1.0, 2.0, 3.0]]), t1([2]))
        self.assertAlmostEqual(loss_batch.data[0], loss_single.data[0], places=10)

    def test_always_positive(self):
        logits = t2([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]])
        targets = t1([1, 0])
        loss = self.ce(logits, targets)
        self.assertGreater(loss.data[0], 0.0)

    def test_numerical_stability_large_logits(self):
        logits = t2([[1000.0, 1001.0, 1002.0]])
        targets = t1([2])
        loss = self.ce(logits, targets)
        self.assertFalse(math.isnan(loss.data[0]))
        self.assertFalse(math.isinf(loss.data[0]))


class TestBinaryCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.bce = BinaryCrossEntropyLoss()

    def test_perfect_positive_prediction(self):
        # Predicting 1.0 for label 1 → loss ≈ 0
        p = t1([1.0 - 1e-7])
        t = t1([1.0])
        loss = self.bce(p, t)
        self.assertAlmostEqual(loss.data[0], 0.0, places=4)

    def test_perfect_negative_prediction(self):
        # Predicting 0.0 for label 0 → loss ≈ 0
        p = t1([1e-7])
        t = t1([0.0])
        loss = self.bce(p, t)
        self.assertAlmostEqual(loss.data[0], 0.0, places=4)

    def test_wrong_confident_prediction_high_loss(self):
        # Predicting ~1.0 for label 0 → very high loss
        p = t1([1.0 - 1e-7])
        t = t1([0.0])
        loss = self.bce(p, t)
        self.assertGreater(loss.data[0], 10.0)

    def test_half_probability_known_value(self):
        # p=0.5, t=1 → loss = -log(0.5) = log(2) ≈ 0.693
        p = t1([0.5])
        t = t1([1.0])
        loss = self.bce(p, t)
        self.assertAlmostEqual(loss.data[0], math.log(2), places=5)

    def test_no_nan_at_boundaries(self):
        p = t1([0.0, 1.0])
        t = t1([0.0, 1.0])
        loss = self.bce(p, t)
        self.assertFalse(math.isnan(loss.data[0]))

    def test_always_positive(self):
        p = t1([0.3, 0.7, 0.5])
        t = t1([1.0, 0.0, 1.0])
        loss = self.bce(p, t)
        self.assertGreater(loss.data[0], 0.0)

    def test_scalar_output(self):
        p = t1([0.5, 0.5])
        t = t1([1.0, 0.0])
        self.assertEqual(self.bce(p, t).data.shape, (1,))


if __name__ == '__main__':
    unittest.main()
