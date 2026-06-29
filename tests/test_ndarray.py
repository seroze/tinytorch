import unittest
import math
from tinytorch.ndarray import NDArray


def make(values, shape=None):
    if shape is None:
        shape = (len(values),)
    return NDArray(shape, data=values)


class TestNDArrayCreation(unittest.TestCase):

    def test_shape(self):
        a = NDArray((3, 4))
        self.assertEqual(a.shape, (3, 4))

    def test_ndim(self):
        self.assertEqual(NDArray((5,)).ndim, 1)
        self.assertEqual(NDArray((3, 4)).ndim, 2)
        self.assertEqual(NDArray((2, 3, 4)).ndim, 3)

    def test_size(self):
        self.assertEqual(NDArray((3, 4)).size, 12)
        self.assertEqual(NDArray((2, 3, 4)).size, 24)

    def test_zero_initialized(self):
        a = NDArray((2, 3))
        for i in range(2):
            for j in range(3):
                self.assertEqual(a[i, j], 0.0)

    def test_init_with_data(self):
        a = NDArray((3,), data=[1.0, 2.0, 3.0])
        self.assertEqual(a[0,], 1.0)
        self.assertEqual(a[1,], 2.0)
        self.assertEqual(a[2,], 3.0)


class TestNDArrayStrides(unittest.TestCase):

    def test_strides_1d(self):
        self.assertEqual(NDArray((5,)).strides, (1,))

    def test_strides_2d(self):
        self.assertEqual(NDArray((3, 4)).strides, (4, 1))

    def test_strides_3d(self):
        self.assertEqual(NDArray((2, 3, 4)).strides, (12, 4, 1))

    def test_strides_single_element(self):
        self.assertEqual(NDArray((1, 1)).strides, (1, 1))


class TestNDArrayIndexing(unittest.TestCase):

    def setUp(self):
        self.a = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_getitem_corners(self):
        self.assertEqual(self.a[0, 0], 1.0)
        self.assertEqual(self.a[0, 2], 3.0)
        self.assertEqual(self.a[1, 0], 4.0)
        self.assertEqual(self.a[1, 2], 6.0)

    def test_setitem(self):
        self.a[0, 1] = 99.0
        self.assertEqual(self.a[0, 1], 99.0)

    def test_negative_index_last(self):
        self.assertEqual(self.a[-1, -1], 6.0)

    def test_negative_index_first(self):
        self.assertEqual(self.a[-2, 0], 1.0)

    def test_out_of_bounds_raises(self):
        with self.assertRaises(IndexError):
            _ = self.a[2, 0]

    def test_wrong_ndim_raises(self):
        with self.assertRaises(IndexError):
            _ = self.a[0, 0, 0]

    def test_1d_single_index(self):
        b = NDArray((4,), data=[10.0, 20.0, 30.0, 40.0])
        self.assertEqual(b[2,], 30.0)


class TestNDArrayFactories(unittest.TestCase):

    def test_zeros(self):
        a = NDArray.zeros((2, 3))
        for i in range(2):
            for j in range(3):
                self.assertEqual(a[i, j], 0.0)

    def test_ones(self):
        a = NDArray.ones((2, 3))
        for i in range(2):
            for j in range(3):
                self.assertEqual(a[i, j], 1.0)

    def test_fill(self):
        a = NDArray((2, 2))
        a.fill(7.0)
        for i in range(2):
            for j in range(2):
                self.assertEqual(a[i, j], 7.0)

    def test_fill_overwrites(self):
        a = NDArray((3,), data=[1.0, 2.0, 3.0])
        a.fill(0.0)
        for i in range(3):
            self.assertEqual(a[i,], 0.0)


class TestNDArrayReshape(unittest.TestCase):

    def test_reshape_changes_shape(self):
        a = NDArray((6,), data=[1, 2, 3, 4, 5, 6])
        a.reshape((2, 3))
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.ndim, 2)

    def test_reshape_updates_strides(self):
        a = NDArray((6,))
        a.reshape((2, 3))
        self.assertEqual(a.strides, (3, 1))

    def test_reshape_preserves_data(self):
        a = NDArray((6,), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a.reshape((2, 3))
        self.assertAlmostEqual(a[1, 2], 6.0)
        self.assertAlmostEqual(a[0, 0], 1.0)

    def test_reshape_wrong_size_raises(self):
        a = NDArray((6,))
        with self.assertRaises(AssertionError):
            a.reshape((2, 4))


class TestNDArrayTranspose(unittest.TestCase):

    def test_transpose_swaps_shape(self):
        a = NDArray((2, 3))
        a.transpose()
        self.assertEqual(a.shape, (3, 2))

    def test_transpose_data_access(self):
        # Row-major [1,2,3; 4,5,6], after transpose col-major
        a = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a.transpose()
        # After transpose shape is (3,2): a[0,0]=1, a[0,1]=4, a[1,0]=2, a[1,1]=5
        self.assertAlmostEqual(a[0, 0], 1.0)
        self.assertAlmostEqual(a[0, 1], 4.0)
        self.assertAlmostEqual(a[1, 0], 2.0)

    def test_transpose_non_2d_raises(self):
        a = NDArray((2, 3, 4))
        with self.assertRaises(AssertionError):
            a.transpose()


class TestNDArrayArithmetic(unittest.TestCase):

    def test_add_ndarray(self):
        a = make([1.0, 2.0, 3.0])
        b = make([4.0, 5.0, 6.0])
        c = a + b
        self.assertAlmostEqual(c[0,], 5.0)
        self.assertAlmostEqual(c[2,], 9.0)

    def test_add_scalar_int(self):
        a = make([1.0, 2.0, 3.0])
        c = a + 10
        self.assertAlmostEqual(c[0,], 11.0)
        self.assertAlmostEqual(c[2,], 13.0)

    def test_add_scalar_float(self):
        a = make([1.0, 2.0])
        c = a + 0.5
        self.assertAlmostEqual(c[0,], 1.5)

    def test_sub_ndarray(self):
        a = make([5.0, 6.0])
        b = make([1.0, 2.0])
        c = a - b
        self.assertAlmostEqual(c[0,], 4.0)
        self.assertAlmostEqual(c[1,], 4.0)

    def test_sub_scalar(self):
        a = make([10.0, 20.0])
        c = a - 5
        self.assertAlmostEqual(c[0,], 5.0)
        self.assertAlmostEqual(c[1,], 15.0)

    def test_mul_ndarray(self):
        a = make([2.0, 3.0])
        b = make([4.0, 5.0])
        c = a * b
        self.assertAlmostEqual(c[0,], 8.0)
        self.assertAlmostEqual(c[1,], 15.0)

    def test_mul_scalar(self):
        a = make([2.0, 3.0])
        c = a * 3
        self.assertAlmostEqual(c[0,], 6.0)
        self.assertAlmostEqual(c[1,], 9.0)

    def test_truediv_ndarray(self):
        a = make([6.0, 9.0])
        b = make([2.0, 3.0])
        c = a / b
        self.assertAlmostEqual(c[0,], 3.0)
        self.assertAlmostEqual(c[1,], 3.0)

    def test_truediv_scalar(self):
        a = make([6.0, 9.0])
        c = a / 3
        self.assertAlmostEqual(c[0,], 2.0)
        self.assertAlmostEqual(c[1,], 3.0)

    def test_neg(self):
        a = make([1.0, -2.0, 3.0])
        b = -a
        self.assertAlmostEqual(b[0,], -1.0)
        self.assertAlmostEqual(b[1,], 2.0)
        self.assertAlmostEqual(b[2,], -3.0)

    def test_iadd(self):
        a = make([1.0, 2.0])
        b = make([3.0, 4.0])
        a += b
        self.assertAlmostEqual(a[0,], 4.0)
        self.assertAlmostEqual(a[1,], 6.0)

    def test_isub(self):
        a = make([5.0, 6.0])
        b = make([1.0, 2.0])
        a -= b
        self.assertAlmostEqual(a[0,], 4.0)
        self.assertAlmostEqual(a[1,], 4.0)

    def test_imul(self):
        a = make([2.0, 3.0])
        b = make([4.0, 5.0])
        a *= b
        self.assertAlmostEqual(a[0,], 8.0)
        self.assertAlmostEqual(a[1,], 15.0)

    def test_shape_mismatch_raises(self):
        a = make([1.0, 2.0])
        b = make([1.0, 2.0, 3.0])
        with self.assertRaises(AssertionError):
            _ = a + b

    def test_eq_same(self):
        a = make([1.0, 2.0])
        b = make([1.0, 2.0])
        self.assertTrue(a == b)

    def test_eq_different_values(self):
        a = make([1.0, 2.0])
        b = make([1.0, 3.0])
        self.assertFalse(a == b)

    def test_eq_different_shapes(self):
        a = make([1.0, 2.0])
        b = make([1.0, 2.0, 3.0])
        self.assertFalse(a == b)

    def test_does_not_mutate_operands(self):
        a = make([1.0, 2.0])
        b = make([3.0, 4.0])
        _ = a + b
        self.assertAlmostEqual(a[0,], 1.0)
        self.assertAlmostEqual(b[0,], 3.0)


class TestNDArrayClip(unittest.TestCase):

    def test_clip_both_sides(self):
        a = NDArray((5,), data=[-2.0, -1.0, 0.0, 1.0, 2.0])
        b = a.clip(-1.0, 1.0)
        expected = [-1.0, -1.0, 0.0, 1.0, 1.0]
        for i, v in enumerate(expected):
            self.assertAlmostEqual(b[i,], v)

    def test_clip_no_change(self):
        a = NDArray((3,), data=[0.0, 0.5, 1.0])
        b = a.clip(-1.0, 2.0)
        for i in range(3):
            self.assertAlmostEqual(b[i,], a[i,])

    def test_clip_does_not_mutate(self):
        a = NDArray((2,), data=[-5.0, 5.0])
        _ = a.clip(-1.0, 1.0)
        self.assertAlmostEqual(a[0,], -5.0)


class TestNDArraySum(unittest.TestCase):

    def test_sum_1d(self):
        a = NDArray((3,), data=[1.0, 2.0, 3.0])
        self.assertAlmostEqual(a.sum(), 6.0)

    def test_sum_2d_all(self):
        a = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(a.sum(), 21.0)

    def test_sum_zeros(self):
        a = NDArray.zeros((4,))
        self.assertAlmostEqual(a.sum(), 0.0)


class TestNDArrayMatmul(unittest.TestCase):

    def test_basic(self):
        # (2,3) @ (3,2) → (2,2)
        a = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = NDArray((3, 2), data=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        c = a @ b
        self.assertEqual(c.shape, (2, 2))
        # Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        self.assertAlmostEqual(c[0, 0], 58.0)
        self.assertAlmostEqual(c[0, 1], 64.0)
        # Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        self.assertAlmostEqual(c[1, 0], 139.0)
        self.assertAlmostEqual(c[1, 1], 154.0)

    def test_identity(self):
        I = NDArray((2, 2), data=[1.0, 0.0, 0.0, 1.0])
        b = NDArray((2, 2), data=[3.0, 4.0, 5.0, 6.0])
        c = I @ b
        self.assertAlmostEqual(c[0, 0], 3.0)
        self.assertAlmostEqual(c[0, 1], 4.0)
        self.assertAlmostEqual(c[1, 0], 5.0)
        self.assertAlmostEqual(c[1, 1], 6.0)

    def test_shape_mismatch_raises(self):
        a = NDArray((2, 3))
        b = NDArray((2, 3))
        with self.assertRaises(AssertionError):
            _ = a @ b

    def test_1d_at_1d_raises(self):
        a = NDArray((3,))
        b = NDArray((3,))
        with self.assertRaises(AssertionError):
            _ = a @ b

    def test_1d_at_2d(self):
        # (k,) @ (k, n) → (n,)
        a = NDArray((3,), data=[1.0, 2.0, 3.0])
        b = NDArray((3, 2), data=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        c = a @ b
        self.assertEqual(c.shape, (2,))
        self.assertAlmostEqual(c[0], 4.0)   # 1*1 + 2*0 + 3*1
        self.assertAlmostEqual(c[1], 2.0)   # 1*0 + 2*1 + 3*0

    def test_2d_at_1d(self):
        # (m, k) @ (k,) → (m,)
        a = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = NDArray((3,), data=[1.0, 0.0, 1.0])
        c = a @ b
        self.assertEqual(c.shape, (2,))
        self.assertAlmostEqual(c[0], 4.0)   # 1+0+3
        self.assertAlmostEqual(c[1], 10.0)  # 4+0+6


class TestNDArrayMathOps(unittest.TestCase):

    def test_exp_zero(self):
        a = NDArray((1,), data=[0.0])
        self.assertAlmostEqual(a.exp()[0,], 1.0)

    def test_exp_one(self):
        a = NDArray((1,), data=[1.0])
        self.assertAlmostEqual(a.exp()[0,], math.e)

    def test_exp_negative(self):
        a = NDArray((1,), data=[-1.0])
        self.assertAlmostEqual(a.exp()[0,], 1.0 / math.e)

    def test_exp_shape_preserved(self):
        a = NDArray((2, 3))
        self.assertEqual(a.exp().shape, (2, 3))

    def test_tanh_zero(self):
        a = NDArray((1,), data=[0.0])
        self.assertAlmostEqual(a.tanh()[0,], 0.0)

    def test_tanh_positive(self):
        a = NDArray((1,), data=[1.0])
        self.assertAlmostEqual(a.tanh()[0,], math.tanh(1.0))

    def test_tanh_negative(self):
        a = NDArray((1,), data=[-1.0])
        self.assertAlmostEqual(a.tanh()[0,], math.tanh(-1.0))

    def test_tanh_range(self):
        a = NDArray((3,), data=[-100.0, 0.0, 100.0])
        b = a.tanh()
        self.assertAlmostEqual(b[0,], -1.0, places=5)
        self.assertAlmostEqual(b[2,], 1.0, places=5)

    def test_maximum_zeros_negatives(self):
        a = NDArray((4,), data=[-2.0, -0.5, 0.0, 3.0])
        b = a.maximum(0)
        self.assertEqual(b[0,], 0.0)
        self.assertEqual(b[1,], 0.0)
        self.assertEqual(b[2,], 0.0)
        self.assertEqual(b[3,], 3.0)

    def test_maximum_with_positive_threshold(self):
        a = NDArray((3,), data=[0.0, 1.0, 5.0])
        b = a.maximum(2.0)
        self.assertEqual(b[0,], 2.0)
        self.assertEqual(b[1,], 2.0)
        self.assertEqual(b[2,], 5.0)

    def test_math_ops_do_not_mutate(self):
        a = NDArray((2,), data=[1.0, -1.0])
        _ = a.exp()
        _ = a.tanh()
        _ = a.maximum(0)
        self.assertAlmostEqual(a[0,], 1.0)
        self.assertAlmostEqual(a[1,], -1.0)


if __name__ == '__main__':
    unittest.main()
