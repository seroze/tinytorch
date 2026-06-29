
import math
from tinytorch.storage import Storage

class NDArray:

    def __init__(self, shape, dtype='d', data = None):
        self.shape = tuple(shape)
        self.dtype = dtype 
        self.ndim = len(shape)
        self.size = math.prod(shape) # total number of elements 
        self.strides = self._compute_strides()

        if data is not None:
            self._storage = Storage(self.size, dtype)
            for i,val in enumerate(data):
                self._storage[i] = val 
        else:
            self._storage = Storage(self.size, dtype) # zero-initialized 

    def _compute_strides(self):
        strides = [1]*self.ndim 
        for i in range(self.ndim-2, -1, -1):
            strides[i] = strides[i+1] * self.shape[i+1]
        return tuple(strides)

    def _flat_index(self, indices: tuple):
        if len(indices) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(indices)}")

        flat = 0 
        for i,(idx, dim) in enumerate(zip(indices, self.shape)):
            if not (-dim <= idx <dim):
                raise IndexError(f"Index {idx} out of bounds for axis {i} with size {dim}")
            if idx < 0:
                idx += dim 

            flat += idx * self.strides[i]
        return flat 


    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        # single integer on a 2D array → return that row as a 1D NDArray
        if self.ndim == 2 and len(indices) == 1:
            i = indices[0]
            if i < 0:
                i += self.shape[0]
            row = NDArray((self.shape[1],), dtype=self.dtype)
            start = i * self.strides[0]
            for j in range(self.shape[1]):
                row._storage[j] = self._storage[start + j * self.strides[1]]
            return row

        return self._storage[self._flat_index(indices)]

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices, )

        self._storage[self._flat_index(indices)] = value 

    def fill(self, value):
        for i in range(self.size):
            self._storage[i] = value

    @classmethod
    def zeros(cls, shape, dtype = 'd'):
        return cls(shape=shape, dtype = dtype) # data defautls to 0

    @classmethod
    def ones(cls, shape, dtype = 'd'):
        return cls(shape=shape, dtype = dtype, data = [1]*math.prod(shape))

    @classmethod
    def uniform(cls, shape, low=0.0, high=1.0, dtype='d'):
        import random
        data = [random.uniform(low, high) for _ in range(math.prod(shape))]
        return cls(shape=shape, dtype=dtype, data=data)

    @property
    def T(self):
        assert self.ndim == 2, "T only supports 2D arrays"
        rows, cols = self.shape
        result = NDArray((cols, rows), dtype=self.dtype)
        for i in range(rows):
            for j in range(cols):
                result[j, i] = self[i, j]
        return result

    def reshape(self, new_shape):
        # just change strides 
        # TODO: assert that new_shape has same product as old shape
        assert math.prod(new_shape) == self.size, (
            f"Cannot reshape {self.shape} into {new_shape}"
        )
        self.shape = tuple(new_shape)
        self.ndim  = len(new_shape)
        self.strides = self._compute_strides() # recompute, call this once with every change
        return self 

    def transpose(self):
        assert self.ndim == 2, "transpose only supports 2D arrays for now"
        self.shape = (self.shape[1], self.shape[0])
        self.strides = (self.strides[1], self.strides[0])  # swap strides, don't recompute
        return self

    def _elementwise(self, other, op):
        result = NDArray(self.shape, dtype=self.dtype)

        if isinstance(other, (int, float)):
            for i in range(self.size):
                result._storage[i] = op(self._storage[i], other)
        elif self.ndim == 2 and other.ndim == 1 and self.shape[1] == other.shape[0]:
            # (m, n) op (n,) — same value added to every row
            m, n = self.shape
            for i in range(m):
                for j in range(n):
                    result[i, j] = op(self[i, j], other[j])
        elif self.ndim == 2 and other.ndim == 2 and other.shape[1] == 1 and self.shape[0] == other.shape[0]:
            # (m, n) op (m, 1) — same value applied across every column of each row
            m, n = self.shape
            for i in range(m):
                for j in range(n):
                    result[i, j] = op(self[i, j], other[i, 0])
        else:
            assert self.shape == other.shape, (
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )
            for i in range(self.size):
                result._storage[i] = op(self._storage[i], other._storage[i])

        return result
                
    def __add__(self, other):
        # a + b 
        return self._elementwise(other, lambda a,b: a+b)
        
    def __mul__(self, other):
        # a * b
        return self._elementwise(other, lambda a,b: a*b)

    def __sub__(self, other):
        # a - b 
        return self._elementwise(other, lambda a,b: a-b)

    def __truediv__(self, other):
        # a / b 
        return self._elementwise(other, lambda a, b: a / b)
    
    def __neg__(self):
        # -a 
        result = NDArray(self.shape, dtype=self.dtype)
        for i in range(self.size):
            result._storage[i] = -self._storage[i]
        return result

    def __iadd__(self, other):
        # a += b __iadd__ is invoked 
        assert self.shape == other.shape
        for i in range(self.size):
            self._storage[i] += other._storage[i]
        return self   # returning self is correct HERE

    def __isub__(self, other):
        assert self.shape == other.shape
        for i in range(self.size):
            self._storage[i] -= other._storage[i]
        return self
    
    def __imul__(self, other):
        assert self.shape == other.shape
        for i in range(self.size):
            self._storage[i] *= other._storage[i]
        return self

    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        return all(self._storage[i] == other._storage[i] for i in range(self.size))

    def clip(self, min_val, max_val):
        result = NDArray(self.shape, dtype=self.dtype)
        for i in range(self.size):
            result._storage[i] = max(min_val, min(max_val, self._storage[i]))
        return result

    def log(self):
        result = NDArray(self.shape, dtype=self.dtype)
        for i in range(self.size):
            result._storage[i] = math.log(self._storage[i])
        return result

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            total = 0.0
            for i in range(self.size):
                total += self._storage[i]
            return total

        assert self.ndim == 2, "axis reduction only supports 2D arrays"
        m, n = self.shape

        if axis == 0:
            out_shape = (1, n) if keepdims else (n,)
            result = NDArray(out_shape, dtype=self.dtype)
            for j in range(n):
                total = 0.0
                for i in range(m):
                    total += self[i, j]
                if keepdims:
                    result[0, j] = total
                else:
                    result[j] = total

        elif axis == 1:
            out_shape = (m, 1) if keepdims else (m,)
            result = NDArray(out_shape, dtype=self.dtype)
            for i in range(m):
                total = 0.0
                for j in range(n):
                    total += self[i, j]
                if keepdims:
                    result[i, 0] = total
                else:
                    result[i] = total

        return result

    def max_reduce(self, axis=None, keepdims=False):
        if axis is None:
            m = self._storage[0]
            for i in range(1, self.size):
                if self._storage[i] > m:
                    m = self._storage[i]
            return m

        assert self.ndim == 2, "axis reduction only supports 2D arrays"
        rows, cols = self.shape

        if axis == 0:
            out_shape = (1, cols) if keepdims else (cols,)
            result = NDArray(out_shape, dtype=self.dtype)
            for j in range(cols):
                col_max = self[0, j]
                for i in range(1, rows):
                    if self[i, j] > col_max:
                        col_max = self[i, j]
                if keepdims:
                    result[0, j] = col_max
                else:
                    result[j] = col_max

        elif axis == 1:
            out_shape = (rows, 1) if keepdims else (rows,)
            result = NDArray(out_shape, dtype=self.dtype)
            for i in range(rows):
                row_max = self[i, 0]
                for j in range(1, cols):
                    if self[i, j] > row_max:
                        row_max = self[i, j]
                if keepdims:
                    result[i, 0] = row_max
                else:
                    result[i] = row_max

        return result
    
    def __matmul__(self, other):
        assert self.ndim in (1, 2), f"left operand must be 1D or 2D, got {self.ndim}D"
        assert other.ndim in (1, 2), f"right operand must be 1D or 2D, got {other.ndim}D"
        assert not (self.ndim == 1 and other.ndim == 1), "1D @ 1D is not supported"
        assert self.shape[-1] == other.shape[0], (
            f"Shape mismatch: {self.shape} @ {other.shape} — inner dims must match"
        )

        if self.ndim == 1 and other.ndim == 2:
            # (k,) @ (k, n) → (n,)
            k, n = other.shape
            result = NDArray((n,), dtype=self.dtype)
            for j in range(n):
                acc = 0.0
                for i in range(k):
                    acc += self[i] * other[i, j]
                result[j] = acc
            return result

        if self.ndim == 2 and other.ndim == 1:
            # (m, k) @ (k,) → (m,)
            m, k = self.shape
            result = NDArray((m,), dtype=self.dtype)
            for i in range(m):
                acc = 0.0
                for j in range(k):
                    acc += self[i, j] * other[j]
                result[i] = acc
            return result

        # (m, k) @ (k, n) → (m, n)
        m, k = self.shape
        _, n = other.shape
        result = NDArray((m, n), dtype=self.dtype)
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for l in range(k):
                    acc += self[i, l] * other[l, j]
                result[i, j] = acc
        return result

    def exp(self):
        result = NDArray(self.shape, dtype=self.dtype)
        for i in range(self.size):
            result._storage[i] = math.exp(self._storage[i])
        return result

    def tanh(self):
        result = NDArray(self.shape, dtype=self.dtype)
        for i in range(self.size):
            result._storage[i] = math.tanh(self._storage[i])
        return result

    def maximum(self, val):
        """Element-wise max(val, x) for a scalar threshold."""
        result = NDArray(self.shape, dtype=self.dtype)
        for i in range(self.size):
            v = self._storage[i]
            result._storage[i] = v if v > val else val
        return result

    def __repr__(self):
        if self.ndim == 1:
            data = [self._storage[i] for i in range(self.size)]
            return f"NDArray({data}, dtype='{self.dtype}')"
        elif self.ndim == 2:
            rows = []
            for i in range(self.shape[0]):
                row = [self._storage[i * self.strides[0] + j] for j in range(self.shape[1])]
                rows.append(str(row))
            return "NDArray([" + ",\n        ".join(rows) + f"], dtype='{self.dtype}')"
        return f"NDArray(shape={self.shape}, dtype='{self.dtype}')"