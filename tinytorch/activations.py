import math
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor


class ReLU:
    """max(0, x) — zeros negatives, preserves positives."""

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(x.data.maximum(0))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        pass

    def __repr__(self):
        return "ReLU()"


class Sigmoid:
    """1 / (1 + e^-x), numerically stable via piecewise formula."""

    def forward(self, x: Tensor) -> Tensor:
        clipped = x.data.clip(-500, 500)
        result = NDArray(clipped.shape, dtype=clipped.dtype)
        for i in range(clipped.size):
            z = clipped._storage[i]
            if z >= 0:
                result._storage[i] = 1.0 / (1.0 + math.exp(-z))
            else:
                e = math.exp(z)
                result._storage[i] = e / (1.0 + e)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        pass

    def __repr__(self):
        return "Sigmoid()"


class Tanh:
    """(e^x - e^-x) / (e^x + e^-x), output in (-1, 1)."""

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(x.data.tanh())

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        pass

    def __repr__(self):
        return "Tanh()"


class GELU:
    """x * sigmoid(1.702 * x) — smooth ReLU approximation used in transformers."""

    def forward(self, x: Tensor) -> Tensor:
        scaled = x * 1.702
        sig = Sigmoid()(scaled)
        return Tensor(sig.data * x.data)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        pass

    def __repr__(self):
        return "GELU()"


class Softmax:
    """e^xi / sum(e^xj), with max-subtraction to prevent overflow."""

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        data = x.data

        if dim < 0:
            dim = data.ndim + dim

        result = NDArray(data.shape, dtype=data.dtype)

        if data.ndim == 1:
            self._softmax_along(data, result, outer=1, inner=data.shape[0])
        elif data.ndim == 2:
            if dim == 1:
                # Each row is an independent distribution
                self._softmax_along(data, result, outer=data.shape[0], inner=data.shape[1])
            elif dim == 0:
                # Each column is an independent distribution
                batch, n = data.shape
                for j in range(n):
                    m = max(data[i, j] for i in range(batch))
                    exps = [math.exp(data[i, j] - m) for i in range(batch)]
                    total = sum(exps)
                    for i in range(batch):
                        result[i, j] = exps[i] / total
            else:
                raise ValueError(f"Invalid dim {dim} for 2D array")
        else:
            raise NotImplementedError("Softmax only supports 1D and 2D inputs")

        return Tensor(result)

    def _softmax_along(self, data, result, outer, inner):
        """Apply softmax along the inner (last) dimension for each outer slice."""
        if data.ndim == 1:
            m = max(data._storage[j] for j in range(inner))
            exps = [math.exp(data._storage[j] - m) for j in range(inner)]
            total = sum(exps)
            for j in range(inner):
                result._storage[j] = exps[j] / total
        else:
            for i in range(outer):
                m = max(data[i, j] for j in range(inner))
                exps = [math.exp(data[i, j] - m) for j in range(inner)]
                total = sum(exps)
                for j in range(inner):
                    result[i, j] = exps[j] / total

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        return self.forward(x, dim)

    def backward(self, grad: Tensor) -> Tensor:
        pass

    def __repr__(self):
        return "Softmax()"
