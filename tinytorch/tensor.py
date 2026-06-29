from tinytorch.ndarray import NDArray

"""
Storage
   │
   ▼
NDArray
   │
   ▼
Tensor
   │
   ├── grad
   ├── grad_fn
   ├── requires_grad
   └── is_leaf
           │
           ▼
      Function
           │
     ┌─────┴──────────┐
     ▼                ▼
 AddBackward      MulBackward
     ▼                ▼
      Computation Graph
           ▼
      loss.backward()
"""


class Tensor:

    def __init__(self, data, requires_grad=False):
        if isinstance(data, NDArray):
            self.data = data
        else:
            raise TypeError("Tensor expects an NDArray")

        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._grad_fn = None
        self.is_leaf = True

    # ------------------------
    # Metadata delegation
    # ------------------------

    
    def __getattr__(self, name):
        """
        Delegate unknown attributes/methods to NDArray.

        Tensor has attribute "shape" ?
        
        ↓
        
        No
        
        ↓
        
        calls __getattr__("shape")
        
        ↓
        
        return getattr(self.data, "shape")
        
        ↓
        
        NDArray.shape
        """
        return getattr(self.data, name)

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.data

        return Tensor(
            self.data + other,
            requires_grad=self.requires_grad
        )

    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = other.data

        return Tensor(
            self.data - other,
            requires_grad=self.requires_grad
        )

    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = other.data

        return Tensor(
            self.data * other,
            requires_grad=self.requires_grad
        )

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            other = other.data

        return Tensor(
            self.data / other,
            requires_grad=self.requires_grad
        )

    def __matmul__(self, other):
        return Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad
        )

    def __neg__(self):
        return Tensor(
            -self.data,
            requires_grad=self.requires_grad
        )

    @property
    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def backward(self, gradient=None):
        raise RuntimeError("Call enable_autograd() before calling backward()")

    def zero_grad(self):
        self.grad = None

    @classmethod
    def scalar(cls, value, dtype='d'):
        arr = NDArray(shape=(1,), dtype=dtype, data=[value])
        return cls(arr)

