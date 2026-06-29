import math
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor


# ── NDArray helpers ───────────────────────────────────────────────────────────

def _ones_like(arr: NDArray) -> NDArray:
    # TODO: return an NDArray of ones with the same shape as arr
    return NDArray.fill(1)

def _zeros_like(arr: NDArray) -> NDArray:
    # TODO: return an NDArray of zeros with the same shape as arr
    return NDArray.fill(0)

def _reduce_grad(grad: NDArray, target_shape: tuple) -> NDArray:
    """Sum gradient over broadcast dimensions to match target_shape.

    Example: bias is (out,) but grad flowing back is (batch, out).
    We must sum over the batch axis to get a gradient of shape (out,).
    """
    # TODO: if shapes already match, return grad unchanged
    #       if grad is 2D (batch, out) and target is 1D (out,), sum axis=0
    pass


# ── Function base class ───────────────────────────────────────────────────────

class Function:
    """Base class for all differentiable operations.

    Every backward class inherits from here, stores inputs in
    self.saved_tensors, and implements apply(grad_output).
    """
    def __init__(self, *tensors):
        # TODO: store tensors as self.saved_tensors
        self.saved_tensors = list(tensors)
        pass

    def apply(self, grad_output: NDArray):
        raise NotImplementedError


# ── Arithmetic backward classes ───────────────────────────────────────────────

class AddBackward(Function):
    """∂(a+b)/∂a = 1  ∂(a+b)/∂b = 1"""
    def apply(self, grad_output):
        a, b = self.saved_tensors
        # TODO: grad for a and b are both grad_output
        #       but use _reduce_grad in case shapes differ (bias broadcasting)
        pass


class SubBackward(Function):
    """∂(a-b)/∂a = 1  ∂(a-b)/∂b = -1"""
    def apply(self, grad_output):
        a, b = self.saved_tensors
        # TODO: grad_a = grad_output, grad_b = -grad_output
        pass


class MulBackward(Function):
    """∂(a*b)/∂a = b  ∂(a*b)/∂b = a"""
    def apply(self, grad_output):
        a, b = self.saved_tensors
        # TODO: grad_a = grad_output * b.data
        #       grad_b = grad_output * a.data
        pass


class DivBackward(Function):
    """∂(a/b)/∂a = 1/b  ∂(a/b)/∂b = -a/b²"""
    def apply(self, grad_output):
        a, b = self.saved_tensors
        # TODO: grad_a = grad_output / b.data
        #       grad_b = -(grad_output * a.data / (b.data * b.data))
        pass


class ScalarMulBackward(Function):
    """∂(a * scalar)/∂a = scalar"""
    def __init__(self, a, scalar):
        self.saved_tensors = (a,)
        self.scalar = scalar

    def apply(self, grad_output):
        a, = self.saved_tensors
        # TODO: grad_a = grad_output * self.scalar
        pass


class ScalarDivBackward(Function):
    """∂(a / scalar)/∂a = 1/scalar"""
    def __init__(self, a, scalar):
        self.saved_tensors = (a,)
        self.scalar = scalar

    def apply(self, grad_output):
        a, = self.saved_tensors
        # TODO: grad_a = grad_output / self.scalar
        pass


class NegBackward(Function):
    """∂(-a)/∂a = -1"""
    def apply(self, grad_output):
        a, = self.saved_tensors
        # TODO: grad_a = -grad_output
        pass


# ── Matrix operation backward classes ────────────────────────────────────────

class MatmulBackward(Function):
    """∂(A@B)/∂A = grad@B.T  ∂(A@B)/∂B = A.T@grad"""
    def apply(self, grad_output):
        a, b = self.saved_tensors
        # TODO: grad_a = grad_output @ b.data.T
        #       grad_b = a.data.T @ grad_output
        pass


class TransposeBackward(Function):
    """∂(A.T)/∂A = grad.T"""
    def apply(self, grad_output):
        a, = self.saved_tensors
        # TODO: grad_a = grad_output.T
        pass


# ── Reduction backward classes ────────────────────────────────────────────────

class SumBackward(Function):
    """∂sum(a)/∂a[i] = 1 — broadcast scalar grad back to input shape"""
    def apply(self, grad_output):
        a, = self.saved_tensors
        # TODO: grad_output is shape (1,), extract the scalar
        #       fill an NDArray of a.data.shape with that scalar
        pass


# ── Activation backward classes ───────────────────────────────────────────────

class ReLUBackward(Function):
    """grad passes through where input > 0, blocked where input <= 0"""
    def apply(self, grad_output):
        a, = self.saved_tensors
        # TODO: for each element:
        #   if a.data._storage[i] > 0: pass grad through
        #   else: zero it out
        pass


# ── Loss backward classes ─────────────────────────────────────────────────────
# how can you implement a cross entropy backward loss man 
class CrossEntropyBackward(Function):
    """Fused softmax + NLL backward.

    grad_logits[i,j] = (softmax(logits)[i,j] - one_hot(target[i])[j]) / batch_size
    """
    def __init__(self, logits, targets):
        self.saved_tensors = (logits,)
        self.targets = targets

    def apply(self, grad_output):
        logits, = self.saved_tensors
        if not logits.requires_grad:
            return (None,)

        batch_size, n_classes = logits.data.shape
        scalar    = grad_output._storage[0]
        log_probs = _log_softmax_ndarray(logits.data)   # helper below
        grad      = NDArray(logits.data.shape, dtype=logits.data.dtype)

        # TODO: for each (i, j):
        #   softmax_ij = math.exp(log_probs[i, j])
        #   one_hot    = 1.0 if j == correct class else 0.0
        #   grad[i, j] = (softmax_ij - one_hot) / batch_size * scalar
        pass


def _log_softmax_ndarray(data: NDArray) -> NDArray:
    """Row-wise log-softmax on a 2D NDArray (helper used inside backward)."""
    rows, cols = data.shape
    result     = NDArray(data.shape, dtype=data.dtype)
    for i in range(rows):
        row_max = max(data[i, j] for j in range(cols))
        exps    = [math.exp(data[i, j] - row_max) for j in range(cols)]
        log_sum = math.log(sum(exps))
        for j in range(cols):
            result[i, j] = (data[i, j] - row_max) - log_sum
    return result


# ── Tensor.backward ───────────────────────────────────────────────────────────

def _backward(self, gradient=None):
    if not self.requires_grad:
        return

    # TODO step 1: if gradient is None and this is a scalar (size==1),
    #              seed it with ones_like(self.data)
    #              otherwise raise ValueError

    # TODO step 2: accumulate gradient
    #   if self.grad is None: self.grad = zeros_like(self.data)
    #   self.grad += gradient

    # TODO step 3: if self._grad_fn is not None:
    #   call self._grad_fn.apply(gradient) → gets a tuple of grads
    #   zip with self._grad_fn.saved_tensors
    #   for each (tensor, grad): if tensor.requires_grad and grad is not None
    #       call tensor.backward(grad)
    pass


# ── enable_autograd ───────────────────────────────────────────────────────────

def enable_autograd(quiet=False):
    """Monkey-patch Tensor, ReLU, and CrossEntropyLoss to build a computation
    graph during the forward pass so loss.backward() can propagate gradients.
    """
    from tinytorch.activations import ReLU
    from tinytorch.losses import CrossEntropyLoss, log_softmax

    # helper: build a tracked result tensor
    def _tracked(data, rg, fn_cls, inputs):
        # TODO: create Tensor(data, requires_grad=rg)
        #       if rg: attach fn_cls(*inputs) as _grad_fn, set is_leaf=False
        pass

    # ── patched Tensor ops ────────────────────────────────────────────────────

    def _add(self, other):
        if isinstance(other, Tensor):
            # TODO: use _tracked with AddBackward
            pass
        return Tensor(self.data + other, requires_grad=self.requires_grad)

    def _sub(self, other):
        if isinstance(other, Tensor):
            # TODO: use _tracked with SubBackward
            pass
        return Tensor(self.data - other, requires_grad=self.requires_grad)

    def _mul(self, other):
        if isinstance(other, Tensor):
            # TODO: use _tracked with MulBackward
            pass
        # scalar case — use ScalarMulBackward
        t = Tensor(self.data * other, requires_grad=self.requires_grad)
        # TODO: if self.requires_grad, attach ScalarMulBackward and set is_leaf=False
        return t

    def _truediv(self, other):
        if isinstance(other, Tensor):
            # TODO: use _tracked with DivBackward
            pass
        t = Tensor(self.data / other, requires_grad=self.requires_grad)
        # TODO: if self.requires_grad, attach ScalarDivBackward and set is_leaf=False
        return t

    def _matmul(self, other):
        # TODO: use _tracked with MatmulBackward
        pass

    def _neg(self):
        t = Tensor(-self.data, requires_grad=self.requires_grad)
        # TODO: if requires_grad, attach NegBackward and set is_leaf=False
        return t

    def _T_prop(self):
        t = Tensor(self.data.T, requires_grad=self.requires_grad)
        # TODO: if requires_grad, attach TransposeBackward and set is_leaf=False
        return t

    def _sum(self):
        s = self.data.sum()
        t = Tensor(NDArray((1,), data=[s]), requires_grad=self.requires_grad)
        # TODO: if requires_grad, attach SumBackward and set is_leaf=False
        return t

    # ── apply patches ─────────────────────────────────────────────────────────
    Tensor.__add__     = _add
    Tensor.__sub__     = _sub
    Tensor.__mul__     = _mul
    Tensor.__truediv__ = _truediv
    Tensor.__matmul__  = _matmul
    Tensor.__neg__     = _neg
    Tensor.T           = property(_T_prop)
    Tensor.sum         = _sum
    Tensor.matmul      = _matmul
    Tensor.backward    = _backward
    Tensor.zero_grad   = lambda self: setattr(self, 'grad', None)

    # ── patch ReLU.forward ────────────────────────────────────────────────────
    def _relu_forward(self, x):
        t = Tensor(x.data.maximum(0), requires_grad=x.requires_grad)
        # TODO: if x.requires_grad, attach ReLUBackward and set is_leaf=False
        return t

    ReLU.forward = _relu_forward

    # ── patch CrossEntropyLoss.forward ────────────────────────────────────────
    def _ce_forward(self, logits, targets):
        log_probs  = log_softmax(logits, dim=-1)
        batch_size = logits.data.shape[0]
        total = 0.0
        for i in range(batch_size):
            c = int(targets.data[i])
            total += log_probs.data[i, c]
        ce = -total / batch_size
        t  = Tensor(NDArray((1,), data=[ce]), requires_grad=logits.requires_grad)
        # TODO: if logits.requires_grad, attach CrossEntropyBackward and set is_leaf=False
        return t

    CrossEntropyLoss.forward = _ce_forward

    if not quiet:
        print("Autograd enabled — loss.backward() is now active.")
