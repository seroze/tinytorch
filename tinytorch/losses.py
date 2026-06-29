import math
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor


"""If a class implements forward, __call__, __repr__ that is called module design
pattern(within the pytorch domain"""

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log-softmax via the log-sum-exp trick.

    Naive log(softmax(x)) materializes exp(x) which overflows for large logits.
    Instead: log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
    The max subtraction keeps every exponent in (-inf, 0] — no overflow possible.
    """
    data = x.data

    if dim < 0:
        dim = data.ndim + dim

    if data.ndim == 1:
        x_max = data.max_reduce()                      # scalar
        shifted = data - x_max                         # (C,) - scalar
        log_sum_exp = math.log(shifted.exp().sum())    # scalar
        return Tensor(shifted - log_sum_exp)

    # 2D batch: (N, C)
    x_max = data.max_reduce(axis=dim, keepdims=True)   # (N, 1)
    shifted = data - x_max                             # (N, C) - (N, 1)
    log_sum_exp = shifted.exp().sum(axis=dim, keepdims=True).log()  # (N, 1)
    return Tensor(shifted - log_sum_exp)               # (N, C) - (N, 1)


class MSELoss:
    """Mean squared error: mean((predictions - targets)^2)

    Use for regression. Squares the error so large mistakes are penalised
    disproportionately — an error of 2x contributes 4x to the loss.
    """

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Tensor ops so enable_autograd() can track the graph through the loss
        diff = predictions - targets        # Tensor sub
        squared = diff * diff               # Tensor mul
        return squared.sum() / squared.data.size  # Tensor sum + scalar div

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

    def __repr__(self):
        return "MSELoss()"


class CrossEntropyLoss:
    """Negative log-likelihood over a softmax distribution.

    Expects raw logits (not probabilities). log_softmax is applied internally
    so you never materialise exp(large_number) → inf.

    targets: 1D tensor of integer class indices, shape (N,)
    logits:  2D tensor of raw scores,            shape (N, C)
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = log_softmax(logits, dim=-1)        # (N, C)
        batch_size = logits.data.shape[0]

        total = 0.0
        for i in range(batch_size):
            c = int(targets.data[i])
            total += log_probs.data[i, c]

        ce = -total / batch_size
        result = Tensor(NDArray((1,), data=[ce]))

        # Store softmax probs so CrossEntropyBackward can use them
        # (set by enable_autograd when autograd is active)
        result._ce_logits  = logits
        result._ce_targets = targets
        return result

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)

    def __repr__(self):
        return "CrossEntropyLoss()"


class BinaryCrossEntropyLoss:
    """Binary cross-entropy for independent binary decisions.

    Expects probabilities in (0, 1) — apply sigmoid before passing in.
    targets: 1D tensor of 0s and 1s, shape (N,)

    Use CrossEntropyLoss for mutually exclusive multi-class problems.
    Use this when each output is an independent yes/no decision.
    """

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        eps = 1e-7
        preds = predictions.data.clip(eps, 1.0 - eps)  # guard against log(0)

        total = 0.0
        for i in range(preds.size):
            p = preds._storage[i]
            t = targets.data._storage[i]
            total += -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))

        bce = total / preds.size
        return Tensor(NDArray((1,), data=[bce]))

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

    def __repr__(self):
        return "BinaryCrossEntropyLoss()"
