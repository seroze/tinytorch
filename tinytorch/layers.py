import math
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor


class Linear:
    """Learnable affine transformation: output = x @ W.T + b

    Args:
        in_features:  size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        # Xavier uniform: keeps activation variance stable across layers.
        # bound = sqrt(6 / (fan_in + fan_out))
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            NDArray.uniform((out_features, in_features), -bound, bound),
            requires_grad=True,
        )
        self.bias = Tensor(
            NDArray.zeros((out_features,)),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # (in,) → (out,)  or  (batch, in) → (batch, out)
        # Tensor ops so enable_autograd() can track the graph through this layer
        return x @ self.weight.T + self.bias

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"


class Sequential:
    """Chains layers so output of one feeds as input to the next."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def __repr__(self):
        lines = ',\n  '.join(repr(l) for l in self.layers)
        return f"Sequential(\n  {lines}\n)"
