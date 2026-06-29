"""
Module 05 — Layers
Linear layer and Sequential container: the building blocks of an MLP.
"""
from tinytorch import NDArray, Tensor, Linear, Sequential, ReLU, Softmax

print("=== Layers ===\n")

# ------------------------------------------------------------------
# Linear: single layer
# ------------------------------------------------------------------
print("--- Linear ---\n")

layer = Linear(4, 3)
print(layer)
print(f"weight shape : {layer.weight.data.shape}   (out_features, in_features)")
print(f"bias shape   : {layer.bias.data.shape}")
print(f"parameters   : {len(layer.parameters())}  (weight + bias)")

# single sample
x = Tensor(NDArray((4,), data=[1.0, 2.0, 3.0, 4.0]))
y = layer(x)
print(f"\ninput  shape : {x.data.shape}")
print(f"output shape : {y.data.shape}")
print(f"output values: {[round(y.data[i,], 4) for i in range(3)]}")

# batch of 2 samples
xb = Tensor(NDArray((2, 4), data=[1.0, 2.0, 3.0, 4.0,
                                   5.0, 6.0, 7.0, 8.0]))
yb = layer(xb)
print(f"\nbatch input  : {xb.data.shape}")
print(f"batch output : {yb.data.shape}")

# manual check: zero weights → output equals bias
layer.weight.data.fill(0.0)
layer.bias.data[0,] = 10.0
layer.bias.data[1,] = 20.0
layer.bias.data[2,] = 30.0
y_check = layer(x)
print(f"\n(W=0, b=[10,20,30]) → output: {[y_check.data[i,] for i in range(3)]}")

print()

# ------------------------------------------------------------------
# Sequential: chain layers
# ------------------------------------------------------------------
print("--- Sequential ---\n")

model = Sequential(
    Linear(4, 8),
    ReLU(),
    Linear(8, 3),
    Softmax(),
)
print(model)

x = Tensor(NDArray((4,), data=[1.0, 2.0, 3.0, 4.0]))
y = model(x)

print(f"input  shape : {x.data.shape}")
print(f"output shape : {y.data.shape}")
probs = [round(y.data[i,], 6) for i in range(3)]
total = sum(y.data[i,] for i in range(3))
print(f"output probs : {probs}")
print(f"sum          : {total:.10f}  (must be 1.0)")
print(f"parameters   : {len(model.parameters())}  (2 Linear layers × 2 each = 4 tensors)")

print()

# ------------------------------------------------------------------
# Deeper MLP
# ------------------------------------------------------------------
print("--- Deeper MLP: 4 → 16 → 8 → 3 ---\n")

deep = Sequential(
    Linear(4, 16), ReLU(),
    Linear(16, 8), ReLU(),
    Linear(8, 3),  Softmax(),
)
print(deep)

y_deep = deep(x)
probs_deep = [round(y_deep.data[i,], 6) for i in range(3)]
print(f"\noutput: {probs_deep}  sum={sum(y_deep.data[i,] for i in range(3)):.10f}")

print()

# ------------------------------------------------------------------
# Batch through MLP
# ------------------------------------------------------------------
print("--- Batch of 3 samples through MLP ---\n")

batch = Tensor(NDArray((3, 4), data=[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
]))

model2 = Sequential(Linear(4, 8), ReLU(), Linear(8, 3), Softmax())
out = model2(batch)

for i in range(3):
    row = [round(out.data[i, j], 4) for j in range(3)]
    s   = sum(out.data[i, j] for j in range(3))
    print(f"sample {i}: {row}  sum={s:.6f}")
