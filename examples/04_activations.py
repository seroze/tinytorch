"""
Module 04 — Activations
Five activation functions: ReLU, Sigmoid, Tanh, GELU, Softmax.
"""
from tinytorch import NDArray, Tensor, ReLU, Sigmoid, Tanh, GELU, Softmax

def show(label, t, n=None):
    n = n or t.data.size
    vals = [round(t.data[i,], 4) for i in range(n)]
    print(f"{label:<28}: {vals}")

print("=== Activations ===\n")

x = Tensor(NDArray((5,), data=[-2.0, -1.0, 0.0, 1.0, 2.0]))
print(f"input: [-2, -1, 0, 1, 2]\n")

# --- ReLU ---
show("ReLU  max(0,x)",         ReLU()(x))
# expected: [0, 0, 0, 1, 2]

# --- Sigmoid ---
show("Sigmoid 1/(1+e^-x)",     Sigmoid()(x))
# expected: [0.119, 0.269, 0.5, 0.731, 0.881]

# --- Tanh ---
show("Tanh",                    Tanh()(x))
# expected: [-0.9640, -0.7616, 0.0, 0.7616, 0.9640]

# --- GELU ---
show("GELU  x*sigmoid(1.702x)", GELU()(x))
# expected: [-0.045, -0.159, 0.0, 0.841, 1.955]

print()

# --- Softmax ---
logits = Tensor(NDArray((3,), data=[1.0, 2.0, 3.0]))
probs  = Softmax()(logits)
show("Softmax([1,2,3])", probs, 3)
total = sum(probs.data[i,] for i in range(3))
print(f"{'sum of probs':<28}: {total:.10f}  (must be exactly 1.0)")

print()

# --- Numerical stability demo ---
# Without max subtraction exp(1000) → inf; with it, still works.
big = Tensor(NDArray((3,), data=[1000.0, 1001.0, 1002.0]))
safe_probs = Softmax()(big)
show("Softmax([1000,1001,1002])", safe_probs, 3)
print(f"{'sum':<28}: {sum(safe_probs.data[i,] for i in range(3)):.10f}")

print()

# --- Sigmoid symmetry ---
pos = Sigmoid()(Tensor(NDArray((1,), data=[2.0])))
neg = Sigmoid()(Tensor(NDArray((1,), data=[-2.0])))
print(f"sigmoid(2) + sigmoid(-2) = {pos.data[0,]:.6f} + {neg.data[0,]:.6f} = {pos.data[0,] + neg.data[0,]:.6f}  (always 1.0)")

# --- Tanh odd function ---
t1  = Tanh()(Tensor(NDArray((1,), data=[1.5])))
tn1 = Tanh()(Tensor(NDArray((1,), data=[-1.5])))
print(f"tanh(1.5) = {t1.data[0,]:.6f},  tanh(-1.5) = {tn1.data[0,]:.6f}  (negatives of each other)")

# --- 2D Softmax (batch) ---
print()
batch_logits = Tensor(NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
batch_probs  = Softmax()(batch_logits)
for i in range(2):
    row = [round(batch_probs.data[i, j], 4) for j in range(3)]
    s   = sum(batch_probs.data[i, j] for j in range(3))
    print(f"Softmax row {i}: {row}  sum={s:.6f}")
