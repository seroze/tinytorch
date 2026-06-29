"""
Module 06 — Losses
log_softmax, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss.
"""
import math
from tinytorch import NDArray, Tensor, Linear, Sequential, ReLU, Sigmoid, Softmax
from tinytorch import log_softmax, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss

def t1(vals):
    return Tensor(NDArray((len(vals),), data=list(map(float, vals))))

def t2(rows):
    flat = [float(v) for row in rows for v in row]
    return Tensor(NDArray((len(rows), len(rows[0])), data=flat))

print("=== Losses ===\n")

# ------------------------------------------------------------------
# log_softmax — the numerical stability foundation
# ------------------------------------------------------------------
print("--- log_softmax ---\n")

x = t1([1.0, 2.0, 3.0])
lp = log_softmax(x)
probs = [round(math.exp(lp.data[i]), 4) for i in range(3)]
print(f"logits        : [1, 2, 3]")
print(f"log_softmax   : {[round(lp.data[i], 4) for i in range(3)]}")
print(f"exp(log_probs): {probs}  sum={sum(probs):.6f}")

# The stability point: large logits that would overflow naive softmax
x_big = t1([1000.0, 1001.0, 1002.0])
lp_big = log_softmax(x_big)
probs_big = [round(math.exp(lp_big.data[i]), 4) for i in range(3)]
print(f"\nlogits [1000,1001,1002] → same distribution as [0,1,2]:")
print(f"log_softmax   : {[round(lp_big.data[i], 4) for i in range(3)]}")
print(f"exp(log_probs): {probs_big}  sum={sum(probs_big):.6f}")

print()

# ------------------------------------------------------------------
# MSELoss — regression
# ------------------------------------------------------------------
print("--- MSELoss ---\n")

mse = MSELoss()

predictions = t1([200.0, 250.0, 300.0])
targets     = t1([195.0, 260.0, 290.0])
loss = mse(predictions, targets)
# errors: [5, -10, 10]  squared: [25, 100, 100]  mean: 75
print(f"predictions : [200, 250, 300]")
print(f"targets     : [195, 260, 290]")
print(f"errors      : [5, -10, 10]")
print(f"MSE loss    : {loss.data[0]:.4f}  (expected 75.0)")

# Perfect prediction
perfect = mse(t1([1.0, 2.0, 3.0]), t1([1.0, 2.0, 3.0]))
print(f"\nperfect prediction loss: {perfect.data[0]}")

# MSE penalises outliers quadratically
small_err = mse(t1([1.0]), t1([0.0]))   # error=1 → loss=1
large_err = mse(t1([10.0]), t1([0.0]))  # error=10 → loss=100
print(f"\nerror=1  → loss={small_err.data[0]:.1f}")
print(f"error=10 → loss={large_err.data[0]:.1f}  (100× not 10× — quadratic penalty)")

print()

# ------------------------------------------------------------------
# CrossEntropyLoss — multi-class classification
# ------------------------------------------------------------------
print("--- CrossEntropyLoss ---\n")

ce = CrossEntropyLoss()

# Confident correct prediction → low loss
logits  = t2([[0.0, 0.0, 100.0]])
targets = t1([2])
loss = ce(logits, targets)
print(f"logits [0, 0, 100], true class=2 → loss={loss.data[0]:.4f}  (near 0, very confident)")

# Confident wrong prediction → high loss
logits  = t2([[100.0, 0.0, 0.0]])
targets = t1([2])
loss = ce(logits, targets)
print(f"logits [100, 0, 0], true class=2 → loss={loss.data[0]:.4f}  (very high, confidently wrong)")

# Uniform logits → loss = log(num_classes)
logits  = t2([[1.0, 1.0, 1.0]])
targets = t1([0])
loss = ce(logits, targets)
print(f"uniform logits [1,1,1]          → loss={loss.data[0]:.4f}  (= log(3) = {math.log(3):.4f})")

# Batch of samples
logits  = t2([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])
targets = t1([0, 1])
loss = ce(logits, targets)
print(f"\nbatch loss: {loss.data[0]:.4f}")

# Numerical stability check
logits  = t2([[1000.0, 1001.0, 1002.0]])
targets = t1([2])
loss = ce(logits, targets)
print(f"large logits [1000,1001,1002]    → loss={loss.data[0]:.4f}  (stable, no nan/inf)")

print()

# ------------------------------------------------------------------
# BinaryCrossEntropyLoss — binary classification
# ------------------------------------------------------------------
print("--- BinaryCrossEntropyLoss ---\n")

bce = BinaryCrossEntropyLoss()

# Confident correct → near 0
p = t1([0.99])
t = t1([1.0])
loss = bce(p, t)
print(f"p=0.99, label=1 → loss={loss.data[0]:.4f}  (confident and correct, near 0)")

# Uncertain → log(2) ≈ 0.693
p = t1([0.5])
t = t1([1.0])
loss = bce(p, t)
print(f"p=0.5,  label=1 → loss={loss.data[0]:.4f}  (= log(2) = {math.log(2):.4f})")

# Confident wrong → high
p = t1([0.01])
t = t1([1.0])
loss = bce(p, t)
print(f"p=0.01, label=1 → loss={loss.data[0]:.4f}  (confident and wrong, high)")

# No nan at boundaries
p = t1([0.0, 1.0])
t = t1([0.0, 1.0])
loss = bce(p, t)
print(f"\nboundary values (p=0, p=1) → loss={loss.data[0]:.6f}  (clipping prevents log(0))")

print()

# ------------------------------------------------------------------
# Full MLP + loss
# ------------------------------------------------------------------
print("--- MLP forward pass + loss ---\n")

model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
criterion = CrossEntropyLoss()

x       = t2([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
targets = t1([2, 0])

logits = model(x)
loss   = criterion(logits, targets)

print(f"input shape  : {x.data.shape}")
print(f"logits shape : {logits.data.shape}")
print(f"loss         : {loss.data[0]:.4f}")
print(f"\nThis scalar is the only signal the optimizer ever sees.")
