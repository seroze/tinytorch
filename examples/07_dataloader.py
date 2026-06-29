"""
Module 07 — DataLoader
Dataset and DataLoader: batching and shuffling for training loops.
"""
from tinytorch.dataloader import Dataset, DataLoader
from tinytorch import Linear, Sequential, ReLU, CrossEntropyLoss
from tinytorch.ndarray import NDArray
from tinytorch.tensor import Tensor

print("=== DataLoader ===\n")

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
print("--- Dataset ---\n")

X = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [1.5, 2.5, 3.5],
    [4.5, 5.5, 6.5],
]
y = [0.0, 1.0, 2.0, 0.0, 1.0]

ds = Dataset(X, y)
print(f"samples  : {len(ds)}")
print(f"X shape  : {ds.X.data.shape}")
print(f"y shape  : {ds.y.data.shape}")

x0, y0 = ds[0]
print(f"\nds[0] X  : {[x0.data._storage[i] for i in range(3)]}")
print(f"ds[0] y  : {y0.data._storage[0]}")

x4, y4 = ds[4]
print(f"ds[4] X  : {[x4.data._storage[i] for i in range(3)]}")
print(f"ds[4] y  : {y4.data._storage[0]}")

print()

# ------------------------------------------------------------------
# DataLoader — no shuffle
# ------------------------------------------------------------------
print("--- DataLoader (shuffle=False) ---\n")

loader = DataLoader(ds, batch_size=2, shuffle=False)
print(f"batches  : {len(loader)}  (5 samples / batch_size=2 → 3 batches)")

for i, (xb, yb) in enumerate(loader):
    y_vals = [yb.data._storage[j] for j in range(yb.data.size)]
    print(f"batch {i}: X={xb.data.shape}  y={y_vals}")

print()

# ------------------------------------------------------------------
# DataLoader — with shuffle
# ------------------------------------------------------------------
print("--- DataLoader (shuffle=True) ---\n")

loader_shuffled = DataLoader(ds, batch_size=2, shuffle=True)
print("epoch 1:")
for i, (xb, yb) in enumerate(loader_shuffled):
    y_vals = [yb.data._storage[j] for j in range(yb.data.size)]
    print(f"  batch {i}: y={y_vals}")

print("epoch 2 (different order):")
for i, (xb, yb) in enumerate(loader_shuffled):
    y_vals = [yb.data._storage[j] for j in range(yb.data.size)]
    print(f"  batch {i}: y={y_vals}")

print()

# ------------------------------------------------------------------
# Full training loop skeleton
# ------------------------------------------------------------------
print("--- Training loop with DataLoader ---\n")

X_train = [[float(i + j) for j in range(4)] for i in range(20)]
y_train = [float(i % 3) for i in range(20)]

train_ds     = Dataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
model        = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
criterion    = CrossEntropyLoss()

print(f"dataset  : {len(train_ds)} samples")
print(f"batches  : {len(train_loader)} per epoch")
print()

for epoch in range(3):
    total_loss = 0.0
    for xb, yb in train_loader:
        logits = model(xb)
        loss   = criterion(logits, yb)
        total_loss += loss.data._storage[0]

    avg = total_loss / len(train_loader)
    print(f"epoch {epoch + 1}  avg loss: {avg:.4f}")

print()
print("DataLoader feeds (X_batch, y_batch) pairs each step.")
print("Shuffle=True means a different order every epoch — prevents the model")
print("from memorising the sequence rather than learning the pattern.")
