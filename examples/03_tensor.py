"""
Module 03 — Tensor
Wraps NDArray and adds autograd metadata: requires_grad, grad, grad_fn.
"""
from tinytorch import NDArray, Tensor

print("=== Tensor ===\n")

# --- creation ---
arr = NDArray((3,), data=[1.0, 2.0, 3.0])
t = Tensor(arr)
print(f"shape        : {t.data.shape}")
print(f"requires_grad: {t.requires_grad}  (False by default)")
print(f"grad         : {t.grad}           (None until backward)")
print(f"is_leaf      : {t.is_leaf}")

# --- with grad tracking ---
w = Tensor(NDArray((3,), data=[0.5, -1.0, 2.0]), requires_grad=True)
print(f"\nw.requires_grad: {w.requires_grad}")

# --- arithmetic: Tensor + scalar ---
x = Tensor(NDArray((3,), data=[1.0, 2.0, 3.0]))
y = x + 10
print(f"\nx + 10 : {[y.data[i,] for i in range(3)]}")

y = x * 3
print(f"x * 3  : {[y.data[i,] for i in range(3)]}")

y = x - 1
print(f"x - 1  : {[y.data[i,] for i in range(3)]}")

y = x / 2
print(f"x / 2  : {[y.data[i,] for i in range(3)]}")

# --- arithmetic: Tensor + Tensor ---
a = Tensor(NDArray((3,), data=[1.0, 2.0, 3.0]))
b = Tensor(NDArray((3,), data=[4.0, 5.0, 6.0]))

print(f"\na + b  : {[(a + b).data[i,] for i in range(3)]}")
print(f"a * b  : {[(a * b).data[i,] for i in range(3)]}")
print(f"a - b  : {[(a - b).data[i,] for i in range(3)]}")

# --- negation ---
neg = -a
print(f"\n-a     : {[neg.data[i,] for i in range(3)]}")

# --- matmul ---
A = Tensor(NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
B = Tensor(NDArray((3, 2), data=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]))
C = A @ B
print(f"\nA @ B shape: {C.data.shape}")
print(f"C[0,:]     : {[C.data[0, j] for j in range(2)]}")

# --- transpose ---
T_mat = Tensor(NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
print(f"\nT shape    : {T_mat.data.shape}")
print(f"T.T shape  : {T_mat.T.data.shape}")
print(f"T[0,1]={T_mat.data[0,1]}  T.T[1,0]={T_mat.T.data[1,0]}  (same element)")

# --- NDArray attribute delegation ---
# Tensor delegates unknown attrs to its NDArray via __getattr__
print(f"\nt.shape  (via delegation): {t.shape}")
print(f"t.ndim   (via delegation): {t.ndim}")
print(f"t.size   (via delegation): {t.size}")
print(f"t.strides(via delegation): {t.strides}")

# --- scalar convenience constructor ---
s = Tensor.scalar(3.14)
print(f"\nTensor.scalar(3.14) : shape={s.data.shape}  value={s.data[0,]}")
