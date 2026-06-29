"""
Module 02 — NDArray
Multi-dimensional array built on top of Storage.
Covers: creation, indexing, strides, arithmetic, matmul, math ops.
"""
from tinytorch import NDArray

print("=== NDArray ===\n")

# --- creation ---
a = NDArray((3,), data=[1.0, 2.0, 3.0])
print(f"1D shape    : {a.shape}  ndim={a.ndim}  size={a.size}  strides={a.strides}")

b = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(f"2D shape    : {b.shape}  ndim={b.ndim}  size={b.size}  strides={b.strides}")

# --- factories ---
zeros = NDArray.zeros((2, 3))
ones  = NDArray.ones((2, 3))
rand  = NDArray.uniform((2, 3), low=-1.0, high=1.0)
print(f"\nzeros[0,0]  : {zeros[0, 0]}")
print(f"ones[1,2]   : {ones[1, 2]}")
print(f"uniform range sample: {rand._storage[0]:.4f}  (random, in [-1, 1])")

# --- indexing ---
print(f"\nb[0, 0]     : {b[0, 0]}   (row 0, col 0)")
print(f"b[1, 2]     : {b[1, 2]}   (row 1, col 2)")
print(f"b[-1, -1]   : {b[-1, -1]}  (last element, negative index)")

# --- element-wise arithmetic ---
x = NDArray((3,), data=[1.0, 2.0, 3.0])
y = NDArray((3,), data=[4.0, 5.0, 6.0])
print(f"\nx + y       : {[(x + y)[i,] for i in range(3)]}")
print(f"x * y       : {[(x * y)[i,] for i in range(3)]}")
print(f"x - y       : {[(x - y)[i,] for i in range(3)]}")
print(f"x / y       : {[(x / y)[i,] for i in range(3)]}")
print(f"-x          : {[(-x)[i,] for i in range(3)]}")

# --- scalar broadcast ---
print(f"\nx + 10      : {[(x + 10)[i,] for i in range(3)]}")
print(f"x * 2       : {[(x * 2)[i,] for i in range(3)]}")

# --- in-place ops ---
z = NDArray((2,), data=[1.0, 2.0])
z += NDArray((2,), data=[9.0, 8.0])
print(f"\nafter +=    : {[z[i,] for i in range(2)]}")

# --- reshape ---
flat = NDArray((6,), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
flat.reshape((2, 3))
print(f"\nreshaped (2,3): {[[flat[i, j] for j in range(3)] for i in range(2)]}")

# --- transpose ---
m = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
t = m.T
print(f"\nm.T shape   : {t.shape}  (was {m.shape})")
print(f"m[0,1]={m[0,1]}  m.T[1,0]={t[1,0]}  (same element)")

# --- matmul ---
A = NDArray((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
B = NDArray((3, 2), data=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
C = A @ B
print(f"\n(2,3) @ (3,2) = {C.shape}")
print(f"C[0,0]={C[0,0]}  C[0,1]={C[0,1]}")   # [58, 64]
print(f"C[1,0]={C[1,0]}  C[1,1]={C[1,1]}")   # [139, 154]

# --- math ops ---
import math
vals = NDArray((3,), data=[0.0, 1.0, -1.0])
print(f"\nexp([0,1,-1]) : {[round(vals.exp()[i,], 4) for i in range(3)]}")
print(f"tanh([0,1,-1]): {[round(vals.tanh()[i,], 4) for i in range(3)]}")
mixed = NDArray((4,), data=[-2.0, -0.5, 0.0, 3.0])
print(f"maximum(0)    : {[mixed.maximum(0)[i,] for i in range(4)]}")

# --- clip and sum ---
c = NDArray((5,), data=[-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\nclip(-1,1)  : {[c.clip(-1.0, 1.0)[i,] for i in range(5)]}")
print(f"sum()       : {NDArray((3,), data=[1.0, 2.0, 3.0]).sum()}")
