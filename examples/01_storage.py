"""
Module 01 — Storage
A flat 1D buffer: the raw memory layer underneath everything else.
"""
from tinytorch import Storage

print("=== Storage ===\n")

# --- creation ---
s = Storage(5)
print(f"size   : {s.size}")
print(f"len()  : {len(s)}")
print(f"s[0]   : {s[0]}  (zero-initialized by default)")

# --- set / get ---
s[0] = 1.5
s[2] = -3.14
s[4] = 99.0
print(f"\nafter writes → s[0]={s[0]}, s[2]={s[2]}, s[4]={s[4]}")

# --- overwrite ---
s[2] = 0.0
print(f"after overwrite → s[2]={s[2]}")

# --- all values ---
print(f"\nall values: {[s[i] for i in range(len(s))]}")

# --- dtype ---
s_float = Storage(3, dtype='f')   # single precision
s_float[0] = 3.14159
print(f"\nfloat32 storage s_float[0] = {s_float[0]:.5f}  (less precise than float64)")
