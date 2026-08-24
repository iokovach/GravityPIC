"""
File format:
    line 1:      n_part
    lines 2 - N:  x y z vx vy vz w  
"""

import numpy as np

# --- settings -------------------------------------------------------------
n_part = 10
seed = 12345

# box bounds -- match these to your inputs file's boxsize
lo = np.array([0.0, 0.0, 0.0])
hi = np.array([1.0, 1.0, 1.0])

mass_weight = 1.0  # per-particle weight (w column)

out_file = "random10_ics.txt"
# ---------------------------------------------------------------------------

rng = np.random.default_rng(seed)

positions = rng.uniform(lo, hi, size=(n_part, 3))
velocities = np.zeros((n_part, 3))
weights = np.full((n_part, 1), mass_weight)

data = np.hstack([positions, velocities, weights])

with open(out_file, "w") as f:
    f.write(f"{n_part}\n")
    for row in data:
        f.write(" ".join(f"{val:.17g}" for val in row) + "\n")

print(f"Wrote {n_part} particles to {out_file}")