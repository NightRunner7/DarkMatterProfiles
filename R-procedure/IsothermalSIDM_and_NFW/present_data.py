import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# -------------------------
# CONFIG: your file path
# -------------------------
file_path = "Riso_M_10.00000_t_7.568_sigma_1.0_con_13.320.csv"

# -------------------------
# CONFIG: column indices (based on your dict insertion order)
# time = log10(Gyr) is column 0
# central-rho is column 5
# -------------------------
TIME_COL = 0
CENTRAL_RHO_COL = 2

# -------------------------
# Load the file (tab-separated, no header)
# -------------------------
df = pd.read_csv(file_path, sep="\t", header=None, engine="python")

# Quick sanity check (uncomment if needed)
# print("Shape:", df.shape)
# print(df.head(10))

# Extract relevant columns
t_log = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy()
rho_c = pd.to_numeric(df[CENTRAL_RHO_COL], errors="coerce").to_numpy()

# Keep finite values only
mask = np.isfinite(t_log) & np.isfinite(rho_c)
t_log = t_log[mask]
rho_c = rho_c[mask]

# Drop non-positive rho (log interpolation needs >0)
mask_pos = rho_c > 0
t_log = t_log[mask_pos]
rho_c = rho_c[mask_pos]

# -------------------------
# Your file likely contains MANY rows per snapshot (different radii),
# and "central-rho" is repeated for each radius at the same time.
# So: group by time and take a robust statistic (median).
# -------------------------
tmp = pd.DataFrame({"t_log": t_log, "rho_c": rho_c})
grp = tmp.groupby("t_log", as_index=False)["rho_c"].median()  # or .mean()

t_unique = grp["t_log"].to_numpy()
rho_unique = grp["rho_c"].to_numpy()

# Sort by time
order = np.argsort(t_unique)
t_unique = t_unique[order]
rho_unique = rho_unique[order]

# Remove duplicates in case of float weirdness (optional safety)
# (PCHIP wants strictly increasing x)
t_unique2 = [t_unique[0]]
rho_unique2 = [rho_unique[0]]
for i in range(1, len(t_unique)):
    if t_unique[i] > t_unique2[-1]:
        t_unique2.append(t_unique[i])
        rho_unique2.append(rho_unique[i])
t_unique = np.array(t_unique2)
rho_unique = np.array(rho_unique2)

# -------------------------
# Make a continuous curve:
# Interpolate in log(rho) vs log(time) to preserve positivity + shape
# -------------------------
log_rho = np.log10(rho_unique)

pchip = PchipInterpolator(t_unique, log_rho, extrapolate=False)

t_fine = np.linspace(t_unique.min(), t_unique.max(), 800)
log_rho_fine = pchip(t_fine)
rho_fine = 10**log_rho_fine

# Convert time to linear Gyr if you want nicer axis
tGyr_unique = 10**t_unique
tGyr_fine = 10**t_fine

# -------------------------
# Plot: central density vs time
# -------------------------
plt.figure(figsize=(7.2, 4.8))

# raw points (per snapshot)
plt.loglog(tGyr_unique, rho_unique, "o", markersize=4, label="snapshots (median per t)")

# smooth curve
plt.loglog(tGyr_fine, rho_fine, "-", linewidth=2.0, label="continuous (PCHIP in log–log)")

plt.xlabel(r"$t\ \mathrm{[Gyr]}$")
plt.ylabel(r"$\rho_c(t)$ (dimensionless, your $\tilde{\rho}$)")
plt.title("Central density evolution")
plt.grid(True, which="both", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()
