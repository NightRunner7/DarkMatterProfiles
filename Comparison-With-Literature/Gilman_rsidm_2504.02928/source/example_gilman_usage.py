"""
example_gilman_usage.py

Example script showing the intended clean workflow.

Run from the directory where your original project modules are importable:
    - TruncatedNFWProfile.py
    - config.py

and where your tabulated data files live, e.g.
    data/Gilma_plot_points.csv
    data/Crossv_May7.csv
"""

import numpy as np
import matplotlib.pyplot as plt

from gilman_cross_sections import (
    YukawaNumerics,
    YukawaPartialWaveCrossSection,
    TabulatedCrossSection,
)
from gilman_averaging import MaxwellianAverager
from gilman_benchmark import (
    make_default_gilman_halos,
    compute_gilman_rows,
    print_gilman_rows,
    plot_raw_profiles,
    plot_sigma_eff_vs_mass,
)


# -----------------------------------------------------------------------------
# 1. Build profiles
# -----------------------------------------------------------------------------

numerics = YukawaNumerics(
    x_min=1e-4,
    range_factor=120.0,
    rtol=1e-8,
    atol=1e-10,
    use_small_x_initial_condition=True,
)

single_fit = YukawaPartialWaveCrossSection.from_gilman_model(
    # "single_profile_fit",
    "single_paper",
    numerics=numerics,
)

# multi_paper = YukawaPartialWaveCrossSection.from_gilman_model(
#     "multi_paper",
#     numerics=numerics,
# )

# Optional table profiles. Uncomment and adjust paths.
table_camilo = TabulatedCrossSection.from_csv("../data/Crossv_May7.csv", name="Camilo table")
table_krzysztof = TabulatedCrossSection.from_csv("../data/Gilma_plot_points.csv", name="Krzysztof digitization")


# -----------------------------------------------------------------------------
# 2. Raw profile comparison
# -----------------------------------------------------------------------------

v_grid = np.logspace(0, np.log10(300), 1200)
plot_raw_profiles(
    {
        "single first principles": single_fit,
        # "multi first principles": multi_paper,
        "Krzysztof table": table_krzysztof,
    },
    v_grid=v_grid,
    title="Gilman raw profiles, no averaging",
)
plt.show()



# -----------------------------------------------------------------------------
# 3. Halo benchmark and kappa averaging
# -----------------------------------------------------------------------------

halos = make_default_gilman_halos(explicit=True)

print("Step 1")
averager = MaxwellianAverager(p=5, vmin=1e-2, vmax=1e4, n_grid=1200)
print("Step 2")

rows_single = compute_gilman_rows(
    halos,
    sigma_profile=single_fit,
    averager=averager,
    velocity_mode="gilman_analytic",
    beta_for_tau=0.85,
)
print("Step 3")

print_gilman_rows(rows_single, title="Single peak reconstructed profile")
print("Step 4")

plot_sigma_eff_vs_mass(
    {"single profile fit": rows_single},
    title="Gilman K5 average from reconstructed profile",
)
plt.show()
print("Step 6")

