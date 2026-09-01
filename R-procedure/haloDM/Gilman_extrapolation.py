import numpy as np

# ============================================================
# Gilman halo table
# ============================================================

M_vir_arr = np.array([1e7, 10**7.5, 1e8, 10**8.5, 1e9])
c_arr = np.array([21.21, 19.81, 18.42, 17.05, 15.69])

r200_arr = np.array([4.55, 6.68, 9.81, 14.4, 21.1])
log_rho_s_arr = np.array([7.57, 7.50, 7.42, 7.33, 7.24])


# ============================================================
# Helper functions
# ============================================================

def nfw_mass_factor(c):
    """
    NFW mass factor:
        f(c) = ln(1+c) - c/(1+c)
    """
    return np.log(1.0 + c) - c / (1.0 + c)


def compute_rho_s(M_vir, c, r200):
    """
    Compute rho_s from NFW relation:

        M_vir = 4 pi rho_s r_s^3 f(c)

    where:
        r_s = r200 / c

    Units:
        M_vir : Msun
        r200  : kpc
        rho_s : Msun / kpc^3
    """
    r_s = r200 / c
    rho_s = M_vir / (4.0 * np.pi * r_s**3 * nfw_mass_factor(c))
    return rho_s


def extrapolate_gilman_halo(M_new, M_vir_arr, c_arr, r200_arr):
    """
    Extrapolate Gilman halo parameters to a new halo mass.

    Procedure:
    1. Fit c = a log10(M_vir) + b.
    2. Extrapolate c(M_new).
    3. Compute r200 using r200 proportional to M^(1/3).
    4. Compute rho_s from the NFW mass relation.

    Returns:
        c_new, r200_new, r_s_new, rho_s_new, log_rho_s_new
    """

    # --------------------------------------------------------
    # 1. Fit concentration-mass relation
    # --------------------------------------------------------
    logM_arr = np.log10(M_vir_arr)

    a_c, b_c = np.polyfit(logM_arr, c_arr, deg=1)

    logM_new = np.log10(M_new)
    c_new = a_c * logM_new + b_c

    # --------------------------------------------------------
    # 2. Extrapolate r200 from M^(1/3)
    # --------------------------------------------------------
    # Use the first Gilman point as reference: M=1e7
    M_ref = M_vir_arr[0]
    r200_ref = r200_arr[0]

    r200_new = r200_ref * (M_new / M_ref)**(1.0 / 3.0)

    # --------------------------------------------------------
    # 3. Compute rho_s from NFW consistency
    # --------------------------------------------------------
    r_s_new = r200_new / c_new
    rho_s_new = compute_rho_s(M_new, c_new, r200_new)
    log_rho_s_new = np.log10(rho_s_new)

    return c_new, r200_new, r_s_new, rho_s_new, log_rho_s_new


# ============================================================
# Extrapolate to 10^6 Msun
# ============================================================

# M_new = (10**9.0)
M_arr = [
    10**7.0, 10**7.5, 10**8.0, 10**8.5, 10**9.0
]


# M_arr = [
#     10 ** 6.500, 10 ** 6.625, 10 ** 6.750, 10 ** 6.875, 10 ** 7.000,
#     10 ** 7.125, 10 ** 7.250, 10 ** 7.375, 10 ** 7.500, 10 ** 7.625,
#     10 ** 7.750, 10 ** 7.875, 10 ** 8.000, 10 ** 8.125, 10 ** 8.250,
#     10 ** 8.375, 10 ** 8.500, 10 ** 8.750, 10 ** 8.875, 10 ** 9.000,
#     10 ** 9.125, 10 ** 9.250
# ]
index=0
M_new=float(M_arr[index])
# M_new = (10**9.0)

c_output = np.array([])
for i in range(0, len(M_arr)):
    M_new = float(M_arr[i])

    c_new, r200_new, r_s_new, rho_s_new, log_rho_s_new = extrapolate_gilman_halo(
        M_new,
        M_vir_arr,
        c_arr,
        r200_arr
    )
    c_output = np.append(c_output, c_new)

print(c_output)

# print("Extrapolated Gilman halo:")
# print(f"M_vir       = {M_new:.3e} Msun")
# print(f"c           = {c_new:.4f}")
# print(f"r200        = {r200_new:.4f} kpc")
# print(f"r_s         = {r_s_new:.4f} kpc")
# print(f"rho_s       = {rho_s_new:.4e} Msun / kpc^3")
# print(f"log10(rho_s)= {log_rho_s_new:.4f}")