import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import units as uni
import numpy as np
import matplotlib as mpl
from RSIMD import build_sigma_m_eff_p5_interpolator
import config as cfg
# ##################################### SETTINGS ##################################### #
plot_save = False
output_plot_dir = "Results"

# --- the base directory will be our directory
OUR_DIR = os.path.dirname(os.path.abspath(__file__))

# --- select gravothermal directory
# gravothermal
DATA_GRAVOTHERMAL = os.path.join(OUR_DIR, 'Data_Extraction')
# isothermal
DATA_ISOTHERMAL_NFW_LoDen_DIR = os.path.join(OUR_DIR, 'Data_Extraction')
DATA_ISOTHERMAL_NFW_HiDen_DIR = os.path.join(OUR_DIR, 'Data_Extraction')

# --- select files:
gravothermal_SIDM_file_1 = 'SIDM_sol_M_9._t_50_sigma_119.0_beta_0.75.csv'
gravothermal_SIDM_file_2 = "SIDM_sol_M_8.8_t_50_sigma_15.2_beta_0.75.csv"
gravothermal_SIDM_file_4 = "SIDM_sol_M_10._t_50_sigma_2500_beta_0.75.csv"
gravothermal_SIDM_file_5 = "SIDM_sol_M_11._t_50_sigma_63.86_beta_0.75.csv"

gravothermal_RSIMD_file_1 = 'RSIDM_sol_M_9._t_20_beta_0.75.csv'
gravothermal_RSIMD_file_2 = 'RSIDM_sol_M_8.8_t_20_beta_0.75.csv'
gravothermal_RSIMD_file_3 = 'RSIDM_sol_M_9.2_t_20_beta_0.75.csv'
gravothermal_RSIMD_file_4 = 'RSIDM_sol_M_10._t_20_beta_0.75.csv'
gravothermal_RSIMD_file_5 = 'RSIDM_sol_M_11._t_20_beta_0.75.csv'

# --- SELECT SIDM
SIDM_file = gravothermal_SIDM_file_2

# --- SELECT RSIDM
RSIDM_file = gravothermal_RSIMD_file_2

# ##################################### SET DATA TO CLASS ############################################################ #
# --- SIDM
gravoEvolution_SIDM = gravoF.create_gravothermalData_from_file(SIDM_file, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_SIDM.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM.put_extra_parameters(10 ** 9.0, 119.0)

# --- RSIDM
gravoEvolution_RSIDM = gravoF.create_gravothermalData_from_file(RSIDM_file, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_RSIDM.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM.put_extra_parameters(10**9.0, 119.0)

# _, time_collapse, time_step_collapse = gravoEvolution_SIDM.find_collapse(elements=2, fixed_limit=10 ** 10)
# print(f"time collapse: {time_collapse} [Gyr], time step: {time_step_collapse}")




# ##################################### PLOTTING COMPARISON ########################################################## #
def apply_logx_linear_y_grid(_ax):
    """
    Apply grid and tick settings for:
    log-scale x axis and linear-scale y axis.
    """

    # Grid
    _ax.grid(which='major', alpha=0.45)
    _ax.grid(which='minor', alpha=0.15)

    # -------------------------
    # X axis (LOG)
    # -------------------------
    locmaj_x = mticker.LogLocator(base=10, numticks=12)
    locmin_x = mticker.LogLocator(
        base=10.0,
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        numticks=12
    )

    _ax.xaxis.set_major_locator(locmaj_x)
    _ax.xaxis.set_minor_locator(locmin_x)
    _ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # -------------------------
    # Y axis (LINEAR)
    # -------------------------

    # major ticks
    _ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

    # minor ticks
    _ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))

    # -------------------------
    # Tick style
    # -------------------------
    _ax.tick_params(
        'both',
        direction='in',
        top=True,
        right=True,
        length=10,
        width=1,
        which='major',
        zorder=301
    )

    _ax.tick_params(
        'both',
        direction='in',
        top=True,
        right=True,
        length=5,
        width=1,
        which='minor',
        zorder=301
    )

def apply_log_scale_grid(_ax):
    """
    Apply grid and log scale settings to a Matplotlib axis.

    Parameters:
        _ax (matplotlib.axes.Axes): The axis to which the settings will be applied.
    """
    # Grid settings
    _ax.grid(which='minor', alpha=0.2)
    _ax.grid(which='major', alpha=0.4)

    # Log-scale tick marks for x-axis
    locmaj_x = mticker.LogLocator(base=10, numticks=12)
    locmin_x = mticker.LogLocator(base=10.0,
                                  subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                  numticks=12)
    _ax.xaxis.set_major_locator(locmaj_x)
    _ax.xaxis.set_minor_locator(locmin_x)
    _ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # Log-scale tick marks for y-axis
    locmaj_y = mticker.LogLocator(base=10, numticks=12)
    locmin_y = mticker.LogLocator(base=10.0,
                                  subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                  numticks=12)
    _ax.yaxis.set_major_locator(locmaj_y)
    _ax.yaxis.set_minor_locator(locmin_y)
    _ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Tick length and direction
    _ax.tick_params('both', direction='in', top=True, right=True, length=10,
                    width=1, which='major', zorder=301)
    _ax.tick_params('both', direction='in', top=True, right=True, length=5,
                    width=1, which='minor', zorder=301)


def plot_veldisp_profiles(
    gravo,
    *,
    # --- choose what to plot ---
    steps=None,          # list[int] of time-step indices (0..time_steps-1)
    times_gyr=None,      # list[float] of physical times in Gyr (nearest snapshot used)
    # --- fallback auto sampling ---
    n_curves=60,
    auto_logtime=True,
    # --- styling ---
    color_by="logtime",  # "logtime" or "time" or "index"
    cmap_name="plasma",
    xlim=(1e-2, 1e2),
    ylim=(8e-2, 2.8e-1),
    lw=1.6,
    alpha=0.9,
    M_exp=9.0,
    title_extra=None,
    savepath=None,
    dpi=300,
):
    """
    Plot nu_hat(r) snapshots.

    Priority:
    - if steps is provided -> plot ONLY those steps
    - elif times_gyr is provided -> plot ONLY those physical times (nearest step)
    - else -> auto sample n_curves snapshots (log-spaced in time if auto_logtime)
    """

    # --------------------------
    # Decide which snapshots
    # --------------------------
    have_steps = steps is not None and len(steps) > 0
    have_times = times_gyr is not None and len(times_gyr) > 0

    if have_steps and have_times:
        raise ValueError("Pass only one of: steps=... or times_gyr=... (not both).")

    # Helper: get (r_hat, nu_hat, step, t_val) given either a step or a time
    def _snapshot_from_step(s):
        return gravo.return_veldis_hat_profile(
            time_argument=int(s),
            time_step_bool=True,
            return_time_step=True,
        )

    def _snapshot_from_time(t):
        return gravo.return_veldis_hat_profile(
            time_argument=float(t),
            time_step_bool=False,
            return_time_step=True,
        )

    snapshots = []  # list of tuples: (r_hat, nu_hat, step, t_val)

    if have_steps:
        # sanitize + keep unique + sorted
        steps_arr = np.array(steps, dtype=int)
        steps_arr = steps_arr[(steps_arr >= 0) & (steps_arr < gravo.time_steps)]
        steps_arr = np.unique(steps_arr)
        for s in steps_arr:
            snapshots.append(_snapshot_from_step(s))

    elif have_times:
        times_arr = np.array(times_gyr, dtype=float)
        times_arr = times_arr[np.isfinite(times_arr)]
        # keep order user gave (don’t sort unless you want)
        for t in times_arr:
            snapshots.append(_snapshot_from_time(t))

        # optional: drop duplicates (same nearest step)
        # (keep first occurrence)
        seen = set()
        uniq = []
        for snap in snapshots:
            step = int(snap[2])
            if step in seen:
                continue
            seen.add(step)
            uniq.append(snap)
        snapshots = uniq

    else:
        # auto sample
        tarr = np.asarray(gravo.data["time-no-repetition"], dtype=float)
        tarr = tarr[np.isfinite(tarr)]
        tarr = tarr[tarr > 0]
        if tarr.size < 2:
            raise RuntimeError("Not enough time snapshots to plot.")

        if n_curves >= tarr.size:
            tsamp = tarr
        else:
            if auto_logtime:
                logt = np.log10(tarr)
                tsamp = 10 ** np.linspace(logt.min(), logt.max(), n_curves)
            else:
                tsamp = np.linspace(tarr.min(), tarr.max(), n_curves)

        for t in tsamp:
            snapshots.append(_snapshot_from_time(t))

    # --------------------------
    # Prepare colormap
    # --------------------------
    cmap = mpl.cm.get_cmap(cmap_name)

    # Choose the scalar we map to color
    if color_by == "logtime":
        cvals = np.array([np.log10(s[3]) for s in snapshots])  # log10(t)
        cbar_label = r'$\log_{10}\!\left(t\,[\mathrm{Gyr}]\right)$'
    elif color_by == "time":
        cvals = np.array([s[3] for s in snapshots])            # t
        cbar_label = r'$t\,[\mathrm{Gyr}]$'
    elif color_by == "index":
        cvals = np.array([s[2] for s in snapshots])            # step index
        cbar_label = "time step"
    else:
        raise ValueError("color_by must be one of: 'logtime', 'time', 'index'")

    norm = mpl.colors.Normalize(vmin=float(np.min(cvals)), vmax=float(np.max(cvals)))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # --------------------------
    # Plot
    # --------------------------
    fig, ax = plt.subplots(figsize=(11.0, 7.0), constrained_layout=True)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_yscale('linear')

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(r'$\hat{r} = r/r_s$', fontsize=20)
    ax.set_ylabel(r'$\hat{\nu} = \nu(r)/\nu_0$', fontsize=20)
    ax.tick_params(labelsize=14)
    apply_logx_linear_y_grid(ax)

    for (r_hat, nu_hat, step, t_val), c in zip(snapshots, cvals):
        ax.plot(r_hat, nu_hat, lw=lw, alpha=alpha, color=cmap(norm(c)), zorder=2)

    # Title
    beta = getattr(gravo, "beta", None)
    beta_str = f"{beta:.3g}" if isinstance(beta, (float, int)) else f"{beta}"
    # title = rf"RSIDM velocity dispersion profiles $\hat{{\nu}}(\hat{{r}})$"
    title = rf"cSIDM velocity dispersion profiles $\hat{{\nu}}(\hat{{r}})$"
    subtitle = rf"$\beta={beta_str}$, $M=10^{{{M_exp}}}\,M_\odot$"
    if title_extra:
        subtitle += rf", {title_extra}"
    ax.set_title(title + "\n" + subtitle, fontsize=18, pad=10)

    # Colorbar
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if savepath:
        fig.savefig(savepath, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


plot_veldisp_profiles(
    gravoEvolution_SIDM,
    steps=[1, 200, 400, 800, 1000, 1200, 1300],
    color_by="logtime",   # still color by log(time), even though you selected by step
    cmap_name="plasma",
    xlim=(1e-2, 1e0),
    ylim=(0.07, 0.23),
    # ylim=(8e-2, 2.8e-1),
)
#########################################################################################################
# sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (nu_grid, sigma_grid) = \
#     build_sigma_m_eff_p5_interpolator(
#         sigma_m0=0.008,
#         m_GeV=0.02,
#         L=0,
#         Gamma=6e-12,
#         vR_kms=85.0,
#         nu_kms_min=0.1,
#         nu_kms_max=3000.0,
#         N=400
#     )
#
# gravoEvolution_RSIDM.build_interpolated_profiles(
#     rho_mode="loglog",
#     vel_mode="loglog",
#     stop_at_collapse=True,
# )
#
# gravoEvolution_RSIDM.build_scattering_matrix_from_interpolators(
#     sigma_m=sigma_eff_from_nu_kms,
#     sigma_input_vel_unit="km/s",
#     ngrid=10000,
# )
#
#
# time_arr, r1_arr_outer, roots_list, dbg = \
#     gravoEvolution_RSIDM.find_all_r1_roots_from_scattering_matrix(
#         pick="outermost"
#     )

##################################################
gravoEvolution_SIDM.build_interpolated_profiles(
    rho_mode="loglog",
    vel_mode="loglog",
    stop_at_collapse=True,
)

gravoEvolution_SIDM.build_scattering_matrix_from_interpolators(
    sigma_m=119.0,
    sigma_input_vel_unit="km/s",
    ngrid=1000,
)


time_arr, r1_arr_outer, roots_list, dbg = \
    gravoEvolution_SIDM.find_all_r1_roots_from_scattering_matrix(
        pick="outermost"
    )


fig, ax = plt.subplots(figsize=(11.0, 7.0))

for t, roots in zip(time_arr, roots_list):
    if len(roots) == 0:
        continue
    ax.scatter(np.full(len(roots), t), roots, s=10, color="black", alpha=0.7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("time [Gyr]")
ax.set_ylabel(r"all $r_1$ roots [$r_s$]")
ax.grid(alpha=0.3, which="both")
ax.set_title("RSIDM: all $r_1$ roots vs time")
plt.show()


r_inner = []
r_middle = []
r_outer = []

for roots in roots_list:

    roots = sorted(roots)

    if len(roots) >= 2:
        r_inner.append(roots[0])
    else:
        r_inner.append(np.nan)

    if len(roots) >= 3:
        r_middle.append(roots[1])
    else:
        r_middle.append(np.nan)

    if len(roots) >= 1:
        r_outer.append(roots[-1])
    else:
        r_outer.append(np.nan)

r_inner = np.array(r_inner)
r_middle = np.array(r_middle)
r_outer = np.array(r_outer)

print("len(r_inner)", len(np.unique(r_inner[~np.isnan(r_inner)])))
print("len(r_middle)", len(np.unique(r_middle[~np.isnan(r_middle)])))
print("len(r_outer)", len(np.unique(r_outer[~np.isnan(r_outer)])))


nroots = np.array([len(rr) for rr in roots_list])

fig, ax = plt.subplots(figsize=(11.0, 7.0))
ax.plot(time_arr, nroots, lw=1.8)
ax.set_xscale("log")
ax.set_xlabel("time [Gyr]")
ax.set_ylabel("number of roots")
ax.grid(alpha=0.3, which="both")
ax.set_title("RSIDM: number of $r_1$ roots")
plt.show()


def plot_branch_segmented(ax, t, r, *, color, label, s=26, lw=1.6,
                          jump_factor=2.0, zorder=3):
    """
    Plot one branch with markers and connect only nearby consecutive points.

    Parameters
    ----------
    t, r : arrays
        time and radius arrays
    color : str
        matplotlib color
    label : str
        legend label
    s : float
        marker size
    lw : float
        line width
    jump_factor : float
        if consecutive points differ by more than this factor in r,
        the connecting line is broken
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)

    mask = np.isfinite(t) & np.isfinite(r)
    t = t[mask]
    r = r[mask]

    if len(t) == 0:
        return

    # scatter points
    ax.scatter(t, r, s=s, color=color, alpha=0.9, label=label, zorder=zorder)

    # segmented line
    start = 0
    for i in range(len(r) - 1):
        ratio = max(r[i+1] / r[i], r[i] / r[i+1])
        if ratio > jump_factor:
            if i + 1 - start >= 2:
                ax.plot(t[start:i+1], r[start:i+1], color=color, lw=lw, alpha=0.9, zorder=zorder-1)
            start = i + 1

    # last segment
    if len(r) - start >= 2:
        ax.plot(t[start:], r[start:], color=color, lw=lw, alpha=0.9, zorder=zorder-1)


fig, ax = plt.subplots(figsize=(10.8, 6.8))

plot_branch_segmented(ax, time_arr, r_inner,
                      color="black", label="inner branch",
                      s=30, lw=1.4, jump_factor=1.6)

plot_branch_segmented(ax, time_arr, r_middle,
                      color="red", label="intermediate branch",
                      s=28, lw=1.4, jump_factor=1.6)

# plot_branch_segmented(ax, time_arr, r_outer,
#                       color="blue", label="outer branch",
#                       s=20, lw=1.6, jump_factor=1.25)

plot_branch_segmented(ax, time_arr, r_outer,
                      color="blue", label="",
                      s=20, lw=1.6, jump_factor=1.25)


ax.set_yscale("log")
ax.set_xscale("log")

# ax.set_xlim(1.0, 2.0)
# ax.set_ylim(1e-2, 0.7)

# ax.set_title(r"RSIDM: evolution of $r_1$ branches", fontsize=20, pad=12)
ax.set_title(r"cSIDM: evolution of $r_1$", fontsize=20, pad=12)
ax.set_xlabel("time [Gyr]", fontsize=16)
ax.set_ylabel(r"$r_1\ [r_s]$", fontsize=16)

ax.tick_params(axis='both', which='major', labelsize=13, length=6)
ax.tick_params(axis='both', which='minor', length=3)

ax.grid(alpha=0.22, which="both")
ax.legend(fontsize=13, frameon=True, loc="lower right")

plt.tight_layout()
plt.show()



# gravoEvolution_RSIDM.build_interpolated_profiles(
#     rho_mode="loglog",
#     vel_mode="loglog",
#     stop_at_collapse=True,
# )
#
# gravoEvolution_RSIDM.build_scattering_matrix_from_interpolators(
#     sigma_m=1.0,                  # cm^2/g
#     sigma_input_vel_unit="km/s",
#     ngrid=1000,
# )
#
# data = gravoEvolution_RSIDM.scattering_data
#
# # vel_grid is stored in [r_s/Gyr]
# # convert to [km/s]:
# nu_kms = data["vel_grid"] * gravoEvolution_RSIDM.parameters["r_s"] * cfg.kpcGyr_to_kms
#
# print("nu min =", np.nanmin(nu_kms))
# print("nu max =", np.nanmax(nu_kms))
#
# sigma_eff_from_nu_kpcGyr, sigma_eff_from_nu_kms, (nu_grid, sigma_grid) = \
#     build_sigma_m_eff_p5_interpolator(
#         sigma_m0=0.008,
#         m_GeV=0.02,
#         L=0,
#         Gamma=6e-12,
#         vR_kms=85.0,
#         nu_kms_min=0.1,
#         nu_kms_max=3000,
#         N=260
#     )
#
# fig, ax = plt.subplots()
#
# # Use precomputed sigma_grid, not the callable itself
# ax.loglog(nu_grid, sigma_grid, lw=2, label=r"$\sigma_{\rm eff}(\nu)$")
#
# nu_sample = nu_kms.flatten()
# nu_sample = nu_sample[np.isfinite(nu_sample)]
#
# # Put sampled velocities on a horizontal floor just for visualization
# y_floor = np.full_like(nu_sample, np.nanmin(sigma_grid) * 1.2)
#
# ax.scatter(nu_sample, y_floor, s=3, color="black", alpha=0.15, label="halo-sampled velocities")
#
# ax.set_xlabel(r"$\nu$ [km/s]")
# ax.set_ylabel(r"$\sigma_{\rm eff}$ [cm$^2$/g]")
# ax.grid(alpha=0.3, which="both")
# ax.legend()
# plt.show()
#
#
# r = data["r_grid"]
# nu = nu_kms
# sigma = sigma_eff_from_nu_kms(nu)
#
# fig, ax = plt.subplots()
#
# for i in range(0, nu.shape[0], 200):   # few time slices
#     ax.loglog(r, sigma[i])
#
# ax.set_xlabel("r [r_s]")
# ax.set_ylabel("sigma_eff [cm^2/g]")
# ax.grid(True, which="both")
# plt.show()
#
# time_arr, r1_arr, roots_list, dbg = gravoEvolution_SIDM.find_all_r1_roots_from_scattering_matrix(
#     pick="outermost"
# )
#
# fig, ax = plt.subplots(figsize=(7, 5))
#
# for t, roots in zip(time_arr, roots_list):
#     if len(roots) == 0:
#         continue
#     ax.scatter(np.full(len(roots), t), roots, s=10)
#
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("time [Gyr]")
# ax.set_ylabel(r"all $r_1$ roots [$r_s$]")
# ax.grid(alpha=0.3, which="both")
# plt.show()