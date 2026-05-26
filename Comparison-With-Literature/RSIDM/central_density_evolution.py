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
DATA_cSIDM = os.path.join(OUR_DIR, 'Data_cSIDM')
DATA_rSIDM = os.path.join(OUR_DIR, 'Data_rSIDM')
DATA_rSIDM_700 = os.path.join(OUR_DIR, 'Data_rSIDM_Nr700')

# # isothermal
# DATA_ISOTHERMAL_NFW_LoDen_DIR = os.path.join(OUR_DIR, 'Data_Extraction')
# DATA_ISOTHERMAL_NFW_HiDen_DIR = os.path.join(OUR_DIR, 'Data_Extraction')

# --- select files: cSIDM
SIDM_file_1  = "CSIDM_M8.4_sigma0.0106_beta0.75.csv"
SIDM_file_2  = "CSIDM_M8.6_sigma0.421_beta0.75.csv"
SIDM_file_3  = "CSIDM_M8.8_sigma15.2_beta0.75.csv"
SIDM_file_4  = "CSIDM_M9.0_sigma119.0_beta0.75.csv"
# SIDM_file_4  = "CSIDM_M9.0_sigma176_beta0.75.csv"
SIDM_file_5  = "CSIDM_M9.2_sigma845_beta0.75.csv"
SIDM_file_6  = "CSIDM_M9.4_sigma2095_beta0.75.csv"
SIDM_file_7  = "CSIDM_M9.6_sigma3148_beta0.75.csv"
SIDM_file_8  = "CSIDM_M9.8_sigma3345_beta0.75.csv"
SIDM_file_9  = "CSIDM_M10.0_sigma2500_beta0.75.csv"
SIDM_file_10 = "CSIDM_M11.0_sigma63.86_beta0.75.csv"

# --- select files: rSIDM
RSIDM_file_1  = "RSIDM_M8.4_beta0.75_Nradi400.csv"
RSIDM_file_2  = "RSIDM_M8.6_beta0.75_Nradi400.csv"
RSIDM_file_3  = "RSIDM_M8.8_beta0.75_Nradi400.csv"
RSIDM_file_4  = "RSIDM_M9.0_beta0.75_Nradi400.csv"
RSIDM_file_5  = "RSIDM_M9.2_beta0.75_Nradi400.csv"
RSIDM_file_6  = "RSIDM_M9.4_beta0.75_Nradi400.csv"
RSIDM_file_7  = "RSIDM_M9.6_beta0.75_Nradi400.csv"
RSIDM_file_8  = "RSIDM_M9.8_beta0.75_Nradi400.csv"
RSIDM_file_9  = "RSIDM_M10.0_beta0.75_Nradi400.csv"
RSIDM_file_10 = "RSIDM_M11.0_beta0.75_Nradi400.csv"

# --- select file (new benchmark): cSIDM
# new_SIDM_file_1  = "New_benchmark/CSIDM_M8.14_sigma672_beta0.75.csv"
# new_SIDM_file_1  = "New_benchmark/CSIDM_M9.09_sigma202_beta0.75.csv"
new_SIDM_file_1  = "New_benchmark/CSIDM_M9.1_sigma96.5_beta0.75.csv"


# --- select file (new benchmark): rSIDM
new_RSIDMf_ile_1 = "New_benchmark/RSIDM_M8.14_beta0.75_Nradi400.csv"
new_RSIDM_file_1 = "RSIDM_M8.8_beta0.75_Nradi700.csv"

# ##################################### SET DATA TO CLASS ############################################################ #
# ------------------------------- SIDM ------------------------------- #
SIDM_1 = gravoF.create_gravothermalData_from_file(SIDM_file_1, DATA_cSIDM, beta=0.75)
SIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_1.put_extra_parameters(10 ** 8.4, 119.0)

SIDM_2 = gravoF.create_gravothermalData_from_file(SIDM_file_2, DATA_cSIDM, beta=0.75)
SIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_2.put_extra_parameters(10 ** 8.6, 70.0)

SIDM_3 = gravoF.create_gravothermalData_from_file(SIDM_file_3, DATA_cSIDM, beta=0.75)
SIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_3.put_extra_parameters(10 ** 8.8, 70.0)

SIDM_4 = gravoF.create_gravothermalData_from_file(SIDM_file_4, DATA_cSIDM, beta=0.75)
SIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_4.put_extra_parameters(10 ** 9.0, 70.0)

SIDM_5 = gravoF.create_gravothermalData_from_file(SIDM_file_5, DATA_cSIDM, beta=0.75)
SIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_5.put_extra_parameters(10 ** 9.2, 70.0)

SIDM_6 = gravoF.create_gravothermalData_from_file(SIDM_file_6, DATA_cSIDM, beta=0.75)
SIDM_6.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_6.put_extra_parameters(10 ** 9.4, 70.0)

SIDM_7 = gravoF.create_gravothermalData_from_file(SIDM_file_7, DATA_cSIDM, beta=0.75)
SIDM_7.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_7.put_extra_parameters(10 ** 9.6, 70.0)

SIDM_8 = gravoF.create_gravothermalData_from_file(SIDM_file_8, DATA_cSIDM, beta=0.75)
SIDM_8.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_8.put_extra_parameters(10 ** 9.8, 70.0)

SIDM_9 = gravoF.create_gravothermalData_from_file(SIDM_file_9, DATA_cSIDM, beta=0.75)
SIDM_9.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_9.put_extra_parameters(10 ** 10.0, 70.0)

SIDM_10 = gravoF.create_gravothermalData_from_file(SIDM_file_10, DATA_cSIDM, beta=0.75)
SIDM_10.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
SIDM_10.put_extra_parameters(10 ** 11.0, 70.0)

# ------------------------------- SIDM (New benchmark)------------------------------- #
new_SIDM_1 = gravoF.create_gravothermalData_from_file(new_SIDM_file_1, DATA_cSIDM, beta=0.75)
new_SIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
new_SIDM_1.put_extra_parameters(10 ** 8.14, 627.0)
# new_SIDM_1.put_extra_parameters(10 ** 9.09, 202.0)
# new_SIDM_1.put_extra_parameters(10 ** 9.1, 96.5)

# ------------------------------- RSIDM ------------------------------- #
rSIDM_1 = gravoF.create_gravothermalData_from_file(RSIDM_file_1, DATA_rSIDM, beta=0.75)
rSIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_1.put_extra_parameters(10 ** 8.4, 119.0)

rSIDM_2 = gravoF.create_gravothermalData_from_file(RSIDM_file_2, DATA_rSIDM, beta=0.75)
rSIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_2.put_extra_parameters(10 ** 8.6, 70.0)

rSIDM_3 = gravoF.create_gravothermalData_from_file(RSIDM_file_3, DATA_rSIDM, beta=0.75)
rSIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_3.put_extra_parameters(10 ** 8.8, 70.0)

rSIDM_4 = gravoF.create_gravothermalData_from_file(RSIDM_file_4, DATA_rSIDM, beta=0.75)
rSIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_4.put_extra_parameters(10 ** 9.0, 70.0)

rSIDM_5 = gravoF.create_gravothermalData_from_file(RSIDM_file_5, DATA_rSIDM, beta=0.75)
rSIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_5.put_extra_parameters(10 ** 9.2, 70.0)

rSIDM_6 = gravoF.create_gravothermalData_from_file(RSIDM_file_6, DATA_rSIDM, beta=0.75)
rSIDM_6.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_6.put_extra_parameters(10 ** 9.4, 70.0)

rSIDM_7 = gravoF.create_gravothermalData_from_file(RSIDM_file_7, DATA_rSIDM, beta=0.75)
rSIDM_7.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_7.put_extra_parameters(10 ** 9.6, 70.0)

rSIDM_8 = gravoF.create_gravothermalData_from_file(RSIDM_file_8, DATA_rSIDM, beta=0.75)
rSIDM_8.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_8.put_extra_parameters(10 ** 9.8, 70.0)

rSIDM_9 = gravoF.create_gravothermalData_from_file(RSIDM_file_9, DATA_rSIDM, beta=0.75)
rSIDM_9.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_9.put_extra_parameters(10 ** 10.0, 70.0)

rSIDM_10 = gravoF.create_gravothermalData_from_file(RSIDM_file_10, DATA_rSIDM, beta=0.75)
rSIDM_10.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
rSIDM_10.put_extra_parameters(10 ** 11.0, 70.0)

# ------------------------------- SIDM (New benchmark)------------------------------- #
new_rSIDM_1 = gravoF.create_gravothermalData_from_file(new_RSIDM_file_1, DATA_rSIDM_700, beta=0.75)
new_rSIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
new_rSIDM_1.put_extra_parameters(10 ** 8.14, 627.0)


# _, time_collapse_1, _ = gravoEvolution_SIDM_2.find_collapse(elements=2, fixed_limit=10 ** 15)
# print(f"beta: {gravoEvolution_SIDM_1.beta}, time collapse: {time_collapse_1} [Gyr]")
#
# # --- from our GRAVO
# rho_s_GRAVO = gravoEvolution_SIDM_1.parameters["rho_s"]
# r_s_GRAVO = gravoEvolution_SIDM_1.parameters["r_s"]
# c_const_GRAVO = gravoEvolution_SIDM_1.parameters["const_c"]
#
# print("rho_s_GRAVO:", rho_s_GRAVO)
# print("r_s_GRAVO:", r_s_GRAVO)
# print("c_const_GRAVO:", c_const_GRAVO)



# ##################################### PLOTTING COMPARISON ########################################################## #
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


def rescale_time(_model, _sigma_eff, beta=0.75):
    r_s = model.parameters["r_s"]
    rho_s = model.parameters["rho_s"]
    sigma_m_SI = _sigma_eff * 1e-4 * 1e3  # Convert from [cm^2/g] to [m^2/kg]
    sigma_m_SU = sigma_m_SI * cfg.kpc_SI ** -2 * cfg.M_solar_SI  # Convert to [kpc^2 * M_sun^-1]
    G_SU = cfg.const_G_starUnits
    time_c = (150/beta) * (1/(r_s * rho_s) * 1/sigma_m_SU) * (4 * np.pi * G_SU * rho_s) ** (-1/2)
    return time_c

################################################## NEW PLOT ############################################################
# ------------------ DATA COLLECTION ------------------ #
sidm_models = [
    # SIDM_1, SIDM_2, SIDM_3,
    SIDM_4, SIDM_5, SIDM_6, SIDM_7, SIDM_8, SIDM_9,
    # SIDM_4, SIDM_5, SIDM_6, SIDM_7, SIDM_8
    # SIDM_10
]

rsidm_models = [
    # rSIDM_1, rSIDM_2, rSIDM_3,
    rSIDM_4, rSIDM_5, rSIDM_6, rSIDM_7, rSIDM_8, rSIDM_9,
    # rSIDM_4, rSIDM_5, rSIDM_6, rSIDM_7, rSIDM_8
    # rSIDM_10
]

masses_log10 = [
                # 8.4, 8.6, 8.8,
                9.0, 9.2, 9.4, 9.6, 9.8, 10.0,
                # 9.0, 9.2, 9.4, 9.6, 9.8,
                # 11.0
                ]

sigma_eff_arr = [
    # 0.0106, 0.421, 15.2,
    119.0, 845, 2095, 3148, 3345, 2500,
    # 119.0, 845, 2095, 3148, 3345
    # 63.86
]

# WORKS ONLY WHEN cSIDM REACHES UNIVERSALITY
markers = ['o', 's', '^']

# ordered colormap
# cmap = mpl.cm.cividis
# cmap = mpl.cm.viridis
cmap = mpl.cm.plasma

norm = mpl.colors.Normalize(vmin=min(masses_log10), vmax=max(masses_log10))
colors = [cmap(norm(m)) for m in masses_log10]

from matplotlib.lines import Line2D

# ------------------ PLOT ------------------ #
fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.set_xscale('log')
ax.set_yscale('log')

# ax.set_xlim(1e-2, 3e1)
ax.set_xlim(1e-2, 2e1)

# FOR Halo masses below 9.0
# ax.set_xlim(2e-1, 3e0)
# ax.set_xlim(1e0, 2e5)
ax.set_ylim(1e0, 3e3)

ax.set_xlabel(r'$t\,[\mathrm{Gyr}]$', fontsize=18)
# ax.set_xlabel(
#     r'$\hat{t} \equiv t/t_c,\quad '
#     r't_c = \frac{150}{\beta\, r_s \, \rho_s \, (\sigma/m)_{\rm eff} \, \sqrt{4\pi G \rho_s}}$',
#     fontsize=18)
ax.set_ylabel(r'$\hat{\rho}_c \equiv \rho_c/\rho_s$', fontsize=18)
ax.set_title(r'Core-density evolution: cSIDM vs rSIDM', fontsize=18)

ax.tick_params(labelsize=13)
apply_log_scale_grid(ax)

for i, (model, mlog, color, sigma_eff) in enumerate(zip(sidm_models, masses_log10, colors, sigma_eff_arr)):
    t, rho = model.return_rho_core_evolution(elements=2)
    time_c = rescale_time(model, sigma_eff)
    ax.plot(
        t, rho,
        # t / time_c, rho,
        linestyle='--',
        linewidth=2.3,
        # marker=markers[i],
        # markersize=5,
        # markevery=35,
        # markerfacecolor='white',
        # markeredgewidth=1.2,
        color=color,
        alpha=0.8,
        label=rf'$10^{{{mlog:.1f}}}\,M_\odot$'
    )

for model, mlog, color, sigma_eff in zip(rsidm_models, masses_log10, colors, sigma_eff_arr):
    t, rho = model.return_rho_core_evolution(elements=2)
    time_c = rescale_time(model, sigma_eff)
    ax.plot(
        t, rho,
        # t / time_c, rho,
        linestyle='-',
        linewidth=2.3,
        color=color,
        alpha=0.95
    )

# legend 1: halo mass
leg1 = ax.legend(
    title=r'Halo mass',
    fontsize=10,
    title_fontsize=11,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.78),
    frameon=True,
    framealpha=0.9
)
ax.add_artist(leg1)

# legend 2: model type
style_handles = [
    Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='cSIDM'),
    Line2D([0], [0], color='black', lw=2.3, linestyle='-', label='rSIDM'),
]
ax.legend(
    handles=style_handles,
    title='Model',
    fontsize=10,
    title_fontsize=11,
    loc='upper left',
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()
if plot_save:
    plt.savefig(f'./{output_plot_dir}/core_density_evolution_csidm.png', dpi=300, bbox_inches='tight')
else:
    plt.show()

plt.close()

# ------------------ PLOT ------------------ #
sidm_models = [new_SIDM_1]

rsidm_models = [new_rSIDM_1]

masses_log10 = [
    # 8.14
    # 9.09
    9.1
]

sigma_eff_arr = [
    # 672.0
    # 202.0
    96.5
]

# WORKS ONLY WHEN cSIDM REACHES UNIVERSALITY
markers = ['o', 's', '^']

# ordered colormap
# cmap = mpl.cm.cividis
# cmap = mpl.cm.viridis
cmap = mpl.cm.plasma

norm = mpl.colors.Normalize(vmin=min(masses_log10), vmax=max(masses_log10))
colors = [cmap(norm(m)) for m in masses_log10]


fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(1e-2, 2e2)
ax.set_ylim(1e0, 3e3)

ax.set_xlabel(r'$t\,[\mathrm{Gyr}]$', fontsize=18)
# ax.set_xlabel(
#     r'$\hat{t} \equiv t/t_c,\quad '
#     r't_c = \frac{150}{\beta\, r_s \, \rho_s \, (\sigma/m)_{\rm eff} \, \sqrt{4\pi G \rho_s}}$',
#     fontsize=18)
ax.set_ylabel(r'$\hat{\rho}_c \equiv \rho_c/\rho_s$', fontsize=18)
ax.set_title(r'Core-density evolution: cSIDM vs rSIDM', fontsize=18)

ax.tick_params(labelsize=13)
apply_log_scale_grid(ax)

for i, (model, mlog, color, sigma_eff) in enumerate(zip(sidm_models, masses_log10, colors, sigma_eff_arr)):
    t, rho = model.return_rho_core_evolution(elements=2)
    time_c = rescale_time(model, sigma_eff)
    ax.plot(
        t, rho,
        # t / time_c, rho,
        linestyle='--',
        linewidth=2.3,
        # marker=markers[i],
        # markersize=5,
        # markevery=35,
        # markerfacecolor='white',
        # markeredgewidth=1.2,
        color=color,
        alpha=0.8,
        label=rf'$10^{{{mlog:.1f}}}\,M_\odot$'
    )

# for model, mlog, color, sigma_eff in zip(rsidm_models, masses_log10, colors, sigma_eff_arr):
#     t, rho = model.return_rho_core_evolution(elements=2)
#     time_c = rescale_time(model, sigma_eff)
#     ax.plot(
#         t, rho,
#         # t / time_c, rho,
#         linestyle='-',
#         linewidth=2.3,
#         color=color,
#         alpha=0.95
#     )

# legend 1: halo mass
leg1 = ax.legend(
    title=r'Halo mass',
    fontsize=10,
    title_fontsize=11,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.78),
    frameon=True,
    framealpha=0.9
)
ax.add_artist(leg1)

# legend 2: model type
style_handles = [
    Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='cSIDM'),
    Line2D([0], [0], color='black', lw=2.3, linestyle='-', label='rSIDM'),
]
ax.legend(
    handles=style_handles,
    title='Model',
    fontsize=10,
    title_fontsize=11,
    loc='upper left',
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.set_xscale('log')
ax.set_yscale('log')

data_new = new_rSIDM_1.return_data_at_fixed_time(10, time_step_bool=False)
data_old = rSIDM_3.return_data_at_fixed_time(10, time_step_bool=False)

print("OLD ONE")
print(data_old["rho"][0], data_old["rho"][1], data_old["rho"][2], data_old["rho"][3])
print(data_old["r"][0], data_old["r"][1], data_old["r"][2], data_old["r"][3])
print("NEW ONE")
print(data_new["rho"][0], data_new["rho"][1], data_new["rho"][2], data_new["rho"][3])
print(data_new["r"][0], data_new["r"][1], data_new["r"][2], data_new["r"][3])

# --- Create plot
ax.plot(data_new["r"], data_new["rho"], label='RSIDM, precise')
ax.plot(data_old["r"], data_old["rho"], label='RSIDM')

# plt.ylim(-6.2, 2.2)
plt.ylabel(r'$\log\rho(r)$', fontsize=18)
plt.xlabel(r'$\log(r)$', fontsize=18)
# plt.title('Density profile on:' + "%10.2e" % (10 ** time_point) + ' Gyr', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.show()
