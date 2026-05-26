import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import units as uni

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
gravothermal_SIDM_file_1 = "SIDM_sol_M_8.8_t_90_sigma_15.2_beta_0.75.csv"

# gravothermal_SIDM_file_2 = 'SIDM_sol_M_9._t_12_sigma_176_beta_0.75.csv'
gravothermal_SIDM_file_2 = 'SIDM_sol_M_9._t_50_sigma_119.0_beta_0.75.csv'


gravothermal_SIDM_file_3 = "SIDM_sol_M_9.2_t_10_sigma_845_beta_0.75.csv"
gravothermal_SIDM_file_4 = "SIDM_sol_M_10._t_50_sigma_2500_beta_0.75.csv"
gravothermal_SIDM_file_5 = "SIDM_sol_M_11._t_50_sigma_63.86_beta_0.75.csv"


# gravothermal_SIDM_file_2 = 'SIDM_sol_M_10._t_50_sigma_70.0_beta_0.75.csv'
# gravothermal_SIDM_file_3 = 'SIDM_sol_M_11._t_50_sigma_2264.0_beta_0.75.csv'
# gravothermal_SIDM_file_4 = 'ρσ_sol_M_10._t_20_sigma_beta_0.75.csv'
# gravothermal_SIDM_file_5 = 'ρσ_sol_M_11._t_20_sigma_beta_0.75.csv'

gravothermal_RSIMD_file_1 = 'RSIDM_sol_M_8.8_t_100_beta_0.75.csv'
gravothermal_RSIMD_file_2 = 'RSIDM_sol_M_9._t_20_beta_0.75.csv'
gravothermal_RSIMD_file_3 = 'RSIDM_sol_M_9.2_t_20_beta_0.75.csv'
gravothermal_RSIMD_file_4 = 'RSIDM_sol_M_10._t_20_beta_0.75.csv'
gravothermal_RSIMD_file_5 = 'RSIDM_sol_M_11._t_20_beta_0.75.csv'


# ##################################### SET DATA TO CLASS ############################################################ #
# ------------------------------- GRAVOTHERMAL ------------------------------- #

# --- SIDM
gravoEvolution_SIDM_1 = gravoF.create_gravothermalData_from_file(gravothermal_SIDM_file_1, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_SIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_1.put_extra_parameters(10 ** 8.8, 119.0)

gravoEvolution_SIDM_2 = gravoF.create_gravothermalData_from_file(gravothermal_SIDM_file_2, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_SIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_2.put_extra_parameters(10 ** 9.0, 70.0)

gravoEvolution_SIDM_3 = gravoF.create_gravothermalData_from_file(gravothermal_SIDM_file_3, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_SIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_3.put_extra_parameters(10 ** 9.2, 70.0)

gravoEvolution_SIDM_4 = gravoF.create_gravothermalData_from_file(gravothermal_SIDM_file_4, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_SIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_4.put_extra_parameters(10 ** 10.0, 70.0)

gravoEvolution_SIDM_5 = gravoF.create_gravothermalData_from_file(gravothermal_SIDM_file_5, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_SIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_SIDM_5.put_extra_parameters(10 ** 11.0, 70.0)

# --- RSIDM
gravoEvolution_RSIDM_1 = gravoF.create_gravothermalData_from_file(gravothermal_RSIMD_file_1, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_RSIDM_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_1.put_extra_parameters(10**8.8, 119.0)

gravoEvolution_RSIDM_2 = gravoF.create_gravothermalData_from_file(gravothermal_RSIMD_file_2, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_RSIDM_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_2.put_extra_parameters(10**9.0, 119.0)

gravoEvolution_RSIDM_3 = gravoF.create_gravothermalData_from_file(gravothermal_RSIMD_file_3, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_RSIDM_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_3.put_extra_parameters(10**9.2, 119.0)

gravoEvolution_RSIDM_4 = gravoF.create_gravothermalData_from_file(gravothermal_RSIMD_file_4, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_RSIDM_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_4.put_extra_parameters(10**10.0, 119.0)

gravoEvolution_RSIDM_5 = gravoF.create_gravothermalData_from_file(gravothermal_RSIMD_file_5, DATA_GRAVOTHERMAL, beta=0.75)
gravoEvolution_RSIDM_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_RSIDM_5.put_extra_parameters(10**11.0, 119.0)


_, time_collapse_1, _ = gravoEvolution_SIDM_2.find_collapse(elements=2, fixed_limit=10 ** 15)
print(f"beta: {gravoEvolution_SIDM_1.beta}, time collapse: {time_collapse_1} [Gyr]")

# --- from our GRAVO
rho_s_GRAVO = gravoEvolution_SIDM_1.parameters["rho_s"]
r_s_GRAVO = gravoEvolution_SIDM_1.parameters["r_s"]
c_const_GRAVO = gravoEvolution_SIDM_1.parameters["const_c"]

print("rho_s_GRAVO:", rho_s_GRAVO)
print("r_s_GRAVO:", r_s_GRAVO)
print("c_const_GRAVO:", c_const_GRAVO)



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


# ------------------ CORE DENSITY EVOLUTION ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 3 * 10 ** 3)
plt.xlim(10 ** -1, 2 * 10 ** 1)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$t$ [Gyr]',
           fontsize=18)
plt.title(r'Evolution of: $\rho_c$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
# gravo_time_l, gravo_rho_l = gravoEvolution_SIDM_1.return_rho_core_evolution(elements=2)
# ax.plot(gravo_time_l, gravo_rho_l, linestyle="--", color="#0072B2",
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{8.8}}\,M_\odot$'
#         )

gravo_time_l, gravo_rho_l = gravoEvolution_SIDM_2.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, linestyle="--", color="#D55E00",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$'
        )

gravo_time_l, gravo_rho_l = gravoEvolution_SIDM_3.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, linestyle="--", color="#009E73",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.2}}\,M_\odot$'
        )

gravo_time_l, gravo_rho_l = gravoEvolution_SIDM_4.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, linestyle="--", color="#CC79A7",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{10.0}}\,M_\odot$'
        )

# The evolution of core density for gravothermal simulation, beta = 1.1
gravo_time_l, gravo_rho_l = gravoEvolution_SIDM_5.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, linestyle="--", color="#E69F00",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{11.0}}\,M_\odot$'
        )


# --- RESONANT
# gravo_time_l, gravo_rho_l = gravoEvolution_RSIDM_1.return_rho_core_evolution(elements=2)
# ax.plot(gravo_time_l, gravo_rho_l, color="#0072B2",
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{8.8}}\,M_\odot$'
#         )

gravo_time_l, gravo_rho_l = gravoEvolution_RSIDM_2.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, color="#D55E00",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$'
        )

gravo_time_l, gravo_rho_l = gravoEvolution_RSIDM_3.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, color="#009E73",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.2}}\,M_\odot$'
        )

gravo_time_l, gravo_rho_l = gravoEvolution_RSIDM_4.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, color="#CC79A7",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{10.0}}\,M_\odot$'
        )

gravo_time_l, gravo_rho_l = gravoEvolution_RSIDM_5.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, color="#E69F00",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{11.0}}\,M_\odot$'
        )


# --- LEGEND AND SAVE
ax.legend(fontsize=10, loc='upper left')
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'Scaling-beta-parameter' + '.png', dpi=300)
else:
    plt.show()


# ------------------ CORE DENSITY EVOLUTION ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 3 * 10 ** 3)
plt.xlim(10 ** -1, 2 * 10 ** 2)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$t$ [Gyr]',
           fontsize=18)
plt.title(r'Evolution of: $\rho_c$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
gravo_time_l, gravo_rho_l = gravoEvolution_SIDM_1.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, linestyle="--", color="#0072B2",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{8.8}}\,M_\odot$'
        )

# --- RESONANT
gravo_time_l, gravo_rho_l = gravoEvolution_RSIDM_1.return_rho_core_evolution(elements=2)
ax.plot(gravo_time_l, gravo_rho_l, color="#0072B2",
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{8.8}}\,M_\odot$'
        )

# --- LEGEND AND SAVE
ax.legend(fontsize=10, loc='upper left')
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'Scaling-beta-parameter' + '.png', dpi=300)
else:
    plt.show()



# ------------------ CORE SIZE EVOLUTION ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** -3, 2 * 10 ** 0)
plt.xlim(10 ** -1, 0.6 * 10 ** 2)
plt.ylabel(r'$\hat{r} = r/r_{s}$', fontsize=18)
plt.xlabel(r'$t$ [Gyr]',
           fontsize=18)
plt.title(r'Evolution of: $r_c$, definition: $\rho(r_c)=\rho_c/2$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
# The evolution of core density for gravothermal simulation, beta = 1.1
gravo_time_l, gravo_rc_l = gravoEvolution_SIDM_1.return_rc_evolution(elements=2)
# gravo_time_l = gravo_time_l * 1.24
ax.plot(gravo_time_l, gravo_rc_l,
        label=f'Gravothermal $\\beta$: {gravoEvolution_SIDM_1.beta}, SIDM')
# The evolution of core density for gravothermal simulation, beta = 1.1
gravo_time_l, gravo_rc_l = gravoEvolution_RSIDM_1.return_rc_evolution(elements=2)
gravo_time_l = gravo_time_l / 1.35
ax.plot(gravo_time_l, gravo_rc_l,
        label=f'Gravothermal $\\beta$: {gravoEvolution_SIDM_1.beta}, RSIDM')


# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'Scaling-beta-parameter' + '.png', dpi=300)
else:
    plt.show()


# ------------------ VELOCITY DISPERSION AROUND MINIMAL RHO ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(1* 10 ** -1, 2.7 * 10 ** -1)
plt.xlim(10 ** -1, 1 * 10 ** 2)
plt.xlabel(r'$\hat{r} = r/r_{s}$', fontsize=18)
plt.ylabel(r'$\hat{\nu} = \nu(r)/\nu_{0}$ ',
           fontsize=18)
# plt.title(r'Evolution of: $r_c$, definition: $\rho(r_c)=\rho_c/2$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
gravo_r_l, gravo_nu_l, time_step = gravoEvolution_RSIDM_1.return_veldis_hat_profile()
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=time_step+200)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step+200}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=time_step+100)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step+100}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=time_step-100)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step-100}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=time_step-200)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step-200}'
        )


plt.title(rf'Velocity dispersion: $\nu(r)$, around min $\rho(t)$, time step: {time_step}', fontsize=18)

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'velocity_dispersion' + '.png', dpi=300)
else:
    plt.show()


# ------------------ VELOCITY DISPERSION AROUND MINIMAL RHO ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(1* 10 ** -1, 2.7 * 10 ** -1)
plt.xlim(10 ** -1, 1 * 10 ** 2)
plt.xlabel(r'$\hat{r} = r/r_{s}$', fontsize=18)
plt.ylabel(r'$\hat{\nu} = \nu(r)/\nu_{0}$ ',
           fontsize=18)
# plt.title(r'Evolution of: $r_c$, definition: $\rho(r_c)=\rho_c/2$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
gravo_r_l, gravo_nu_l, time_step = gravoEvolution_SIDM_1.return_veldis_hat_profile()
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_SIDM_1.return_veldis_hat_profile(time_argument=time_step+200)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step+200}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_SIDM_1.return_veldis_hat_profile(time_argument=time_step+100)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step+100}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_SIDM_1.return_veldis_hat_profile(time_argument=time_step-100)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step-100}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_SIDM_1.return_veldis_hat_profile(time_argument=time_step-200)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step-200}'
        )


plt.title(rf'Velocity dispersion: $\nu(r)$, around min $\rho(t)$, time step: {time_step}', fontsize=18)

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'velocity_dispersion' + '.png', dpi=300)
else:
    plt.show()


# ------------------ VELOCITY DISPERSION AROUND MINIMAL RHO ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(1* 10 ** -1, 2.7 * 10 ** -1)
plt.xlim(10 ** -1, 1 * 10 ** 2)
plt.xlabel(r'$\hat{r} = r/r_{s}$', fontsize=18)
plt.ylabel(r'$\hat{\nu} = \nu(r)/\nu_{0}$ ',
           fontsize=18)
# plt.title(r'Evolution of: $r_c$, definition: $\rho(r_c)=\rho_c/2$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
# gravo_r_l, gravo_nu_l, time_step_RSIDM = gravoEvolution_RSIDM_1.return_veldis_hat_profile()
# ax.plot(gravo_r_l, gravo_nu_l,
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step}'
#         )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=1)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{1}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=200)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{200}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=400)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_RSIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{400}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=800)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_RSIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{800}'
        )

gravo_r_l, gravo_nu_l, _ = gravoEvolution_RSIDM_1.return_veldis_hat_profile(time_argument=1000)
ax.plot(gravo_r_l, gravo_nu_l,
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{1000}'
        )


# gravo_r_l, gravo_nu_l, time_step_SIDM = gravoEvolution_SIDM_1.return_veldis_hat_profile()
# ax.plot(gravo_r_l, gravo_nu_l,
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step}'
#         )

plt.title(rf'Velocity dispersion: $\nu(r)$, around min $\rho(t)$, time step: {time_step}', fontsize=18)

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'velocity_dispersion' + '.png', dpi=300)
else:
    plt.show()


# ------------------ VELOCITY DISPERSION AROUND MINIMAL RHO ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
# plt.ylim(1* 10 ** -1, 2.7 * 10 ** -1)
# plt.xlim(10 ** -1, 1 * 10 ** 2)
plt.xlabel(r'$\hat{r} = r/r_{s}$', fontsize=18)
plt.ylabel(r'$\hat{\rho} = \rho(r)/\rho_{s}$ ',
           fontsize=18)
# plt.title(r'Evolution of: $r_c$, definition: $\rho(r_c)=\rho_c/2$', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
# data = gravoEvolution_RSIDM_1.(time_argument=1)
# ax.plot(data["r"], data["rho"],
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{1}'
#         )

data = gravoEvolution_RSIDM_1.return_data_at_fixed_time(time_argument=200)
ax.plot(data["r"], data["rho"],
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{200}'
        )

data = gravoEvolution_RSIDM_1.return_data_at_fixed_time(time_argument=400)
ax.plot(data["r"], data["rho"],
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{400}'
        )

data = gravoEvolution_RSIDM_1.return_data_at_fixed_time(time_argument=600)
ax.plot(data["r"], data["rho"],
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{600}'
        )

data = gravoEvolution_RSIDM_1.return_data_at_fixed_time(time_argument=1000)
ax.plot(data["r"], data["rho"],
        label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{1000}'
        )



# data = gravoEvolution_SIDM_1.return_data_at_fixed_time(time_argument=1)
# ax.plot(data["r"], data["rho"],
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{1}'
#         )
#
# data = gravoEvolution_SIDM_1.return_data_at_fixed_time(time_argument=1)
# ax.plot(data["r"], data["rho"],
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{1}'
#         )
# data = gravoEvolution_SIDM_1.return_data_at_fixed_time(time_argument=1)
# ax.plot(data["r"], data["rho"],
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, RSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{1}'
#         )


# data = gravoEvolution_SIDM_1.return_data_at_fixed_time(time_step_SIDM)
# ax.plot(data["r"], data["rho"],
#         label=rf'$\beta = {gravoEvolution_SIDM_1.beta}$, cSIDM, $M = 10^{{9.0}}\,M_\odot$, time step:{time_step_SIDM}'
#         )

# plt.title(rf'Velocity dispersion: $\nu(r)$, around min $\rho(t)$, time step: {time_step}', fontsize=18)

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'velocity_dispersion' + '.png', dpi=300)
else:
    plt.show()
