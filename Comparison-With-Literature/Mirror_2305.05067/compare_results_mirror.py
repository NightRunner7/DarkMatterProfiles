import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- IMPORT FROM FILES
import GravothermalData as gravoF
import RprocedureData as rprocF
import unit_conversion as unic

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
gravothermal_file_1 = 'ρσ_sol_M_9.89_t_175_sigma_5.0_beta_1.1.csv'
gravothermal_file_2 = 'ρσ_sol_M_9.89_t_175_sigma_5.0_beta_1.0.csv'
gravothermal_file_3 = 'ρσ_sol_M_9.89_t_330_sigma_5.0_beta_0.5.csv'
gravothermal_file_4 = 'ρσ_sol_M_9.89_t_330_sigma_5.0_beta_1.05.csv'
gravothermal_file_5 = 'ρσ_sol_M_9.89_t_330_sigma_5.0_beta_1.04.csv'

# our Isothermal simulation results
isothermal_loDen_file = 'Riso_M_9.89000_t_110.355_sigma_5.0_con_15.800.csv'
isothermal_HiDen_file = 'RisoHiDens_M_9.89000_t_110.355_sigma_m_5.0_con_15.800.csv'
# Jiang Isothermal simulation results
isothermalJiang_loDen_file = 'RisoJiang_M_9.89000_t_135.723_sigma_5.0_con_15.800.csv'
isothermalJiang_HiDen_file = 'RisoHiDensJiang_M_9.89000_t_135.723_sigma_m_5.0_con_15.800.csv'

# --- proportion parameter
# proportionOur = 1.32  # can be changed
proportionOur = 2.0  # can be changed
proportionJiang = 2.0  # must be

# ##################################### SET DATA TO CLASS ############################################################ #
# ------------------------------- GRAVOTHERMAL ------------------------------- #
# --- file with beta = 1.1
gravoEvolution_1 = gravoF.create_gravothermalData_from_file(gravothermal_file_1, DATA_GRAVOTHERMAL, beta=1.1)
gravoEvolution_1.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)

# --- file with beta = 1.0
gravoEvolution_2 = gravoF.create_gravothermalData_from_file(gravothermal_file_2, DATA_GRAVOTHERMAL, beta=1.0)
gravoEvolution_2.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)

# --- file with beta = 0.5
gravoEvolution_3 = gravoF.create_gravothermalData_from_file(gravothermal_file_3, DATA_GRAVOTHERMAL, beta=0.50)
gravoEvolution_3.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)
gravoEvolution_3.put_extra_parameters(10**9.89, 5.0, 200.0)

# --- file with beta = 1.05
gravoEvolution_4 = gravoF.create_gravothermalData_from_file(gravothermal_file_4, DATA_GRAVOTHERMAL, beta=1.04)
gravoEvolution_4.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)

# --- file with beta = 1.04
gravoEvolution_5 = gravoF.create_gravothermalData_from_file(gravothermal_file_5, DATA_GRAVOTHERMAL, beta=1.05)
gravoEvolution_5.put_the_name("NFW")  # put the name to differentiate data in class (what initial profile is)


_, time_collapse_3, _ = gravoEvolution_3.find_collapse(elements=2, fixed_limit=10**15)
print(f"beta: {gravoEvolution_3.beta}, time collapse: {time_collapse_3} [Gyr]")


# ------------------------------- ISOTHERMAL (our results) ------------------------------- #
isoEvolution_our = dict()
# --- low dense
isoEvolution_our['LoDen'] = rprocF.create_RprocedureData_from_file(isothermal_loDen_file, DATA_ISOTHERMAL_NFW_LoDen_DIR)
isoEvolution_our['LoDen'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# --- high dense
isoEvolution_our['HiDen'] = rprocF.create_RprocedureData_from_file(isothermal_HiDen_file, DATA_ISOTHERMAL_NFW_HiDen_DIR)
isoEvolution_our['HiDen'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# --- mirror dense
isoEvolution_our['Mirror'] = rprocF.create_RprocedureDataMirror_from_file(isothermal_HiDen_file,
                                                                          DATA_ISOTHERMAL_NFW_HiDen_DIR,
                                                                          proportion=proportionOur)
isoEvolution_our['Mirror'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# ------------------------------- ISOTHERMAL (Jiang's results) ------------------------------- #
isoEvolution_Jiang = dict()
# --- low dense
isoEvolution_Jiang['LoDen'] = rprocF.create_RprocedureData_from_file(isothermalJiang_loDen_file,
                                                                     DATA_ISOTHERMAL_NFW_LoDen_DIR)
isoEvolution_Jiang['LoDen'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# --- high dense
isoEvolution_Jiang['HiDen'] = rprocF.create_RprocedureData_from_file(isothermalJiang_HiDen_file,
                                                                     DATA_ISOTHERMAL_NFW_HiDen_DIR)
isoEvolution_Jiang['HiDen'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# --- mirror dense
isoEvolution_Jiang['Mirror'] = rprocF.create_RprocedureDataMirror_from_file(isothermalJiang_HiDen_file,
                                                                            DATA_ISOTHERMAL_NFW_HiDen_DIR,
                                                                            proportion=proportionJiang)
isoEvolution_Jiang['Mirror'].put_the_name("IsothermalAndNFW")  # put the name to differentiate data in class

# ##################################### TAKE SIMULATION PARAMETERS ################################################### #
# --- from our simulation
ourIsoParameters = isoEvolution_our['LoDen'].return_parameters()
rho_s = ourIsoParameters["rho_s"]
r_s = ourIsoParameters["r_s"]
sigma_m = ourIsoParameters["sigma_m"]
ourTimeSteps = isoEvolution_our['LoDen'].time_steps

# --- from our GRAVO
rho_s_GRAVO = gravoEvolution_3.parameters["rho_s"]
r_s_GRAVO = gravoEvolution_3.parameters["r_s"]
c_const_GRAVO = gravoEvolution_3.parameters["const_c"]

print("rho_s_GRAVO:", rho_s_GRAVO)
print("r_s_GRAVO:", r_s_GRAVO)
print("c_const_GRAVO:", c_const_GRAVO)

# --- from Jiang simulation
jiangIsoParameters = isoEvolution_Jiang['LoDen'].return_parameters()
jiangTimeSteps = isoEvolution_Jiang['LoDen'].time_steps

# --- from gravothermal simulation
_, time_collapse_1, _ = gravoEvolution_1.find_collapse(elements=2, fixed_limit=10**23)
print(f"beta: {gravoEvolution_1.beta}, time collapse: {time_collapse_1} [Gyr]")

_, time_collapse_2, _ = gravoEvolution_2.find_collapse(elements=2, fixed_limit=10**23)
print(f"beta: {gravoEvolution_2.beta}, time collapse: {time_collapse_2} [Gyr]")

_, time_collapse_3, _ = gravoEvolution_3.find_collapse(elements=2, fixed_limit=10**15)
print(f"beta: {gravoEvolution_3.beta}, time collapse: {time_collapse_3} [Gyr]")


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

# ------------------ RESCALING BETA PARAMETER ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$\hat{\sigma_{m}} \, \hat{t} = \hat{\sigma_{m}} \, \left( t \cdot \sqrt{4 \pi G \rho_s}\right)$',
           fontsize=18)
plt.title(r'Rescale $\beta$ parameter', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
# The evolution of core density for gravothermal simulation, beta = 1.1
gravo_time_l, gravo_rho_l = gravoEvolution_1.return_rho_core_evolution(elements=2)
ax.plot(unic.time_units_paper(gravo_time_l, rho_s_GRAVO, r_s_GRAVO, sigma_m), gravo_rho_l,
        label=f'Gravothermal $\\beta$: {gravoEvolution_1.beta}')
plt.axvline(x=unic.time_units_paper(time_collapse_1, rho_s, r_s, sigma_m), linestyle=':', color='grey',
            label=f'Collapse time: {"{:.3f}".format(unic.time_units_paper(time_collapse_1, rho_s_GRAVO, r_s_GRAVO, sigma_m))}, '
            f'refers to  {"{:.3f}".format(time_collapse_1)} [Gyr].')
# The evolution of core density for gravothermal simulation, beta = 1.0
gravo_time_l, gravo_rho_l = gravoEvolution_2.return_rho_core_evolution(elements=2)
ax.plot(unic.time_units_paper(gravo_time_l, rho_s_GRAVO, r_s_GRAVO, sigma_m), gravo_rho_l,
        label=f'Gravothermal $\\beta$: {gravoEvolution_2.beta}')
plt.axvline(x=unic.time_units_paper(time_collapse_2, rho_s_GRAVO, r_s_GRAVO, sigma_m), linestyle=':', color='black',
            label=f'Collapse time: {"{:.3f}".format(unic.time_units_paper(time_collapse_2, rho_s_GRAVO, r_s_GRAVO, sigma_m))}, '
            f'refers to {"{:.3f}".format(time_collapse_2)} [Gyr].')

# The evolution of core density for gravothermal simulation, beta = 1.05
gravo_time_l, gravo_rho_l = gravoEvolution_4.return_rho_core_evolution(elements=2)
ax.plot(unic.time_units_paper(gravo_time_l, rho_s_GRAVO, r_s_GRAVO, sigma_m), gravo_rho_l,
        label=f'Gravothermal $\\beta$: {gravoEvolution_4.beta}')
# The evolution of core density for gravothermal simulation, beta = 1.04
gravo_time_l, gravo_rho_l = gravoEvolution_5.return_rho_core_evolution(elements=2)
ax.plot(unic.time_units_paper(gravo_time_l, rho_s_GRAVO, r_s_GRAVO, sigma_m), gravo_rho_l,
        label=f'Gravothermal $\\beta$: {gravoEvolution_5.beta}')


# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'Scaling-beta-parameter' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: ISOTHERMAL VS GRAVOTHERMAL: JIANG ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$\hat{\sigma_{m}} \, \hat{t} = \hat{\sigma_{m}} \, \left( t \cdot \sqrt{4 \pi G \rho_s}\right)$',
           fontsize=18)
plt.title(r'Jiang: Gravothermal vs Isothermal', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
# # The evolution of core density for gravothermal simulation, beta = 0.75
# gravo_time_l, gravo_rho_l = gravoEvolution_1.return_rho_core_evolution(elements=2)
# ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
#         label=f'Gravothermal $\\beta$: {gravoEvolution_1.beta}')
# # The evolution of core density for gravothermal simulation, beta = 1.1
# gravo_time_l, gravo_rho_l = gravoEvolution_2.return_rho_core_evolution(elements=2)
# ax.plot(uni.time_tilda(gravo_time_l, rho_s, r_s, sigma_m), gravo_rho_l,
#         label=f'Gravothermal $\\beta$: {gravoEvolution_2.beta} (their $\\beta: 0.75$)')
# plt.axvline(x=uni.time_tilda(time_collapse_2, rho_s, r_s, sigma_m), linestyle=':', color='grey',
#             label=f'Collapse time: {"{:.3f}".format(uni.time_tilda(time_collapse_2, rho_s, r_s, sigma_m))}, refers to '
#             f'{"{:.3f}".format(time_collapse_2)} [Gyr].')

# --- GRAVO
gravo_time_l, gravo_rho_l = gravoEvolution_3.return_rho_core_evolution(elements=2)

gravo_time_l = gravo_time_l * 0.69/1.05

# ax.plot(unic.time_units_paper(gravo_time_l, rho_s_GRAVO, r_s_GRAVO, sigma_m), gravo_rho_l,
#         label=f'Gravothermal $\\beta$: {gravoEvolution_3.beta}')
ax.plot(unic.time_units_paper(gravo_time_l, rho_s_GRAVO, r_s_GRAVO, sigma_m), gravo_rho_l,
        label=f'Gravothermal $\\beta$: {0.4}')


# --- The evolution of central density for R-procedure simulation: JIANG
iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_Jiang['LoDen'].return_central_rho()
ax.plot(unic.time_units_paper(iso_lowDen_central_time_l, rho_s, r_s, sigma_m), iso_lowDen_central_rho_l,
        label='Isothermal (low dense): Jiang')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_Jiang['HiDen'].return_central_rho()
ax.plot(unic.time_units_paper(iso_HigDen_central_time_l, rho_s, r_s, sigma_m), iso_HigDen_central_rho_l,
        label='Isothermal (high dense): Jiang')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_Jiang['Mirror'].return_central_rho()
ax.plot(unic.time_units_paper(iso_HigDen_central_time_l, rho_s, r_s, sigma_m), iso_HigDen_central_rho_l,
        "--", label=f'Isothermal (mirror, $\\chi = {proportionJiang}$): Jiang')
# --- Jiang tmerge
ax.scatter(unic.time_units_paper(jiangIsoParameters["time-end"], rho_s, r_s, sigma_m),
           isoEvolution_Jiang['LoDen'].central_data['rho'][jiangTimeSteps - 1],
           marker='*', s=90.0, zorder=3,
           label=f'tmerge: {"{:.3f}".format(unic.time_units_paper(jiangIsoParameters["time-end"], rho_s, r_s, sigma_m))}, refers to '
                 f'{"{:.3f}".format(jiangIsoParameters["time-end"])} [Gyr].',
           facecolor='red', edgecolor='black', linewidth=1.0, rasterized=True)
# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'Jiang-Gravothermal-vs-Isothermal' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: ISOTHERMAL VS GRAVOTHERMAL: US ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$\hat{\sigma_{m}} \, \hat{t} = \hat{\sigma_{m}} \, \left( t \cdot \sqrt{4 \pi G \rho_s}\right)$',
           fontsize=18)
plt.title(r'Us: Gravothermal vs Isothermal', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
gravo_time_l, gravo_rho_l = gravoEvolution_3.return_rho_core_evolution(elements=2)

# gravo_time_l = gravo_time_l * 0.69/0.40

ax.plot(unic.time_units_paper(gravo_time_l, rho_s_GRAVO, r_s_GRAVO, sigma_m), gravo_rho_l,
        label=f'Gravothermal $\\beta$: {gravoEvolution_3.beta}')


# --- The evolution of central density for R-procedure simulation: OUR
iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_our['LoDen'].return_central_rho()
ax.plot(unic.time_units_paper(iso_lowDen_central_time_l, rho_s, r_s, sigma_m), iso_lowDen_central_rho_l,
        label='Isothermal (low dense): our')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_our['HiDen'].return_central_rho()
ax.plot(unic.time_units_paper(iso_HigDen_central_time_l, rho_s, r_s, sigma_m), iso_HigDen_central_rho_l,
        label='Isothermal (high dense): our')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_our['Mirror'].return_central_rho()
ax.plot(unic.time_units_paper(iso_HigDen_central_time_l, rho_s, r_s, sigma_m), iso_HigDen_central_rho_l,
        "--", label=f'Isothermal (mirror, $\\chi = {proportionOur}$): our')
# --- OUR tmerge
ax.scatter(unic.time_units_paper(ourIsoParameters["time-end"], rho_s, r_s, sigma_m),
           isoEvolution_our['LoDen'].central_data['rho'][ourTimeSteps - 1],
           marker='o', s=50.0, zorder=3,
           label=f'tmerge: {"{:.3f}".format(unic.time_units_paper(ourIsoParameters["time-end"], rho_s, r_s, sigma_m))}, refers to '
                 f'{"{:.3f}".format(ourIsoParameters["time-end"])} [Gyr].',
           facecolor='grey', edgecolor='black', linewidth=1.0, rasterized=True)
# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'Us-Gravothermal-vs-Isothermal' + '.png', dpi=300)
else:
    plt.show()

# ------------------ PLOT: ISOTHERMAL: JIANG VS US ------------------------------------- #
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# log scale
ax.set_xscale('log')
ax.set_yscale('log')
# describe plot
plt.ylim(10 ** 0, 2 * 10 ** 3)
plt.ylabel(r'$\hat{\rho} = \rho/\rho_{s}$', fontsize=18)
plt.xlabel(r'$\hat{\sigma_{m}} \, \hat{t} = \hat{\sigma_{m}} \, \left( t \cdot \sqrt{4 \pi G \rho_s}\right)$',
           fontsize=18)
plt.title(r'Jiang: Gravothermal vs Isothermal', fontsize=18)
ax.tick_params(labelsize=14)

# Apply the log scale and grid settings
apply_log_scale_grid(ax)

# --- DATA
# --- The evolution of central density for R-procedure simulation: JIANG
iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_Jiang['LoDen'].return_central_rho()
ax.plot(unic.time_units_paper(iso_lowDen_central_time_l, rho_s, r_s, sigma_m), iso_lowDen_central_rho_l,
        label='Isothermal (low dense): Jiang')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_Jiang['Mirror'].return_central_rho()
ax.plot(unic.time_units_paper(iso_HigDen_central_time_l, rho_s, r_s, sigma_m), iso_HigDen_central_rho_l,
        "--", label=f'Isothermal (mirror, $\\chi = {proportionJiang}$): Jiang')
# --- Jiang tmerge
ax.scatter(unic.time_units_paper(jiangIsoParameters["time-end"], rho_s, r_s, sigma_m),
           isoEvolution_Jiang['LoDen'].central_data['rho'][jiangTimeSteps - 1],
           marker='*', s=90.0, zorder=3,
           label=f'tmerge: {"{:.3f}".format(unic.time_units_paper(jiangIsoParameters["time-end"], rho_s, r_s, sigma_m))}, refers to '
                 f'{"{:.3f}".format(jiangIsoParameters["time-end"])} [Gyr].',
           facecolor='red', edgecolor='black', linewidth=1.0, rasterized=True)

# --- The evolution of central density for R-procedure simulation: OUR
iso_lowDen_central_time_l, iso_lowDen_central_rho_l = isoEvolution_our['LoDen'].return_central_rho()
ax.plot(unic.time_units_paper(iso_lowDen_central_time_l, rho_s, r_s, sigma_m), iso_lowDen_central_rho_l,
        label='Isothermal (low dense): our')
iso_HigDen_central_time_l, iso_HigDen_central_rho_l = isoEvolution_our['Mirror'].return_central_rho()
ax.plot(unic.time_units_paper(iso_HigDen_central_time_l, rho_s, r_s, sigma_m), iso_HigDen_central_rho_l,
        "--", label=f'Isothermal (mirror, $\\chi = {proportionOur}$): our')
# --- OUR tmerge
ax.scatter(unic.time_units_paper(ourIsoParameters["time-end"], rho_s, r_s, sigma_m),
           isoEvolution_our['LoDen'].central_data['rho'][ourTimeSteps - 1],
           marker='o', s=50.0, zorder=3,
           label=f'tmerge: {"{:.3f}".format(unic.time_units_paper(ourIsoParameters["time-end"], rho_s, r_s, sigma_m))}, refers to '
                 f'{"{:.3f}".format(ourIsoParameters["time-end"])} [Gyr].',
           facecolor='grey', edgecolor='black', linewidth=1.0, rasterized=True)

# --- LEGEND AND SAVE
ax.legend(fontsize=14)
if plot_save is True:
    plt.savefig('./' + output_plot_dir + '/' + 'Isothermal-Jiang-vs-Us' + '.png', dpi=300)
else:
    plt.show()
