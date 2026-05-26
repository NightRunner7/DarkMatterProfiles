"""
This file is intended for little analysis of values / behaviour of parameters, that
was set in paper 2205.02957. We will gonna investigate how changes in time and so on.
In some cases we will check for sure that those parameters behaves well - sometimes
we can expect how values they should have or how look like their multiplications.

For more details see the 16-19 page in paper, where the scientist presents their fitting / model.
In Early time in evolution - beginning of core formation we have:

rho^(r^) = tanh(r^ / r_{early_core}) / (r^ * (1+r^)^2),

    where, scientist use dimensionless notation for radius and density
        rho = rho_0 * rho^ => rho^ = rho / rho_0
        rho: [M_sun / kpc^3] the true value of density.
        rho_0: [M_sun / kpc^3] some constant, which just scaling (more details in paper).
        [rho] = [rho_0].
        r_{early_core}: [dimensionless] some constant depends on calibrated constants, which they use to
                        reproduce gravothermal simulation.
    we use that model for: log(\beta * sigma_m^ * t^) < 1.341

In the Late time in evolution - ending of core formation / starting collapsing the core:

rho^(r^) = rho_core / (1 + (r^ / r_{late_core})^(s) * (1 + r^ / r_out)^(3-s) )

    where,
        rho_core: [dimensionless] some constant depends on calibrated constants, which they use to
                  reproduce gravothermal simulation.
        r_{late_core}, r_out: [dimensionless] - the same as above.
        s: [double] constant, which is equally to 2.19
"""
import math
import matplotlib.pyplot as plt
import numpy as np
# --- import files
import modelGravothermalDensities as mod

def evolutionRadiusCore_Model(_mvir, _sigma_m, _beta, _max_time,
                              _path,
                              _z=0, plot_show=False):
    """
    We want to show / present how changes the parameters: r_{early_core} (Early)
                                                          r_{late_core}, r_out (Late)
    """
    # --- set r_s and rho_s
    r_s = mod.cal_r_s(_mvir, _z)
    rho_s = mod.cal_rho_s(_mvir, _z)
    _power_mass = math.log10(_mvir)
    max_dimless_time = mod.cal_time_tilda(_max_time, rho_s, r_s, _sigma_m)

    # --- create class: model
    GravoModelClass = mod.ModelGravothermal(beta=_beta)

    # --- settings dimensionless time
    log_min_dimless_T = -4.0
    log_max_dimless_T = math.log10(max_dimless_time)
    trans_dimless_T = GravoModelClass.return_time_transition()  # transition to core contraction in [dimensionless]
    log_trans_dimless_T = math.log10(trans_dimless_T)
    # divide time into two regimes
    dimless_early_T = np.logspace(log_min_dimless_T, log_trans_dimless_T)
    dimless_late_T = np.logspace(log_trans_dimless_T, log_max_dimless_T)

    # --- find values of different radius's
    r_early_list = [10 ** GravoModelClass.cof_log_r_core_early(time) for time in dimless_early_T]  # [r_s]
    r_late_list = [10 ** GravoModelClass.cof_log_r_core(time) for time in dimless_late_T]  # [r_s]
    r_out_list = [10 ** GravoModelClass.cof_log_r_out(time) for time in dimless_late_T]  # [r_s]

    # --- draw PLOT
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.plot(dimless_early_T, r_early_list, label='radius core for early time')
    ax.plot(dimless_late_T, r_late_list, label='radius core for late time')
    ax.plot(dimless_late_T, r_out_list, label='radius out for late time')
    plt.axvline(x=trans_dimless_T, color='black',
                label='change formula in model,' + "\n" +
                      f'ocured in: {"{:.2f}".format(trans_dimless_T)} [dimless].')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel(r'$\hat{t} \ \left[\frac{4}{\sqrt{4}} \hat{\sigma}_{m} \hat{\nu} \hat{\rho} \right]$', fontsize=18)
    plt.ylabel(r"radius's [r_s]", fontsize=18)
    plt.title(f'Evolution of radius in model. Mass=10^{_power_mass}, sigma={_sigma_m}, time={_max_time}', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.savefig(_path + '/' + 'Evolution_radius_model' + '.png', dpi=300)
    if plot_show is True:
        plt.show()
    # we have to close figure.
    plt.close(fig)

# evolutionRadiusCore_Model(10.0, 50.0, 10.0)

def differenceWtithFLRW(_mvir, _sigma_m, _beta, _max_time,
                        _path,
                        _z=0, plot_show=False):
    """
    Analysing the equation for dimesionless density for late time, we can easy seen that mutliplication:
        rho_core * (r_{late_core})^(s) * (r_out)^(3-s) == 1,

    This have to be one if this model want to reproduce physical density profiles: NFW profile.
    """
    # --- set r_s and rho_s
    r_s = mod.cal_r_s(_mvir, _z)
    rho_s = mod.cal_rho_s(_mvir, _z)
    _power_mass = math.log10(_mvir)
    max_dimless_time = mod.cal_time_tilda(_max_time, rho_s, r_s, _sigma_m)

    # --- create class: model
    GravoModelClass = mod.ModelGravothermal(beta=_beta)
    par_s = GravoModelClass.new_param["s"]

    # --- settings dimensionless time
    log_max_dimless_T = math.log10(max_dimless_time)
    trans_dimless_T = GravoModelClass.return_time_transition()  # transition to core contraction in [dimensionless]
    log_trans_dimless_T = math.log10(trans_dimless_T)
    # divide time into two regimes
    dimless_late_T = np.logspace(log_trans_dimless_T, log_max_dimless_T)

    # --- find what we're looking for in ModelGravothermal
    def cal_constants():
        rho_core = 10 ** GravoModelClass.cof_log_rho_core(dimless_late_T)  # [rho_s]
        r_core = 10 ** GravoModelClass.cof_log_r_core(dimless_late_T)  # [r_s]
        r_out = 10 ** GravoModelClass.cof_log_r_out(dimless_late_T)  # [r_s]

        return rho_core * r_core ** par_s * r_out ** (3 - par_s)

    # --- draw PLOT
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.plot(dimless_late_T, cal_constants(), label='coefficient from model')
    plt.axhline(y=1.0, color='violet', linestyle=':',
                label='physically proper value of ceofficient,')
    plt.axvline(x=trans_dimless_T, color='black',
                label='change formula in model,' + "\n" +
                      f'ocured in: {"{:.2f}".format(trans_dimless_T)} [dimless].')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.xlabel(r'$\hat{t} \ \left[\frac{4}{\sqrt{4}} \hat{\sigma}_{m} \hat{\nu} \hat{\rho} \right]$', fontsize=18)
    plt.ylabel(r'$\frac{\rho(r \gg r_{out})}{\rho_{NFW}(r \gg r_{s})}$', fontsize=18)
    plt.title(f'Anphysical coefficient value. Mass=10^{_power_mass}, sigma={_sigma_m}, time={_max_time}',
              fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    plt.savefig(_path + '/' + 'Unphysical_coefficient' + '.png', dpi=300)
    if plot_show is True:
        plt.show()
    # we have to close figure.
    plt.close(fig)

# differenceWtithFLRW(10.0, 50.0, 20.0)
