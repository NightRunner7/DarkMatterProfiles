import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pylab as pyl
# import files
import config as cfg
from IsothermalSIDMModel import deltaSqare

def plotFullComparison(NFW_profile, SIDM_LoDens, SIDM_HiDens, val_r1, sim_set, r_low=-3.0, r_bin=500):
    """
    Draw all the plots, which presents how looks like the CDM-only case, Isothermal case
    for low and high density (we will have two solution for Isothermal and first is physical and second
    is unphysical).

    where,

        NFW_profile: class, which contains all methods necessary to deal with profile CDM-only.
        (object)
        SIDM_LoDens: list contains all necessary data, which is corresponding to physical Isothermal
        solution.
        (list)
        SIDM_LoDens: list contains all necessary data, which is corresponding to unphysical Isothermal
        solution.
        (list)
        val_r1: the distance where we're switching formula from Isothermal to NFW. [kpc]
        (float)
        sim_set: list of simulation sets, which include all necessary values of variables need to
        get our results.
        (list)
        r_low: the lowest radius, which we consider `10^x`.
        (float)
        r_bin: how many radi we consider in selected region.
        (float)
    """
    # where save
    outfig1 = './IsothermalAndCMD_Mvir%.2f_c%.1f_sigmamx%.1f_tage%.4f.pdf'  # %(M_vir,c,sigmamx,tage)

    # set up the figure window
    fig1 = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    fig1.subplots_adjust(left=0.16, right=0.93, bottom=0.12, top=0.91,
                         hspace=0.32, wspace=0.32)
    gs = gridspec.GridSpec(2, 2)
    # take simulation parameters
    M_vir = np.log10(sim_set[0])
    c = sim_set[1]
    sigmamx = sim_set[2]
    tage = sim_set[3]
    fig1.suptitle(r'$M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}, t_\mathrm{age}=%.4f\mathrm{Gyr}$' \
        % (M_vir, c, sigmamx, tage), fontsize=14)

    # ##################### VARIABLES PLOTS ##################### #
    plot_r = np.logspace(r_low, 2.0, num=r_bin)  # [kpc]
    # ------------------------------ NFW PROFILE ------------------------------#
    # NFW profile: density profile
    plot_rho_NFW = NFW_profile.rho(plot_r)  # [M_sun/kpc^3]
    # NFW profile: circular velocity
    Vcirc_NFW = NFW_profile.Vcirc(plot_r)  # [kpc/Gyr]

    Vcirc_NFW_SI = []
    for i in range(0, len(plot_r)):
        Vcirc_SI = Vcirc_NFW[i] * cfg.kpc_SI / cfg.Gyr  # [m/s]
        # append
        Vcirc_NFW_SI.append(Vcirc_SI * 10 ** (-3))  # [km/s]

    # NFW profile: dispersion velocity
    Vdis_NFW = NFW_profile.sigma_accurate(plot_r)  # [kpc/Gyr]

    Vdis_NFW_km_s = []
    for i in range(0, len(plot_r)):
        Vdis_SI = Vdis_NFW[i] * cfg.kpc_SI / cfg.Gyr  # [m/s]
        # append
        Vdis_NFW_km_s.append(Vdis_SI * 10**(-3))  # [km/s]

    # ------------------------------ ISOTHERMAL ------------------------------#
    # Isothermal - low dense: density profile, radius's
    r_LoDens = SIDM_LoDens[4]  # [kpc]
    rho_LoDens = SIDM_LoDens[2]  # [M_sun/kpc^3]
    # Isothermal - high dense: density profile, radius's
    r_HiDens = SIDM_HiDens[4]  # [kpc]
    rho_HiDens = SIDM_HiDens[2]  # [M_sun/kpc^3]

    # Isothermal - low dense: circular velocity
    Vcirc_LoDens_starUnit = SIDM_LoDens[3]  # kpc/Gyr
    Vcirc_LoDens = Vcirc_LoDens_starUnit * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    # Isothermal - high dense: circular velocity
    Vcirc_HiDens_starUnit = SIDM_HiDens[3]
    Vcirc_HiDens = Vcirc_HiDens_starUnit * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]

    # Isothermal - low dense: dispersion velocity
    sigma0_LoDens = SIDM_LoDens[1]  # [kpc/Gyr]
    sigma0_LoDens_km_s = sigma0_LoDens * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma0_LoDens_list = [sigma0_LoDens_km_s] * len(r_LoDens)
    # Isothermal - high dense dense: dispersion velocity
    sigma0_HiDens = SIDM_HiDens[1]  # [kpc/Gyr]
    sigma0_HiDens_km_s = sigma0_HiDens * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma0_HiDens_list = [sigma0_HiDens_km_s] * len(r_HiDens)

    # ------------------------------ LINES ------------------------------#
    r_s = NFW_profile.r_s  # [kpc]
    Rres = cfg.Rres  # [kpc]
    r_1 = val_r1  # [kpc]

    # ------------------------------ PLOTTING DENSITY ------------------------------#
    ax = fig1.add_subplot(gs[0, 0])
    ax.plot(plot_r, plot_rho_NFW, '--', label="NFW: CDM-only")
    ax.plot(r_LoDens, rho_LoDens, label="SIDM: low density")
    ax.plot(r_HiDens, rho_HiDens, label="SIDM: high density")

    # lines
    plt.axvline(x=r_s, color='black', linestyle='dashdot',
                label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
    plt.axvline(x=r_1, color='grey', linestyle='dotted',
                label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
    plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9,
                label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # describe NFW profile
    plt.ylabel(r"$\rho \, \, [M_{\odot} \, kpc^{-3}]$", fontsize=18)
    plt.xlabel(r"$r$ [kpc]", fontsize=18)
    plt.title("Density profiles", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()

    # ------------------------------ PLOTTING CIRCULAR VELOCITY ------------------------------#
    ax = fig1.add_subplot(gs[0, 1])
    ax.plot(plot_r, Vcirc_NFW_SI, '--', label="NFW: CDM-only")
    ax.plot(r_LoDens, Vcirc_LoDens, label="SIDM: low density")
    ax.plot(r_HiDens, Vcirc_HiDens, label="SIDM: high density")

    # lines
    plt.axvline(x=r_s, color='black', linestyle='dashdot',
                label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
    plt.axvline(x=r_1, color='grey', linestyle='dotted',
                label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
    plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9,
                label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # describe NFW profile
    plt.ylim(10 ** 0, 5 * 10 ** 2)
    plt.ylabel(r"$V_{circular}$ [km / s]", fontsize=18)
    plt.xlabel(r"$r$ [kpc]", fontsize=18)
    plt.title("Circular velocity", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()

    # ------------------------------ PLOTTING DISPERSION VELOCITY ------------------------------#
    ax = fig1.add_subplot(gs[1, 0])
    ax.plot(plot_r, Vdis_NFW_km_s, '--', label="NFW: CDM-only")
    ax.plot(r_LoDens, sigma0_LoDens_list, label="SIDM: low density")
    ax.plot(r_HiDens, sigma0_HiDens_list, label="SIDM: high density")

    # lines
    plt.axvline(x=r_s, color='black', linestyle='dashdot',
                label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
    plt.axvline(x=r_1, color='grey', linestyle='dotted',
                label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
    plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9,
                label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
    # log scale
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # describe NFW profile
    plt.ylim(10 ** 1, 7 * 10 ** 1)
    plt.ylabel(r"$\nu$ [km / s]", fontsize=18)
    plt.xlabel(r"$r$ [kpc]", fontsize=18)
    plt.title("Dispersion velocity", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()

    # ##################### CONTOUR PLOTS ##################### #
    # ------------------------ PLOT CONTROL ------------------------ #
    # lw = 2.5
    size = 50.
    edgewidth = 0.
    r = np.logspace(r_low, np.log10(r_1), 500)  # [kpc] for plotting the full profile,

    # ------------------------ SETTING POSSIBLE VALUES OF PARAMETERS ------------------------ #
    beg_power = float(np.log10(NFW_profile.rho(r_1)))  # where we have to start

    rho_dm0 = np.logspace(beg_power, 14., 100)
    sigma_beg = 0.05 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma_end = 2 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma_0 = np.logspace(np.log10(sigma_beg), np.log10(sigma_end), 100)

    # ------------------------ NFW PROFILE ------------------------ #
    rhoCDM1 = NFW_profile.rho(r_1)
    MCDM1 = NFW_profile.Mass(r_1)

    # ------------------------ PRINTING IMPORTANT VALUES ------------------------ #
    # printing important values - which we will present on plot
    # print(f"CDM-olny, rho value at r1: {'{:.2E}'.format(rhoCDM1)} [M_sun / kpc^3] --> rho(r_1) ")
    rhoCDMRres = NFW_profile.rho(cfg.Rres)
    # print(f"CDM-olny, rho value at spatial resolution (0.01 kpc): {'{:.2E}'.format(rhoCDMRres)} [M_sun / kpc^3]. --> rho(r_res)")

    # velocity dispersion
    nu_at_r1 = NFW_profile.sigma_accurate(r_1)  # [kpc/Gyr]
    # nu_at_r1_SI = nu_at_r1 * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    # print(f"Velocity dispersion (nu) at r1: {'{:.1f}'.format(nu_at_r1_SI)} [km/s] --> v(r_1)")

    # circular velocity
    Vcirc_at_r1 = NFW_profile.Vcirc(r_1)  # [kpc/Gyr]
    Vcirc_at_r1_SI = Vcirc_at_r1 * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    # print(f"Circular velocity at r1: {'{:.1f}'.format(Vcirc_at_r1_SI)} [km/s]")

    # ------------------------ CALCULATE DELTA ------------------------ #
    def cal_delta(rho_dm0, sigma_0):
        """
        calculate the deltaSqare, which was appeared in file IsothermalSIDMModel.py
        """
        log_rho_dm0 = np.log10(rho_dm0)
        log_sigma_0 = np.log10(sigma_0)
        p_list = [log_rho_dm0, log_sigma_0]
        p = np.array(p_list)

        return deltaSqare(p, r_s, rhoCDM1, MCDM1, r)

    # ------------------------ PREPARE AXIS IN CONTOUR PLOT ------------------------ #
    x = sigma_0  # sigma_0 = np.logspace(np.log10(30), np.log10(118), 500)
    y = rho_dm0  # rho_dm0 = np.logspace(7., 14., 500)
    X, Y = pyl.meshgrid(x, y)  # grid of point

    # creating the Z's values (log delta)
    Z = []
    for yy in y:
        zy = []
        for xx in x:
            # calculate log delta
            log_delta = np.log10(cal_delta(yy, xx)) * 1 / 2
            zy.append(log_delta)

        Z.append(zy)

    # ------------------------ PLOTTING ------------------------#
    ax = fig1.add_subplot(gs[1, 1])

    # Z_to_show = np.arange(min(min(Z)),max(max(Z)),.1) #Adjust the .001 to get finer gradient
    # Z_to_show = [-2., -1.5, -1., -0.5, 0.]

    CS = ax.contourf(X, Y, Z, levels=50, cmap='nipy_spectral')

    # axis
    plt.ylim(10 ** 5, 2 * 10 ** 14)
    plt.xlim(10 ** 1, 2 * 10 ** 2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # grid
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.4)
    # for refined control of log-scale tick marks
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # tick length
    ax.tick_params('both', direction='in', top='on', right='on', length=10,
                   width=1, which='major', zorder=301)
    ax.tick_params('both', direction='in', top='on', right='on', length=5,
                   width=1, which='minor', zorder=301)

    # ax2.set_title('$\log \delta$', fontsize=18 )
    ax.set_xlabel(r"$v_{0}$ [km/s]", fontsize=18)
    ax.set_ylabel(r"$rho_{0} \, \, [M_{\odot} \, kpc^{-3}]$", fontsize=18)

    # scatter
    sigma0_LoDens = SIDM_LoDens[1]
    rhodm0_LoDens = SIDM_LoDens[0]
    sigma0_HiDens = SIDM_HiDens[1]
    rhodm0_HiDens = SIDM_HiDens[0]

    ax.scatter(sigma0_LoDens, rhodm0_LoDens, marker='*', s=size,
               facecolor='k', edgecolor='k', linewidth=edgewidth, rasterized=True)
    ax.scatter(sigma0_HiDens, rhodm0_HiDens, marker='*', s=size,
               facecolor='r', edgecolor='r', linewidth=edgewidth, rasterized=True)
    # lines
    plt.axvline(x=0.5 * nu_at_r1, color='black', linestyle='--',
                label=f'0.5 * Velocity dispersion (nu) at r1: {"{:.0f}".format(0.5 * nu_at_r1)} [km/s].')
    plt.axvline(x=2 * nu_at_r1, color='black', linestyle='--',
                label=f'2 * Velocity dispersion (nu) at r1: {"{:.0f}".format(2 * nu_at_r1)} [km/s].')
    plt.axhline(y=rhoCDM1, color='black', linestyle='--',
                label=f'rho value at r1: {"{:.2E}".format(rhoCDM1)} [M_sun / kpc^3].')
    plt.axhline(y=rhoCDMRres, color='black', linestyle='--',
                label=f'rho value at spatial resolution (0.01 kpc): {"{:.2E}".format(rhoCDMRres)} [M_sun / kpc^3].')
    # annotation to lines
    ax.text(0.5 * nu_at_r1, 2. * ax.get_ylim()[0], r'$0.5\, v(r_{1})$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=90)
    ax.text(2 * nu_at_r1, 2. * ax.get_ylim()[0], r'$2\, v(r_{1})}$', color='k', fontsize=16,
            ha='left', va='bottom', transform=ax.transData, rotation=90)

    ax.text(2. * ax.get_xlim()[0], rhoCDMRres, r'$\rho(r_{res})$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=0)
    ax.text(2. * ax.get_xlim()[0], rhoCDM1, r'$\rho(r_\mathrm{1})$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=0)

    # Make a color bar for the ContourSet returned by the contourf call.
    cbar = fig1.colorbar(CS, ticks=mticker.MaxNLocator(6))
    cbar.ax.set_ylabel(r'$\log \delta$', fontsize=18)

    # ---save figure
    plt.savefig(outfig1 % (M_vir, c, sigmamx, tage), dpi=300)
    # fig1.canvas.manager.window.raise_()
    # plt.get_current_fig_manager().window.setGeometry(50,50,1200,1200)

    # ---we have to close figure.
    plt.close(fig1)


def plotFullComparison_vol2(NFW_profile,
                            SIDM_LoDens,
                            SIDM_HiDens,
                            val_r1,
                            sim_set,
                            path_to_directory,
                            r_low=-3.0,
                            r_bin=500):
    """
    Draw all the plots, which presents how looks like the CDM-only case, Isothermal case
    for low and high density (we will have two solution for Isothermal and first is physical and second
    is unphysical).

    where

        NFW_profile: class, which contains all methods necessary to deal with profile CDM-only.
        (object)
        SIDM_LoDens: list contains all necessary data, which is corresponding to physical Isothermal
        solution.
        (list)
        SIDM_LoDens: list contains all necessary data, which is corresponding to unphysical Isothermal
        solution.
        (list)
        val_r1: the distance where we're switching formula from Isothermal to NFW. [kpc]
        (float)
        sim_set: list of simulation sets, which include all necessary values of variables need to
        get our results.
        (list)
        path_to_directory: a path to directory were you want to store your plot.
        (string)
        r_low: the lowest radius, which we consider `10^x`.
        (float)
        r_bin: how many radi we consider in selected region.
        (float)
    """
    # where save
    outfig1 = path_to_directory + '/IsothermalAndCMD_Mvir%.2f_c%.1f_sigmamx%.1f_tage%.4f.png'  # %(M_vir,c,sigmamx,tage)

    # set up the figure window
    fig1 = plt.figure(figsize=(18, 5.0), dpi=80, facecolor='w', edgecolor='k')
    fig1.subplots_adjust(left=0.16, right=0.93, bottom=0.12, top=0.91,
                         hspace=0.32, wspace=0.32)
    gs = gridspec.GridSpec(1, 3)
    # take simulation parameters
    M_vir = np.log10(sim_set[0])
    c = sim_set[1]
    sigmamx = sim_set[2]
    tage = sim_set[3]
    fig1.suptitle(r'$M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}, t_\mathrm{age}=%.4f\mathrm{Gyr}$' \
        % (M_vir, c, sigmamx, tage), fontsize=14)

    # ##################### VARIABLES PLOTS ##################### #
    plot_r = np.logspace(r_low, 2.0, num=r_bin)  # [kpc]
    # ------------------------------ NFW PROFILE ------------------------------#
    # NFW profile: density profile
    plot_rho_NFW = NFW_profile.rho(plot_r)  # [M_sun/kpc^3]
    # NFW profile: circular velocity
    Vcirc_NFW = NFW_profile.Vcirc(plot_r)  # [kpc/Gyr]

    Vcirc_NFW_SI = []
    for i in range(0, len(plot_r)):
        Vcirc_SI = Vcirc_NFW[i] * cfg.kpc_SI / cfg.Gyr  # [m/s]
        # append
        Vcirc_NFW_SI.append(Vcirc_SI * 10 ** (-3))  # [km/s]

    # NFW profile: dispersion velosity
    Vdis_NFW = NFW_profile.sigma_accurate(plot_r)  # [kpc/Gyr]

    Vdis_NFW_km_s = []
    for i in range(0, len(plot_r)):
        Vdis_SI = Vdis_NFW[i] * cfg.kpc_SI / cfg.Gyr  # [m/s]
        # append
        Vdis_NFW_km_s.append(Vdis_SI * 10**(-3))  # [km/s]

    # ------------------------------ ISOTHERMAL ------------------------------#
    # Isothermal - low dense: density profile, radius's
    r_LoDens = SIDM_LoDens[4]  # [kpc]
    rho_LoDens = SIDM_LoDens[2]  # [M_sun/kpc^3]
    # Isothermal - high dense: density profile, radius's
    r_HiDens = SIDM_HiDens[4]  # [kpc]
    rho_HiDens = SIDM_HiDens[2]  # [M_sun/kpc^3]

    # Isothermal - low dense: circular velocity
    Vcirc_LoDens_starUnit = SIDM_LoDens[3]  # kpc/Gyr
    Vcirc_LoDens = Vcirc_LoDens_starUnit * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    # Isothermal - high dense: circular velocity
    Vcirc_HiDens_starUnit = SIDM_HiDens[3]
    Vcirc_HiDens = Vcirc_HiDens_starUnit * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]

    # Isothermal - low dense: dispersion velocity
    sigma0_LoDens = SIDM_LoDens[1]  # [kpc/Gyr]
    sigma0_LoDens_km_s = sigma0_LoDens * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma0_LoDens_list = [sigma0_LoDens_km_s] * len(r_LoDens)
    # Isothermal - high dense dense: dispersion velocity
    sigma0_HiDens = SIDM_HiDens[1]  # [kpc/Gyr]
    sigma0_HiDens_km_s = sigma0_HiDens * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma0_HiDens_list = [sigma0_HiDens_km_s] * len(r_HiDens)

    # ------------------------------ LINES ------------------------------#
    r_s = NFW_profile.r_s  # [kpc]
    Rres = cfg.Rres  # [kpc]
    r_1 = val_r1  # [kpc]

    # ------------------------------ PLOTTING DENSITY ------------------------------#
    ax = fig1.add_subplot(gs[0, 0])
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # describe NFW profile
    plt.ylabel(r"$\rho \, \, [M_{\odot} \, kpc^{-3}]$", fontsize=18)
    plt.xlabel(r"$r$ [kpc]", fontsize=18)
    # data
    ax.plot(plot_r, plot_rho_NFW, '--', label="NFW: CDM-only")
    ax.plot(r_LoDens, rho_LoDens, label="SIDM: low density")
    ax.plot(r_HiDens, rho_HiDens, label="SIDM: high density")

    # grid
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.4)
    # for refined control of log-scale tick marks
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # tick length
    ax.tick_params('both', direction='in', top='on', right='on', length=10,
                   width=1, which='major', zorder=301)
    ax.tick_params('both', direction='in', top='on', right='on', length=5,
                   width=1, which='minor', zorder=301)

    # lines
    plt.axvline(x=r_s, color='black', linestyle='dashdot')
                # label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
    plt.axvline(x=r_1, color='grey', linestyle='dotted')
                # label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
    plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
                # label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
    # annotation to lines
    ax.text(r_s, 0.5 * ax.get_ylim()[1], r'$r_{s}$', color='k', fontsize=16,
            ha='right', va='top', transform=ax.transData, rotation=90)
    ax.text(r_1, 0.5 * ax.get_ylim()[1], r'$r_{1}$', color='grey', fontsize=16,
            ha='right', va='top', transform=ax.transData, rotation=90)
    ax.text(Rres, 0.5 * ax.get_ylim()[1], r'$r_{res}$', color='k', fontsize=16,
            ha='left', va='top', transform=ax.transData, rotation=90)

    # plt.title("Density profiles", fontsize=18)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()

    # ------------------------------ PLOTTING CIRCULAR VELOCITY ------------------------------#
    ax = fig1.add_subplot(gs[0, 1])
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # describe
    # plt.ylim(10 ** 0, 5 * 10 ** 2)
    plt.ylabel(r"$V_{circular}$ [km / s]", fontsize=18)
    plt.xlabel(r"$r$ [kpc]", fontsize=18)
    # data
    ax.plot(plot_r, Vcirc_NFW_SI, '--', label="NFW: CDM-only")
    ax.plot(r_LoDens, Vcirc_LoDens, label="SIDM: low density")
    ax.plot(r_HiDens, Vcirc_HiDens, label="SIDM: high density")

    # grid
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.4)
    # for refined control of log-scale tick marks
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # tick length
    ax.tick_params('both', direction='in', top='on', right='on', length=10,
                   width=1, which='major', zorder=301)
    ax.tick_params('both', direction='in', top='on', right='on', length=5,
                   width=1, which='minor', zorder=301)

    # lines
    plt.axvline(x=r_s, color='black', linestyle='dashdot')
                # label=f'r_s = {"{:.3f}".format(r_s)} [kpc].')
    plt.axvline(x=r_1, color='grey', linestyle='dotted')
                # label=f'r_1 = {"{:.3f}".format(r_1)} [kpc].')
    plt.axvline(x=Rres, color='black', linestyle='dotted', alpha=0.9)
                # label=f'Spatial resolution = {"{:.2f}".format(Rres)} [kpc].')
    # annotation to lines
    ax.text(r_s, 2. * ax.get_ylim()[0], r'$r_{s}$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=90)
    ax.text(r_1, 2. * ax.get_ylim()[0], r'$r_{1}$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=90)
    ax.text(Rres, 2. * ax.get_ylim()[0], r'$r_{res}$', color='k', fontsize=16,
            ha='left', va='bottom', transform=ax.transData, rotation=90)

    # plt.title("Circular velocity", fontsize=18)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()

    # ##################### CONTOUR PLOTS ##################### #
    # ------------------------ PLOT CONTROL ------------------------ #
    # lw = 2.5
    size = 50.
    edgewidth = 0.
    r = np.logspace(r_low, np.log10(r_1), r_bin)  # [kpc] for plotting the full profile

    # ------------------------ SETTING POSSIBLE VALUES OF PARAMETERS ------------------------ #
    beg_power = float(np.log10(NFW_profile.rho(r_1)))  # where we have to start

    # rho_dm0 = np.logspace(beg_power, 14., 100)  # instabilities occurring here
    rho_dm0 = np.logspace(beg_power, 13., 100)
    sigma_beg = 0.05 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma_end = 2 * NFW_profile.sigma_accurate(r_1) * cfg.kpc_SI / cfg.Gyr * 10**(-3)  # [km/s]
    sigma_0 = np.logspace(np.log10(sigma_beg), np.log10(sigma_end), 100)

    # ------------------------ NFW PROFILE ------------------------ #
    rhoCDM1 = NFW_profile.rho(r_1)
    MCDM1 = NFW_profile.Mass(r_1)

    # ------------------------ PRINTING IMPORTANT VALUES ------------------------ #
    # printing important values - which we will present on plot
    # print(f"CDM-olny, rho value at r1: {'{:.2E}'.format(rhoCDM1)} [M_sun / kpc^3] --> rho(r_1) ")
    rhoCDMRres = NFW_profile.rho(cfg.Rres)
    # print(f"CDM-olny, rho value at spatial resolution (0.01 kpc): {'{:.2E}'.format(rhoCDMRres)} [M_sun / kpc^3]. --> rho(r_res)")

    # velocity dispersion
    nu_at_r1 = NFW_profile.sigma_accurate(r_1)  # [kpc/Gyr]
    nu_at_r1_SI =nu_at_r1 * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    # print(f"Velocity dispersion (nu) at r1: {'{:.1f}'.format(nu_at_r1_SI)} [km/s] --> v(r_1)")

    # circular velocity
    Vcirc_at_r1 = NFW_profile.Vcirc(r_1)  # [kpc/Gyr]
    Vcirc_at_r1_SI = Vcirc_at_r1 * cfg.kpc_SI / cfg.Gyr * 10 ** (-3)  # [km/s]
    # print(f"Circular velocity at r1: {'{:.1f}'.format(Vcirc_at_r1_SI)} [km/s]")

    # ------------------------ CALCULATE DELTA ------------------------ #
    def cal_delta(rho_dm0, sigma_0):
        """
        calculate the deltaSqare, which was appeared in file IsothermalSIDMModel.py
        """
        log_rho_dm0 = np.log10(rho_dm0)
        log_sigma_0 = np.log10(sigma_0)
        p_list = [log_rho_dm0, log_sigma_0]
        p = np.array(p_list)

        return deltaSqare(p, r_s, rhoCDM1, MCDM1, r)

    # ------------------------ PREPARE AXIX IN CONTOUR PLOT ------------------------ #
    x = sigma_0  # sigma_0 = np.logspace(np.log10(30), np.log10(118), 500)
    y = rho_dm0  # rho_dm0 = np.logspace(7., 14., 500)
    X, Y = pyl.meshgrid(x, y)  # grid of point

    # creating the Z's values (log delta)
    Z = []
    for yy in y:
        zy = []
        for xx in x:
            # calculate log delta
            log_delta = np.log10(cal_delta(yy, xx)) * 1 / 2
            zy.append(log_delta)

        Z.append(zy)

    # ------------------------ PLOTTING ------------------------#
    ax = fig1.add_subplot(gs[0, 2])

    # Z_to_show = np.arange(min(min(Z)),max(max(Z)),.1) #Adjust the .001 to get finer gradient
    # Z_to_show = [-2., -1.5, -1., -0.5, 0.]

    CS = ax.contourf(X, Y, Z, levels=50, cmap='nipy_spectral')

    # axis
    plt.ylim(0.3 * rhoCDM1, 2 * 10 ** 14)
    plt.xlim(0.3 * 0.05 * nu_at_r1, 3 * 2.0 * nu_at_r1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # grid
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.4)
    # for refined control of log-scale tick marks
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    locmaj = mticker.LogLocator(base=10, numticks=12)
    locmin = mticker.LogLocator(base=10.0,
                                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # tick length
    ax.tick_params('both', direction='in', top='on', right='on', length=10,
                   width=1, which='major', zorder=301)
    ax.tick_params('both', direction='in', top='on', right='on', length=5,
                   width=1, which='minor', zorder=301)

    # ax2.set_title('$\log \delta$', fontsize=18 )
    ax.set_xlabel(r"$v_{0}$ [km/s]", fontsize=18)
    ax.set_ylabel(r"$rho_{0} \, \, [M_{\odot} \, kpc^{-3}]$", fontsize=18)

    # scatter
    sigma0_LoDens = SIDM_LoDens[1]
    rhodm0_LoDens = SIDM_LoDens[0]
    sigma0_HiDens = SIDM_HiDens[1]
    rhodm0_HiDens = SIDM_HiDens[0]

    ax.scatter(sigma0_LoDens, rhodm0_LoDens, marker='*', s=size,
               facecolor='k', edgecolor='k', linewidth=edgewidth, rasterized=True)
    ax.scatter(sigma0_HiDens, rhodm0_HiDens, marker='*', s=size,
               facecolor='r', edgecolor='r', linewidth=edgewidth, rasterized=True)
    # lines
    plt.axvline(x=0.05 * nu_at_r1, color='black', linestyle='--',
                label=f'0.05 * Velocity dispersion (nu) at r1: {"{:.0f}".format(0.05 * nu_at_r1)} [km/s].')
    plt.axvline(x=2 * nu_at_r1, color='black', linestyle='--',
                label=f'2 * Velocity dispersion (nu) at r1: {"{:.0f}".format(2 * nu_at_r1)} [km/s].')
    plt.axhline(y=rhoCDM1, color='black', linestyle='--',
                label=f'rho value at r1: {"{:.2E}".format(rhoCDM1)} [M_sun / kpc^3].')
    plt.axhline(y=rhoCDMRres, color='black', linestyle='--',
                label=f'rho value at spatial resolution (0.01 kpc): {"{:.2E}".format(rhoCDMRres)} [M_sun / kpc^3].')
    # annotation to lines
    ax.text(0.05 * nu_at_r1, 2. * ax.get_ylim()[0], r'$0.05\, v(r_{1})$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=90)
    ax.text(2 * nu_at_r1, 2. * ax.get_ylim()[0], r'$2\, v(r_{1})}$', color='k', fontsize=16,
            ha='left', va='bottom', transform=ax.transData, rotation=90)

    ax.text(2. * ax.get_xlim()[0], rhoCDMRres, r'$\rho(r_{res})$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=0)
    ax.text(2. * ax.get_xlim()[0], rhoCDM1, r'$\rho(r_\mathrm{1})$', color='k', fontsize=16,
            ha='right', va='bottom', transform=ax.transData, rotation=0)

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig1.colorbar(CS, ticks=mticker.MaxNLocator(6))
    cbar.ax.set_ylabel(r'$\log \delta$', fontsize=18)

    # ---save figure
    plt.savefig(outfig1 % (M_vir, c, sigmamx, tage), dpi=300)
    # fig1.canvas.manager.window.raise_()
    # plt.get_current_fig_manager().window.setGeometry(50,50,1200,1200)

    # ---we have to close figure.
    plt.close(fig1)
