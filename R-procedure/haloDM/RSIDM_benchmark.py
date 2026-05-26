import numpy as np
from RSIMD import sigma_m_eff_p5, sigma_m_eff_p1
import matplotlib.pyplot as plt

def plot_ufd_almeida_point(ax):
    """
    Add the UFD point from Almeida 2025 as a visual overlay only.
    Not used in the fit.
    """

    # central value chosen from the figure
    x = 5.0          # km/s
    y = 8.0          # cm^2/g

    # approximate asymmetric errors read from the figure
    # vertical span roughly ~0.35 to ~180 cm^2/g
    y_low = 0.35
    y_high = 180.0

    # horizontal span roughly ~3 to ~9 km/s
    x_low = 3.0
    x_high = 9.0

    xerr = np.array([[x - x_low], [x_high - x]])
    yerr = np.array([[y - y_low], [y_high - y]])

    ax.errorbar(
        x, y,
        xerr=xerr,
        yerr=yerr,
        fmt='o',
        ms=8,
        mew=1.2,
        color='red',
        ecolor='red',
        elinewidth=2.0,
        capsize=3.0,
        zorder=6,
        label='UFD (Almeida 2025)'
    )

def from_log_err_to_linear(xlog, ylog, xerr_log=None, yerr_log=None):
    x = 10.0 ** xlog
    y = 10.0 ** ylog

    if xerr_log is None:
        xerr = None
    else:
        xerr = np.vstack([
            x - 10.0 ** (xlog - xerr_log),
            10.0 ** (xlog + xerr_log) - x
        ])

    if yerr_log is None:
        yerr = None
    else:
        yerr = np.vstack([
            y - 10.0 ** (ylog - yerr_log),
            10.0 ** (ylog + yerr_log) - y
        ])

    return x, y, xerr, yerr


def plot_with_errors(ax, data, color, label, zorder=3, marker='o', ms=5.5):
    for pt in data:
        x, y, xerr, yerr = from_log_err_to_linear(
            pt["xlog"], pt["ylog"],
            pt.get("xerr"), pt.get("yerr")
        )
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr,
            fmt=marker, ms=ms, mew=1.0,
            elinewidth=1.2, capsize=2.5,
            color=color, ecolor=color,
            label=None, zorder=zorder
        )
    ax.plot([], [], marker=marker, color=color, linestyle='None', label=label)


def to_sigma_over_m(data):
    """
    Convert (log10 v, log10[(sigma*v)/m]) -> (log10 v, log10[sigma/m])
    """
    out = []
    for d in data:
        out.append({
            "xlog": d["xlog"],
            "ylog": d["ylog"] - d["xlog"],
            "xerr": d.get("xerr"),
            "yerr": d.get("yerr"),
        })
    return out

dwarfdata = [
    {"xlog":1.434, "ylog":2.109, "xerr":0.06779, "yerr":0.4713},
    {"xlog":1.656, "ylog":1.181, "xerr":0.06979, "yerr":0.4282},
    {"xlog":1.726, "ylog":2.156, "xerr":0.03695, "yerr":0.3697},
    {"xlog":1.792, "ylog":2.331, "xerr":0.03477, "yerr":0.3584},
    {"xlog":2.044, "ylog":3.186, "xerr":0.07675, "yerr":0.3988},
]

LSBdata = [
    {"xlog":1.824, "ylog":1.722, "xerr":0.09354, "yerr":0.5019},
    {"xlog":2.001, "ylog":1.695, "xerr":0.09466, "yerr":0.5327},
    {"xlog":2.003, "ylog":2.452, "xerr":0.02586, "yerr":0.3217},
    {"xlog":2.021, "ylog":3.152, "xerr":0.06952, "yerr":0.3573},
    {"xlog":2.092, "ylog":2.379, "xerr":0.03584, "yerr":0.342},
    {"xlog":2.097, "ylog":2.027, "xerr":0.03592, "yerr":0.342},
    {"xlog":2.248, "ylog":1.844, "xerr":0.04258, "yerr":0.3494},
]

clusterdata = [
    {"xlog":3.086, "ylog":1.872, "xerr":0.05747, "yerr":0.1735},
    {"xlog":3.105, "ylog":1.988, "xerr":0.1001,  "yerr":0.2854},
    {"xlog":3.092, "ylog":2.341, "xerr":0.0906,  "yerr":0.2682},
    {"xlog":3.148, "ylog":2.285, "xerr":0.07107, "yerr":0.3634},
    {"xlog":3.197, "ylog":2.246, "xerr":0.0689,  "yerr":0.3721},
    {"xlog":3.243, "ylog":2.547, "xerr":0.03906, "yerr":0.1881},
]

MWdataSigma = [
    {"xlog":1.318, "ylog":2.08,  "yerr":0.09},
    {"xlog":1.505, "ylog":1.975, "yerr":0.12},
    {"xlog":1.487, "ylog":1.650, "yerr":0.04},
    {"xlog":1.579, "ylog":1.80,  "yerr":0.10},
    {"xlog":1.680, "ylog":1.54,  "yerr":0.11},
    {"xlog":1.763, "ylog":1.77,  "yerr":0.08},
    {"xlog":1.760, "ylog":1.598, "yerr":0.10},
    {"xlog":1.795, "ylog":1.53,  "yerr":0.07},
    {"xlog":1.749, "ylog":1.460, "yerr":0.07},
]

dwarf_sigma   = to_sigma_over_m(dwarfdata)
LSB_sigma     = to_sigma_over_m(LSBdata)
cluster_sigma = to_sigma_over_m(clusterdata)
MW_sigma      = MWdataSigma

def sigma_model_vs_vrel(vrel_kms, sigma_m0, m_GeV, L, Gamma, vR_kms, mode="p5"):
    """
    Model curve as a function of <v_rel> [km/s], matching the observational x-axis.

    Internally convert:
        <v_rel> = (4/sqrt(pi)) * nu
    so
        nu = (sqrt(pi)/4) * <v_rel>.
    """
    vrel_kms = np.atleast_1d(vrel_kms)
    nu_kms = (np.sqrt(np.pi) / 4.0) * vrel_kms

    if mode == "p5":
        y = np.array([
            sigma_m_eff_p5(
                nu_kms=nu,
                sigma_m0=sigma_m0,
                m_GeV=m_GeV,
                L=L,
                Gamma=Gamma,
                vR_kms=vR_kms
            )
            for nu in nu_kms
        ])
    elif mode == "p1":
        y = np.array([
            sigma_m_eff_p1(
                nu_kms=nu,
                sigma_m0=sigma_m0,
                m_GeV=m_GeV,
                L=L,
                Gamma=Gamma,
                vR_kms=vR_kms
            )
            for nu in nu_kms
        ])
    else:
        raise ValueError("mode must be 'p5' or 'p1'")

    return y

def plot_benchmark_vs_data(
        sigma_m0=0.008,
        m_GeV=0.02,
        L=0,
        Gamma=6e-12,
        vR_kms=85.0,
        mode="p5",
        vmin=2.0,
        vmax=4000.0,
        ngrid=2500,
        benchmark_vpoints=None,
        savepath=None,
):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    # ---------------- data ----------------
    plot_with_errors(ax, dwarf_sigma,   color="royalblue",   label="Dwarfs",   zorder=3)
    plot_with_errors(ax, LSB_sigma,     color="forestgreen", label="LSB",      zorder=3)
    plot_with_errors(ax, cluster_sigma, color="orange",      label="Clusters", zorder=3)
    plot_with_errors(ax, MW_sigma,      color="gray",        label="MW dwarfs", zorder=3)
    plot_ufd_almeida_point(ax)

    # ---------------- benchmark curve ----------------
    vgrid = np.logspace(np.log10(vmin), np.log10(vmax), ngrid)
    ygrid = sigma_model_vs_vrel(
        vgrid,
        sigma_m0=sigma_m0,
        m_GeV=m_GeV,
        L=L,
        Gamma=Gamma,
        vR_kms=vR_kms,
        mode=mode
    )

    ax.loglog(
        vgrid, ygrid,
        color="peru", lw=1.8,
        label=rf"benchmark {mode}: $\sigma_0={sigma_m0},\ m={m_GeV},$"+"\n"+rf"$\Gamma={Gamma},\ v_R={vR_kms}$",
        zorder=2
    )

    # ---------------- benchmark points on the curve ----------------
    if benchmark_vpoints is not None:
        vpts = np.array(benchmark_vpoints, dtype=float)
        ypts = sigma_model_vs_vrel(
            vpts,
            sigma_m0=sigma_m0,
            m_GeV=m_GeV,
            L=L,
            Gamma=Gamma,
            vR_kms=vR_kms,
            mode=mode
        )

        ax.scatter(
            vpts, ypts,
            marker='s', s=36,
            color="goldenrod",
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label="benchmark points"
        )

    # ---------------- cosmetics ----------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(vmin, vmax)
    # ax.set_ylim(2e-4, 1e3)
    ax.set_ylim(2e-4, 4e3)

    ax.set_xlabel(r'$\langle v_{\rm rel}\rangle\ [{\rm km/s}]$')
    if mode=="p5":
        ax.set_ylabel(r'${\rm K}_{5}=\sigma_{\rm self}/m\ [{\rm cm^2/g}]$')
    elif mode=="p1":
        ax.set_ylabel(r'${\rm K}_{1}=\sigma_{\rm self}/m\ [{\rm cm^2/g}]$')

    ax.legend(frameon=False, loc='lower left', ncol=2)
    ax.grid(False)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=220)

    plt.show()

if __name__ == "__main__":
    # --- LUCA BENCHMARK
    # sigma_m0 = 0.007,
    # m_GeV = 0.02,
    # L = 0,
    # Gamma = 9.5e-12,
    # vR_kms = 200.0,

    # --- Old Benchmark
    # sigma_m0 = 0.008,
    # m_GeV = 0.02,
    # L = 0,
    # Gamma = 6e-12,
    # vR_kms = 85.0,

    # --- Ayuki proposition
    # sigma_m0 = 0.1,
    # m_GeV = 0.04,
    # L = 0,
    # Gamma = 1e-13,
    # vR_kms = 10.0,

    plot_benchmark_vs_data(
        sigma_m0=0.141,
        m_GeV=6.81,
        L=0,
        Gamma=1.93e-6,
        vR_kms=55.8,
        mode="p5",
        benchmark_vpoints=[25, 40, 60, 90, 130, 200, 1200, 1700],
        savepath="benchmark_k5_gradually.png"
    )

# SCANN MASS
# sigma_m0 = 0.141,
# m_GeV = 0.0301,
# L = 0,
# Gamma = 1.93e-13,
# vR_kms = 55.8,

# SCANN MASS
# sigma_m0 = 0.141,
# m_GeV = 6.81,
# L = 0,
# Gamma = 1.93e-6,
# vR_kms = 55.8,
