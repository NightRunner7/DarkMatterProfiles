import numpy as np
from RSIMD import sigma_m_eff_p5, sigma_m_eff_p1
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

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

# --- Collect all transformed points
def build_fixed_dataset(include=("dwarf", "LSB", "clusters", "MW")):
    """
    Return arrays:
      v      - velocities [km/s]
      y      - sigma/m [cm^2/g]
      labels - group labels for each point
    Return arrays of data points (v, sigma_over_m), ignoring error bars.
    """
    data = []

    if "dwarf" in include:
        for d in dwarf_sigma:
            data.append((10.0**d["xlog"], 10.0**d["ylog"], "dwarf"))

    if "LSB" in include:
        for d in LSB_sigma:
            data.append((10.0**d["xlog"], 10.0**d["ylog"], "LSB"))

    if "clusters" in include:
        for d in cluster_sigma:
            data.append((10.0**d["xlog"], 10.0**d["ylog"], "clusters"))

    if "MW" in include:
        for d in MW_sigma:
            data.append((10.0**d["xlog"], 10.0**d["ylog"], "MW"))

    v = np.array([x[0] for x in data], dtype=float)
    y = np.array([x[1] for x in data], dtype=float)
    labels = np.array([x[2] for x in data], dtype=object)

    return v, y, labels

# --- Model evaluated at the observational x-axis
def sigma_model_vs_vrel(vrel_kms, sigma_m0, m_GeV, L, Gamma, vR_kms, mode="p5"):
    """
    Evaluate model as a function of <v_rel> [km/s].
    Convert to nu via  nu = (sqrt(pi)/4) * <v_rel>.
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

def benchmark_loss_grouped_weighted(
    params,
    v_data,
    y_data,
    labels,
    L=0,
    mode="p5",
    group_weights=None,
):
    """
    Weighted grouped loss in log-space.

    params = (sigma_m0, m_GeV, Gamma, vR_kms)

    group_weights example:
        {
            "dwarf": 1.0,
            "LSB": 1.0,
            "clusters": 1.0,
            "MW": 0.5,
        }
    """
    sigma_m0, m_GeV, Gamma, vR_kms = params

    if sigma_m0 <= 0 or m_GeV <= 0 or Gamma <= 0 or vR_kms <= 0:
        return 1e100

    if group_weights is None:
        group_weights = {
            "dwarf": 1.0,
            "LSB": 1.0,
            "clusters": 1.0,
            "MW": 1.0,
        }

    try:
        y_model = sigma_model_vs_vrel(
            v_data,
            sigma_m0=sigma_m0,
            m_GeV=m_GeV,
            L=L,
            Gamma=Gamma,
            vR_kms=vR_kms,
            mode=mode,
        )

        if np.any(~np.isfinite(y_model)) or np.any(y_model <= 0):
            return 1e100

        res2 = (np.log10(y_model) - np.log10(y_data))**2

        loss = 0.0
        used_groups = []

        for group in ["dwarf", "LSB", "clusters", "MW"]:
            mask = (labels == group)
            if np.any(mask):
                w = group_weights.get(group, 1.0)
                loss += w * np.mean(res2[mask])
                used_groups.append(group)

        if len(used_groups) == 0:
            return 1e100

        return loss

    except Exception:
        return 1e100

# --- Plot the best benchmark against the data
def plot_best_benchmark_fit(
        fit_result,
        include=("dwarf", "LSB", "clusters", "MW"),
        L=0,
        mode="p5",
        savepath="best_benchmark_fit.png"
):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    plot_with_errors(ax, dwarf_sigma,   color="royalblue",   label="Dwarfs",    zorder=3)
    plot_with_errors(ax, LSB_sigma,     color="forestgreen", label="LSB",       zorder=3)
    plot_with_errors(ax, cluster_sigma, color="orange",      label="Clusters",  zorder=3)
    plot_with_errors(ax, MW_sigma,      color="gray",        label="MW dwarfs", zorder=3)
    plot_ufd_almeida_point(ax)

    vgrid = np.logspace(np.log10(2.0), np.log10(4000.0), 800)
    ygrid = sigma_model_vs_vrel(
        vgrid,
        sigma_m0=fit_result["sigma_m0"],
        m_GeV=fit_result["m_GeV"],
        L=L,
        Gamma=fit_result["Gamma"],
        vR_kms=fit_result["vR_kms"],
        mode=mode
    )

    ax.loglog(
        vgrid, ygrid,
        color="peru", lw=2.0,
        label=(rf"best {mode}: "
               rf"$\sigma_0={fit_result['sigma_m0']:.3g}$, "
               rf"$m={fit_result['m_GeV']:.3g}$ GeV, "
               rf"$\Gamma={fit_result['Gamma']:.2e}$, "
               rf"$v_R={fit_result['vR_kms']:.1f}$ km/s")
    )

    # model at actual data x-positions
    v_data, y_data, labels = build_fixed_dataset(include=include)
    y_best = sigma_model_vs_vrel(
        v_data,
        sigma_m0=fit_result["sigma_m0"],
        m_GeV=fit_result["m_GeV"],
        L=L,
        Gamma=fit_result["Gamma"],
        vR_kms=fit_result["vR_kms"],
        mode=mode
    )

    ax.scatter(
        v_data, y_best,
        marker='s', s=32,
        color='goldenrod', edgecolor='black',
        linewidth=0.6, zorder=5,
        label='model at data velocities'
    )

    ax.set_xlim(2, 4000)
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel(r'$\langle v_{\rm rel}\rangle\ [{\rm km/s}]$')
    if mode=="p5":
        ax.set_ylabel(r'${\rm K}_{5}=\sigma_{\rm self}/m\ [{\rm cm^2/g}]$')
    elif mode=="p1":
        ax.set_ylabel(r'${\rm K}_{1}=\sigma_{\rm self}/m\ [{\rm cm^2/g}]$')
    ax.legend(frameon=False, loc='lower left', ncol=2)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(savepath, dpi=220)
    plt.show()

# --- NEW MODEL
def evaluate_profile_on_grid(
    params,
    L=0,
    mode="p5",
    vmin=2.0,
    vmax=3000.0,
    ngrid=250
):
    """
    Evaluate sigma/m profile on a log-spaced velocity grid.
    """
    sigma_m0, m_GeV, Gamma, vR_kms = params

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
    return vgrid, ygrid

def physical_score(
    params,
    L=0,
    mode="p5",
    verbose=False,
):
    """
    Physical benchmark score for rSIDM-like profiles.

    Smaller is better.

    The score rewards:
    - reasonably large cross section at low velocities,
    - small cross section at cluster velocities,
    - a peak in a sensible velocity range,
    - non-flat velocity dependence.
    """
    sigma_m0, m_GeV, Gamma, vR_kms = params

    # safety guard
    if sigma_m0 <= 0 or m_GeV <= 0 or Gamma <= 0 or vR_kms <= 0:
        return 1e100

    try:
        vgrid, ygrid = evaluate_profile_on_grid(
            params,
            L=L,
            mode=mode,
            vmin=2.0,
            vmax=3000.0,
            ngrid=250
        )

        if np.any(~np.isfinite(ygrid)) or np.any(ygrid <= 0):
            return 1e100

        # --- representative scales ---
        sigma_30   = float(sigma_model_vs_vrel([30.0],   sigma_m0, m_GeV, L, Gamma, vR_kms, mode=mode)[0])
        sigma_50   = float(sigma_model_vs_vrel([50.0],   sigma_m0, m_GeV, L, Gamma, vR_kms, mode=mode)[0])
        sigma_100  = float(sigma_model_vs_vrel([100.0],  sigma_m0, m_GeV, L, Gamma, vR_kms, mode=mode)[0])
        sigma_1000 = float(sigma_model_vs_vrel([1000.0], sigma_m0, m_GeV, L, Gamma, vR_kms, mode=mode)[0])

        # --- peak diagnostics ---
        imax = np.argmax(ygrid)
        v_peak = float(vgrid[imax])
        y_peak = float(ygrid[imax])

        # amplitude of velocity dependence
        amp = np.log10(np.max(ygrid)) - np.log10(np.min(ygrid))

        score = 0.0

        # ==========================================================
        # 1) Low-velocity scale should be moderately/highly interacting
        # target: sigma(100 km/s) around ~1 cm^2/g, but not rigidly
        # ==========================================================
        score += 1.0 * (np.log10(sigma_100) - np.log10(1.0))**2

        # ==========================================================
        # 2) Cluster scale should be small
        # target: sigma(1000 km/s) around ~0.1 cm^2/g
        # ==========================================================
        score += 2.0 * (np.log10(sigma_1000) - np.log10(0.1))**2

        # ==========================================================
        # 3) Reward strong contrast between low-v and cluster scales
        # target contrast sigma(30)/sigma(1000) ~ 100
        # ==========================================================
        # contrast = sigma_30 / sigma_1000
        # score += 1.5 * (np.log10(contrast) - np.log10(100.0))**2

        # ==========================================================
        # 4) Peak should lie in a physically interesting range
        # prefer roughly 4-80 km/s
        # ==========================================================
        if v_peak < 4.0:
            score += 3.0 * (np.log10(4.0 / v_peak))**2
        elif v_peak > 80.0:
            score += 3.0 * (np.log10(v_peak / 80.0))**2

        # ==========================================================
        # 5) Penalize profiles that are too flat
        # ==========================================================
        if amp < 1.0:
            score += 4.0 * (1.0 - amp)**2

        # ==========================================================
        # 6) Penalize if "peak" is not actually above the plateau enough
        # ==========================================================
        baseline = 0.5 * (sigma_30 + sigma_1000)
        if y_peak < 2.0 * baseline:
            score += 2.0 * (1.0 - y_peak / (2.0 * baseline))**2

        if verbose:
            print("\n=== physical_score diagnostics ===")
            print(f"sigma(30)    = {sigma_30:.6g}")
            print(f"sigma(50)    = {sigma_50:.6g}")
            print(f"sigma(100)   = {sigma_100:.6g}")
            print(f"sigma(1000)  = {sigma_1000:.6g}")
            print(f"v_peak       = {v_peak:.6g}")
            print(f"y_peak       = {y_peak:.6g}")
            print(f"amplitude    = {amp:.6g}")
            # print(f"contrast     = {contrast:.6g}")
            print(f"score        = {score:.6g}")

        return score

    except Exception:
        return 1e100

def combined_score(
    params,
    include=("dwarf", "LSB", "clusters", "MW"),
    group_weights=None,
    L=0,
    mode="p5",
    lambda_data=0.35,
    lambda_phys=1.0,
    verbose=False,
):
    """
    Combined score:
    - physical score dominates,
    - data-fit term is only a secondary guidance.
    """
    v_data, y_data, labels = build_fixed_dataset(include=include)

    loss_data = benchmark_loss_grouped_weighted(
        params,
        v_data=v_data,
        y_data=y_data,
        labels=labels,
        L=L,
        mode=mode,
        group_weights=group_weights,
    )

    loss_phys = physical_score(
        params,
        L=L,
        mode=mode,
        verbose=False,
    )

    total = lambda_data * loss_data + lambda_phys * loss_phys

    if verbose:
        print("\n=== combined_score ===")
        print(f"loss_data = {loss_data:.6g}")
        print(f"loss_phys = {loss_phys:.6g}")
        print(f"total     = {total:.6g}")

    return total

def make_physical_objective(
    include=("dwarf", "LSB", "clusters", "MW"),
    group_weights=None,
    L=0,
    mode="p5",
    lambda_data=0.35,
    lambda_phys=1.0,
):
    def objective_log(logp):
        p = 10.0**np.array(logp)
        return combined_score(
            p,
            include=include,
            group_weights=group_weights,
            L=L,
            mode=mode,
            lambda_data=lambda_data,
            lambda_phys=lambda_phys,
        )
    return objective_log


def fit_benchmark_physical(
        include=("dwarf", "LSB", "clusters", "MW"),
        group_weights=None,
        L=0,
        mode="p5",
        init=(0.1, 0.1, 1e-12, 120.0),
        bounds_log10=(
            (-4.0,  0.0),   # sigma_m0 in [1e-4, 1]
            (-3.0,  0.0),   # m_GeV   in [1e-3, 1]
            (-14.0, -8.0),  # Gamma   in [1e-14, 1e-8]
            ( 1.5,  3.0),   # vR_kms  in [30, 1000]
        ),
        lambda_data=0.35,
        lambda_phys=1.0,
        verbose=True,
):
    """
    Fit benchmark using physical score + weak data guidance.
    """
    objective = make_physical_objective(
        include=include,
        group_weights=group_weights,
        L=L,
        mode=mode,
        lambda_data=lambda_data,
        lambda_phys=lambda_phys,
    )

    x0 = np.log10(np.array(init, dtype=float))

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds_log10
    )

    best = 10.0**result.x

    fit_result = {
        "sigma_m0": best[0],
        "m_GeV": best[1],
        "Gamma": best[2],
        "vR_kms": best[3],
        "loss": result.fun,
        "success": result.success,
        "message": result.message,
        "include": include,
        "group_weights": group_weights,
        "mode": mode,
        "init": init,
        "lambda_data": lambda_data,
        "lambda_phys": lambda_phys,
        "raw_result": result,
    }

    if verbose:
        print("\n=== Physical benchmark fit result ===")
        print("included groups =", include)
        print("weights =", group_weights)
        print(f"mode        = {mode}")
        print(f"sigma0      = {fit_result['sigma_m0']:.6g} cm^2/g")
        print(f"m_GeV       = {fit_result['m_GeV']:.6g} GeV")
        print(f"Gamma       = {fit_result['Gamma']:.6g}")
        print(f"vR_kms      = {fit_result['vR_kms']:.6g} km/s")
        print(f"score       = {fit_result['loss']:.6g}")
        print(f"success     = {fit_result['success']}")
        print(f"message     = {fit_result['message']}")
        print(f"lambda_data = {lambda_data}")
        print(f"lambda_phys = {lambda_phys}")

    return fit_result

# --- Optional: print residuals point by point
def print_pointwise_residuals(fit_result, include=("dwarf", "LSB", "clusters", "MW"), L=0, mode="p5"):
    v_data, y_data, labels = build_fixed_dataset(include=include)

    y_model = sigma_model_vs_vrel(
        v_data,
        sigma_m0=fit_result["sigma_m0"],
        m_GeV=fit_result["m_GeV"],
        L=L,
        Gamma=fit_result["Gamma"],
        vR_kms=fit_result["vR_kms"],
        mode=mode
    )

    print("\nPointwise residuals:")
    print(" idx   set        v[km/s]      data         model        log10(model/data)")
    for i, (lab, v, yd, ym) in enumerate(zip(labels, v_data, y_data, y_model), start=1):
        r = np.log10(ym / yd)
        print(f"{i:3d}   {lab:8s}  {v:9.2f}   {yd:10.4g}   {ym:10.4g}   {r: .4f}")

def print_fit_summary(name, fit_result):
    print(f"\n--- {name} ---")
    print("include :", fit_result["include"])
    print("weights :", fit_result["group_weights"])
    print("mode    :", fit_result["mode"])
    print(f"sigma0  : {fit_result['sigma_m0']:.6g}")
    print(f"m_GeV   : {fit_result['m_GeV']:.6g}")
    print(f"Gamma   : {fit_result['Gamma']:.6g}")
    print(f"vR_kms  : {fit_result['vR_kms']:.6g}")
    print(f"loss    : {fit_result['loss']:.6g}")

if __name__ == "__main__":
    include_data = ("LSB", "clusters", "MW")
    mode = "p5"

    fit_res = fit_benchmark_physical(
        include=include_data,
        group_weights={
            "dwarf": 1.0,
            "LSB": 1.0,
            "clusters": 1.0,
            "MW": 1.0,
        },
        L=0,
        mode=mode,
        init=(0.1, 0.1, 1e-12, 120.0),
        lambda_data=0.15,
        lambda_phys=1.0,
        verbose=True,
    )

    plot_best_benchmark_fit(
        fit_res,
        include=include_data,
        L=0,
        mode=mode,
        savepath="physical_benchmark_p5.png"
    )

    print_pointwise_residuals(
        fit_res,
        include=include_data,
        L=0,
        mode=mode
    )

    print_fit_summary(f"physical: {include_data}", fit_res)

    # optional diagnostics
    physical_score(
        (
            fit_res["sigma_m0"],
            fit_res["m_GeV"],
            fit_res["Gamma"],
            fit_res["vR_kms"],
        ),
        L=0,
        mode=mode,
        verbose=True,
    )