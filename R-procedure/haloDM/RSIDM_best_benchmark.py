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

# --- Loss function without error bars
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

# --- Helpfull function
def make_objective(
    include=("dwarf", "LSB", "clusters", "MW"),
    group_weights=None,
    L=0,
    mode="p5",
):
    v_data, y_data, labels = build_fixed_dataset(include=include)

    def objective_log(logp):
        p = 10.0**np.array(logp)
        return benchmark_loss_grouped_weighted(
            p,
            v_data=v_data,
            y_data=y_data,
            labels=labels,
            L=L,
            mode=mode,
            group_weights=group_weights,
        )

    return objective_log

# --- Fit routine
def fit_benchmark_to_points(
        include=("dwarf", "LSB", "clusters", "MW"),
        group_weights=None,
        L=0,
        mode="p5",
        init=(0.007, 0.02, 1e-11, 200.0),
        bounds_log10=(
            (-4.0,  0.0),   # sigma_m0 in [1e-4, 1]
            (-3.0,  0.0),   # m_GeV   in [1e-3, 1]
            (-14.0, -8.0),  # Gamma   in [1e-14, 1e-8]
            ( 1.5,  3.0),   # vR_kms  in [30, 1000]
        ),
        verbose=True,
):
    """
    Local fit using L-BFGS-B in log-parameter space.
    """

    objective = make_objective(
        include=include,
        group_weights=group_weights,
        L=L,
        mode=mode,
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
        "raw_result": result,
    }

    if verbose:
        print("\n=== Fit result ===")
        print("included groups =", include)
        print("weights =", group_weights)
        print(f"mode     = {mode}")
        print(f"sigma0   = {fit_result['sigma_m0']:.6g} cm^2/g")
        print(f"m_GeV    = {fit_result['m_GeV']:.6g} GeV")
        print(f"Gamma    = {fit_result['Gamma']:.6g}")
        print(f"vR_kms   = {fit_result['vR_kms']:.6g} km/s")
        print(f"loss     = {fit_result['loss']:.6g}")
        print(f"success  = {fit_result['success']}")
        print(f"message  = {fit_result['message']}")

    return fit_result
# --- Fit routine: Global
def fit_benchmark_global_then_local(
        include=("dwarf", "LSB", "clusters", "MW"),
        group_weights=None,
        L=0,
        mode="p5",
        bounds_log10=(
            (-4.0,  0.0),   # sigma_m0 in [1e-4, 1]
            (-3.0,  0.0),   # m_GeV   in [1e-3, 1]
            (-14.0, -8.0),  # Gamma   in [1e-14, 1e-8]
            ( 1.0,  3.0),   # vR_kms  in [10, 1000]
        ),
        seed=42,
        de_maxiter=40,
        de_popsize=18,
        verbose=True,
):
    """
    Global search with differential_evolution, then local refinement with L-BFGS-B.
    """

    objective = make_objective(
        include=include,
        group_weights=group_weights,
        L=L,
        mode=mode,
    )

    # --- step 1: global search ---
    result_global = differential_evolution(
        objective,
        bounds=bounds_log10,
        strategy="best1bin",
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        seed=seed,
        disp=verbose,
    )

    init_local = tuple(10.0**result_global.x)

    if verbose:
        print("\n=== Global search result ===")
        print(f"init for local refinement = {init_local}")
        print(f"global loss = {result_global.fun:.6g}")

    # --- step 2: local refinement ---
    fit_result = fit_benchmark_to_points(
        include=include,
        group_weights=group_weights,
        L=L,
        mode=mode,
        init=init_local,
        bounds_log10=bounds_log10,
        verbose=verbose,
    )

    fit_result["global_result"] = result_global
    fit_result["global_init_for_local"] = init_local

    return fit_result

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

# ###################################### Randomise to choose best benchmark ############################################
def run_multistart_benchmark_scan(
    initial_guesses,
    include=("dwarf", "LSB", "clusters", "MW"),
    group_weights=None,
    L=0,
    mode="p5",
    bounds_log10=(
        (-4.0,  0.0),   # sigma_m0 in [1e-4, 1]
        (-3.0,  0.0),   # m_GeV   in [1e-3, 1]
        (-14.0, -8.0),  # Gamma   in [1e-14, 1e-8]
        ( 1.5,  3.7),   # vR_kms  in [~32, 5000]
    ),
):
    """
    Run many fits from different initial conditions and collect all results.
    """
    results = []

    for i, init in enumerate(initial_guesses, start=1):
        print(f"\n### Multistart run {i}/{len(initial_guesses)} | init = {init}")

        fit_res = fit_benchmark_to_points(
            include=include,
            group_weights=group_weights,
            L=L,
            mode=mode,
            init=init,
            bounds_log10=bounds_log10,
        )

        fit_res["init"] = init
        results.append(fit_res)

    return results

def summarize_multistart_results(results, sort_by_loss=True):
    """
    Print a compact summary of all multistart solutions.
    """
    if sort_by_loss:
        results = sorted(results, key=lambda r: r["loss"])

    print("\n=== MULTISTART SUMMARY ===")
    for i, r in enumerate(results, start=1):
        print(
            f"{i:2d}) "
            f"loss={r['loss']:.6g} | "
            f"sigma0={r['sigma_m0']:.6g} | "
            f"m={r['m_GeV']:.6g} | "
            f"Gamma={r['Gamma']:.6g} | "
            f"vR={r['vR_kms']:.6g} | "
            f"init={r.get('init', None)}"
        )

def select_near_best_solutions(results, delta_loss=0.1):
    """
    Select all solutions with loss <= loss_min + delta_loss.
    """
    if len(results) == 0:
        return []

    loss_min = min(r["loss"] for r in results)
    selected = [r for r in results if r["loss"] <= loss_min + delta_loss]

    print(f"\nBest loss      = {loss_min:.6g}")
    print(f"delta_loss     = {delta_loss}")
    print(f"near-best count= {len(selected)} / {len(results)}")

    return selected

def choose_central_benchmark(results):
    """
    Choose a representative benchmark from a family of near-best solutions.

    Method:
    - compute the median in log-parameter space,
    - return the actual fitted solution closest to that median.
    """
    if len(results) == 0:
        raise ValueError("No results provided to choose_central_benchmark().")

    log_params = np.array([
        [
            np.log10(r["sigma_m0"]),
            np.log10(r["m_GeV"]),
            np.log10(r["Gamma"]),
            np.log10(r["vR_kms"]),
        ]
        for r in results
    ])

    log_median = np.median(log_params, axis=0)

    distances = np.sum((log_params - log_median[None, :])**2, axis=1)
    idx = np.argmin(distances)

    central = results[idx].copy()
    central["selection_method"] = "closest_to_log_median_of_near_best_family"

    print("\n=== CENTRAL BENCHMARK ===")
    print(f"sigma0  = {central['sigma_m0']:.6g}")
    print(f"m_GeV   = {central['m_GeV']:.6g}")
    print(f"Gamma   = {central['Gamma']:.6g}")
    print(f"vR_kms  = {central['vR_kms']:.6g}")
    print(f"loss    = {central['loss']:.6g}")
    print(f"init    = {central.get('init', None)}")

    return central

def print_near_best_spread(results):
    """
    Print ranges and medians for the near-best family.
    """
    if len(results) == 0:
        print("No near-best solutions.")
        return

    sigma0_vals = np.array([r["sigma_m0"] for r in results])
    m_vals      = np.array([r["m_GeV"] for r in results])
    Gamma_vals  = np.array([r["Gamma"] for r in results])
    vR_vals     = np.array([r["vR_kms"] for r in results])
    loss_vals   = np.array([r["loss"] for r in results])

    print("\n=== NEAR-BEST FAMILY SPREAD ===")
    print(f"count = {len(results)}")
    print(f"loss   : min={loss_vals.min():.6g}, median={np.median(loss_vals):.6g}, max={loss_vals.max():.6g}")
    print(f"sigma0 : min={sigma0_vals.min():.6g}, median={np.median(sigma0_vals):.6g}, max={sigma0_vals.max():.6g}")
    print(f"m_GeV  : min={m_vals.min():.6g}, median={np.median(m_vals):.6g}, max={m_vals.max():.6g}")
    print(f"Gamma  : min={Gamma_vals.min():.6g}, median={np.median(Gamma_vals):.6g}, max={Gamma_vals.max():.6g}")
    print(f"vR_kms : min={vR_vals.min():.6g}, median={np.median(vR_vals):.6g}, max={vR_vals.max():.6g}")

def plot_near_best_family(
    results,
    include=("dwarf", "LSB", "clusters", "MW"),
    L=0,
    mode="p5",
    savepath="near_best_family.png"
):
    """
    Plot all near-best curves together with data.
    """
    if len(results) == 0:
        print("No results to plot.")
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.8, 5.8))

    if "dwarf" in include:
        plot_with_errors(ax, dwarf_sigma, color="royalblue", label="Dwarfs", zorder=3)
    if "LSB" in include:
        plot_with_errors(ax, LSB_sigma, color="forestgreen", label="LSB", zorder=3)
    if "clusters" in include:
        plot_with_errors(ax, cluster_sigma, color="orange", label="Clusters", zorder=3)
    if "MW" in include:
        plot_with_errors(ax, MW_sigma, color="gray", label="MW dwarfs", zorder=3)

    vgrid = np.logspace(np.log10(10.0), np.log10(4000.0), 400)

    for i, r in enumerate(sorted(results, key=lambda rr: rr["loss"])):
        ygrid = sigma_model_vs_vrel(
            vgrid,
            sigma_m0=r["sigma_m0"],
            m_GeV=r["m_GeV"],
            L=L,
            Gamma=r["Gamma"],
            vR_kms=r["vR_kms"],
            mode=mode
        )

        ax.loglog(
            vgrid, ygrid,
            lw=1.2,
            alpha=0.5,
            color="peru",
            label="near-best family" if i == 0 else None
        )

    ax.set_xlim(10, 4000)
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel(r'$\langle v_{\rm rel}\rangle\ [{\rm km/s}]$')
    ax.set_ylabel(r'$\sigma_{\rm self}/m\ [{\rm cm^2/g}]$')
    ax.legend(frameon=False, loc='lower left', ncol=2)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(savepath, dpi=220)
    plt.show()

def compute_group_losses(
    fit_result,
    include=("dwarf", "LSB", "clusters", "MW"),
    L=0,
    mode="p5"
):
    """
    Compute per-group mean squared residual in log-space.
    """
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

    res2 = (np.log10(y_model) - np.log10(y_data))**2

    out = {}
    for group in ["dwarf", "LSB", "clusters", "MW"]:
        mask = (labels == group)
        if np.any(mask):
            out[group] = np.mean(res2[mask])

    return out


def print_group_losses(
    fit_result,
    include=("dwarf", "LSB", "clusters", "MW"),
    L=0,
    mode="p5"
):
    out = compute_group_losses(fit_result, include=include, L=L, mode=mode)

    print("\n=== GROUP CONTRIBUTIONS ===")
    for g, val in out.items():
        print(f"{g:8s} : {val:.6g}")

# if __name__ == "__main__":
#     initial_guesses = [
#         (0.007, 0.02, 1e-11,  80.0),
#         (0.010, 0.03, 1e-12, 120.0),
#         (0.003, 0.05, 1e-10, 200.0),
#         (0.050, 0.02, 1e-11, 300.0),
#         (0.100, 0.10, 1e-12,  60.0),
#         (0.020, 0.08, 1e-11,  50.0),
#         (0.300, 0.05, 1e-12,  70.0),
#         (0.500, 0.20, 1e-12,  50.0),
#     ]
#
#     all_results = run_multistart_benchmark_scan(
#         initial_guesses=initial_guesses,
#         include=("dwarf", "LSB", "clusters", "MW"),
#         group_weights={
#             "dwarf": 0.2,
#             "LSB": 0.2,
#             "clusters": 1.0,
#             "MW": 1.0,
#         },
#         L=0,
#         mode="p5",
#     )
#
#     summarize_multistart_results(all_results)
#
#     near_best = select_near_best_solutions(all_results, delta_loss=0.1)
#
#     print_near_best_spread(near_best)
#
#     central_benchmark = choose_central_benchmark(near_best)
#
#     print_group_losses(
#         central_benchmark,
#         include=("dwarf", "LSB", "clusters", "MW"),
#         L=0,
#         mode="p5"
#     )
#
#     plot_near_best_family(
#         near_best,
#         include=("dwarf", "LSB", "clusters", "MW"),
#         L=0,
#         mode="p5",
#         savepath="near_best_family.png"
#     )
#
#     plot_best_benchmark_fit(
#         central_benchmark,
#         include=("dwarf", "LSB", "clusters", "MW"),
#         L=0,
#         mode="p5",
#         savepath="central_benchmark_fit.png"
#     )
#
#     print_pointwise_residuals(
#         central_benchmark,
#         include=("dwarf", "LSB", "clusters", "MW"),
#         L=0,
#         mode="p5"
#     )
# ######################################################################################################################

if __name__ == "__main__":
    # include_data="dwarf", "LSB",  "clusters", "MW"
    include_data="dwarf", "LSB", "clusters", "MW"
    mode="p5"

    # --- Simple fit
    # fit_res = fit_benchmark_to_points(
    #     include=(include_data),
    #     group_weights={
    #         "dwarf": 1.0,
    #         "LSB": 1.0,
    #         "clusters": 1.0,
    #         "MW": 1.0,
    #     },
    #
    #     L=0,
    #     mode=mode,
    #     init=(0.100, 0.10, 1e-13,  260.0)
    # )

    # --- More sophisticated fit
    fit_res = fit_benchmark_global_then_local(
        include=include_data,
        group_weights={
            "dwarf": 0.2,
            "LSB": 0.2,
            "clusters": 1.0,
            "MW": 1.0,
        },
        L=0,
        mode=mode,
        bounds_log10=(
            (-4.0, 0.0),  # sigma_m0 in [1e-4, 1]
            (-3.0, 0.0),  # m_GeV   in [1e-3, 1]
            (-14.0, -8.0),  # Gamma   in [1e-14, 1e-8]
            (1.0, 3.0),  # vR_kms  in [30, 1000]
        ),
        seed=42,
        de_maxiter=40,
        de_popsize=18,
        verbose=True,
    )

    plot_best_benchmark_fit(
        fit_res,
        include=(include_data),
        L=0,
        mode=mode,
        # savepath="best_benchmark_p5.png"
    )

    print_pointwise_residuals(
        fit_res,
        include=(include_data),
        L=0,
        mode=mode
    )
    print_fit_summary(f"A: {include_data}", fit_res)
