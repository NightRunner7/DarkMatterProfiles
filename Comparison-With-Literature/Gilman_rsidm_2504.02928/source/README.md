# Clean Gilman benchmark modules

This folder contains a cleaned, modular version of the code used to reproduce and diagnose the Gilman/Tran RSIDM benchmark.

## Design principle

The code is split into three independent layers:

1. **Raw cross-section profile**
   \[
   \sigma(v_{\rm rel})/m \quad [{\rm cm^2/g}]
   \]
   implemented in `gilman_cross_sections.py`.

2. **Halo construction and velocity scale**
   implemented in `gilman_halos.py`.

3. **Maxwellian averaging**
   \[
   \sigma_{\rm eff}^{(p)} = \frac{\langle \sigma(v) v^p\rangle}{\langle v^p\rangle}
   \]
   implemented in `gilman_averaging.py`.

The high-level glue is in `gilman_benchmark.py`.

## Main workflow

```python
from gilman_cross_sections import YukawaNumerics, YukawaPartialWaveCrossSection
from gilman_averaging import MaxwellianAverager
from gilman_benchmark import make_default_gilman_halos, compute_gilman_rows

numerics = YukawaNumerics(x_min=1e-4, range_factor=120, rtol=1e-8, atol=1e-10)
profile = YukawaPartialWaveCrossSection.from_gilman_model("single_profile_fit", numerics=numerics)

halos = make_default_gilman_halos(explicit=True)
averager = MaxwellianAverager(p=5)
rows = compute_gilman_rows(halos, profile, averager, velocity_mode="gilman_analytic")
```

## Named Yukawa models

- `single_paper`: values printed in the paper.
- `single_email_rounded`: corrected rounded values from Daniel's email.
- `single_profile_fit`: working reconstruction using `m_phi = 6.56 MeV`.
- `multi_paper`: values printed in the paper for the multi-peak benchmark.

## Important note

The old `gilman_benchmark_profiles.py` computed `delta0` but still passed `y0=[0.0]` to `solve_ivp`. In this cleaned version the small-x initial condition is actually used by default.
