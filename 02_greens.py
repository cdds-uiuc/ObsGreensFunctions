# %% [markdown]
# # 02 · Green's functions from amip-piForcing
#
# **The idea.** In `amip-piForcing` the forcing is held at pre-industrial, so the net
# top-of-atmosphere anomaly is *only* the radiative response: **N′(t) = λ·T′(t)** (no
# unknown forcing to worry about). We use this clean setting to learn two **Green's
# functions** — maps that turn an SST *pattern* into a global-mean response:
#
# $$N'(t) = \sum_x G_\text{toa}(x)\,T'(x,t), \qquad T'(t) = \sum_x G_\text{tas}(x)\,T'(x,t)$$
#
# where T′(x,t) is the tas anomaly at ocean point x (our SST proxy — tos isn't reported
# in amip-piForcing). There are ~5000 ocean points and only 145 years, so we fit with
# **ridge regression**. Anomalies are relative to the 1870–1919 climatology.
#
# We walk through **CanESM5** end to end — anomalies, the ridge, how α is chosen, the
# held-out validation — then loop the same recipe over every model.

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from obsgf import config, plotting
from obsgf.config import (ANALYSIS_YEARS, BASELINE_YEARS, DERIVED_DIR,
                          analysis_models, find_preprocessed)
from obsgf.masks import ocean_mask, global_mean

# --- knobs for this notebook (edit freely) ---
ALPHAS = np.logspace(0, 6, 25)   # ridge penalties to search over
N_CV_BLOCKS = 5                  # blocked cross-validation folds (contiguous in time)
HOLDOUT_LENGTH = 20              # length (yr) of each held-out validation segment
PREDICTOR_LAT_BOUND = 55         # keep SST predictors within ±this latitude (avoid sea ice)
WALKTHROUGH_MODEL = "CanESM5"

# %% [markdown]
# ## The two global-mean targets
#
# Load tas and toa, take anomalies vs 1870–1919, and form the two things the GFs predict:
# global-mean net TOA **N** and global-mean temperature **T** (both area-weighted over the
# *full* globe — λ is defined per unit global surface-air temperature).

# %%
def anomaly(da):
    """Anomaly relative to the 1870–1919 baseline climatology."""
    return da - da.sel(year=slice(*BASELINE_YEARS)).mean("year")


def load_amip(var, model):
    ds = xr.open_dataset(find_preprocessed(var, "amip-piForcing", model))
    return ds[var].sel(year=slice(*ANALYSIS_YEARS))


N = global_mean(anomaly(load_amip("toa", WALKTHROUGH_MODEL))).values   # global net TOA (W/m²)
T = global_mean(anomaly(load_amip("tas", WALKTHROUGH_MODEL))).values   # global temperature (K)
year = anomaly(load_amip("tas", WALKTHROUGH_MODEL)).year.values

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3))
a1.plot(year, N, "k"); a1.set_title(f"{WALKTHROUGH_MODEL}: N (global net TOA)"); a1.set_ylabel("W/m²")
a2.plot(year, T, "k"); a2.set_title("T (global temperature)"); a2.set_ylabel("K");

# %% [markdown]
# ## Predictor: the ocean SST-anomaly pattern (from `ts`, within ±55°)
#
# The predictor is the map of **`ts` anomalies at ocean points** — over open ocean `ts`
# *is* the prescribed SST, a cleaner signal than 2-m air temp. We restrict to |lat| ≤
# `PREDICTOR_LAT_BOUND`: poleward of that, `ts` over sea ice is the ice-surface
# temperature, not the ocean beneath (AMIP can't give us the ocean under ice), and those
# cells carry little of the SST-forced radiation anyway. Columns are √cos(lat)-scaled so
# the ridge penalty is area-fair; missing SST → 0.

# %%
def predictor_mask(model, lat_bound=PREDICTOR_LAT_BOUND):
    """Ocean points within ±lat_bound — the cells whose ts we trust as SST."""
    m = ocean_mask(model)
    return m & (np.abs(m.lat) <= lat_bound)


def ocean_points(mask):
    m = mask.stack(pt=("lat", "lon"))
    return m.pt[m.values]           # the ocean-point index (fixes column order)


def ocean_sst_matrix(anom, pts):
    return anom.stack(pt=("lat", "lon")).sel(pt=pts).fillna(0.0)   # (year, ocean-point)


def area_weights(X):
    # cos clipped at 0: some regridded grids carry a latitude a hair beyond ±90
    return np.sqrt(np.clip(np.cos(np.deg2rad(X.lat.values)), 0, None))


ts_anom = anomaly(load_amip("ts", WALKTHROUGH_MODEL))   # SST proxy — the predictor field
mask = predictor_mask(WALKTHROUGH_MODEL)
pts = ocean_points(mask)
X = ocean_sst_matrix(ts_anom, pts)       # (year, ocean-point) DataArray
w = area_weights(X)                      # √cos(lat) column scaling
Xw = X.values * w                        # area-fair design matrix
print("design matrix:", Xw.shape, "= (years, ocean points within ±%d°)" % PREDICTOR_LAT_BOUND)
plotting.map_plot(mask.astype(float), title=f"{WALKTHROUGH_MODEL} predictor mask (ocean, ±{PREDICTOR_LAT_BOUND}°)",
                  cmap="Blues", cbar=False);

# %% [markdown]
# ## Ridge regression (dual form)
#
# We minimize ‖y − Xβ‖² + α‖β‖². With far more points than years, the **dual form**
# `β = Xᵀ(XXᵀ + αI)⁻¹y` is exact and cheap (it inverts a 145×145 matrix). No intercept:
# zero SST anomaly must give zero response. α is the ridge penalty — larger = smoother.

# %%
def dual_ridge(X, y, alpha):
    """β minimizing ‖y − Xβ‖² + α‖β‖², no intercept (dual form)."""
    K = X @ X.T
    a = np.linalg.solve(K + alpha * np.eye(len(y)), y)
    return X.T @ a

# %% [markdown]
# ## Choosing α by blocked cross-validation
#
# We hold out contiguous blocks of years (blocked, because SST is autocorrelated — random
# folds would leak), and pick the α with the lowest cross-validated error. The curve below
# shows the trade-off; the star marks the choice for N.

# %%
def cv_mse(X, y, alpha, folds):
    errs = []
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        beta = dual_ridge(X[tr], y[tr], alpha)
        errs.append(np.mean((y[f] - X[f] @ beta) ** 2))
    return np.mean(errs)


def select_alpha(X, y, alphas=ALPHAS, k=N_CV_BLOCKS):
    folds = np.array_split(np.arange(len(y)), k)
    scores = [cv_mse(X, y, a, folds) for a in alphas]
    return float(alphas[int(np.argmin(scores))])


folds = np.array_split(np.arange(len(N)), N_CV_BLOCKS)
cv_curve = [cv_mse(Xw, N, a, folds) for a in ALPHAS]
alpha_N = ALPHAS[int(np.argmin(cv_curve))]
plt.figure(figsize=(6, 3.2))
plt.loglog(ALPHAS, cv_curve, "o-")
plt.loglog(alpha_N, min(cv_curve), "r*", ms=15)
plt.xlabel("ridge α"); plt.ylabel("blocked-CV MSE of N"); plt.title(f"α for N = {alpha_N:.0f}");

# %% [markdown]
# ## Held-out validation — the honest skill test
#
# All validation stays *inside* amip-piForcing (historical has no clean truth). We hold
# out a 20-yr segment, refit on the rest (α re-selected within the training years), and
# check how well N and T are reconstructed on the unseen segment. We do this for an early,
# middle, and late segment — the late one matters most (that's when the pattern effect is
# strongest).

# %%
def holdout_segments(n_years, length=HOLDOUT_LENGTH):
    mid = n_years // 2 - length // 2
    return {"early": np.arange(0, length),
            "middle": np.arange(mid, mid + length),
            "late": np.arange(n_years - length, n_years)}


def reconstruction_skill(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {"r2": float(1 - ss_res / ss_tot),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))}


# validate N reconstruction on each held-out segment, and plot it
fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=True)
for ax, (name, test) in zip(axes, holdout_segments(len(N)).items()):
    train = np.setdiff1d(np.arange(len(N)), test)
    alpha = select_alpha(Xw[train], N[train])
    beta = dual_ridge(Xw[train], N[train], alpha)
    N_hat = Xw @ beta
    r2 = reconstruction_skill(N[test], N_hat[test])["r2"]
    ax.plot(year, N, "k", label="N true")
    ax.plot(year, N_hat, "tab:red", lw=1, label="N reconstructed")
    ax.axvspan(year[test][0], year[test][-1], color="orange", alpha=0.15)
    ax.set_title(f"{name}: held-out r² = {r2:.2f}"); ax.legend(fontsize=7)
fig.suptitle(f"{WALKTHROUGH_MODEL}: reconstructing N on held-out segments");

# %% [markdown]
# ## Production fit and the GF maps
#
# Refit on the full record with the CV-selected α, then scatter β back onto the map to
# see the Green's function. Negative (blue) means warming *there* loses energy to space —
# concentrated over the tropical west Pacific, the fingerprint behind the pattern effect.

# %%
def gf_map(g, pts, mask):
    """Scatter a per-ocean-point GF vector back onto the (lat, lon) grid (NaN over land)."""
    m = mask.stack(pt=("lat", "lon"))
    full = xr.full_like(m, np.nan, dtype=float)
    full.values[m.values] = g
    return full.unstack("pt").rename(None)


beta_N = dual_ridge(Xw, N, select_alpha(Xw, N))
G_toa = gf_map(w * beta_N, pts, mask)          # physical units (applies to unscaled SST)
fig = plt.figure(figsize=(8, 4))
plotting.map_plot(G_toa, robust=True, cbar_label="∂N/∂T(x)  [W m⁻² K⁻¹ per cell]",
                  title=f"{WALKTHROUGH_MODEL}  $G_{{toa}}$");

# %% [markdown]
# ## Batch: fit and save every model
#
# The steps above, packaged into `fit_gf(model)` — fit both GFs on the full record, save
# the maps and the reconstructed N/T series, and return the held-out skill rows. Then loop
# every model and write the skill table.

# %%
def fit_gf(model):
    """Fit both GFs for one model; save GF maps + amip N/T series; return skill rows."""
    # targets: global-mean net TOA (N) and global-mean temperature (T), full globe
    targets = {"toa": global_mean(anomaly(load_amip("toa", model))).values,
               "tas": global_mean(anomaly(load_amip("tas", model))).values}
    # predictor: ocean ts anomalies within ±PREDICTOR_LAT_BOUND (the SST pattern)
    ts_a = anomaly(load_amip("ts", model))
    m = predictor_mask(model)
    p = ocean_points(m)
    Xm = ocean_sst_matrix(ts_a, p)
    wm = area_weights(Xm)
    Xwm = Xm.values * wm

    skill = []
    for target, y in targets.items():                          # held-out validation
        for name, test in holdout_segments(len(y)).items():
            train = np.setdiff1d(np.arange(len(y)), test)
            alpha = select_alpha(Xwm[train], y[train])
            beta = dual_ridge(Xwm[train], y[train], alpha)
            skill.append({"model": model, "target": target, "segment": name,
                          "alpha": alpha, **reconstruction_skill(y[test], Xwm[test] @ beta)})

    gf_maps, series = {}, {}                                    # production fit on full record
    for target, y in targets.items():
        alpha = select_alpha(Xwm, y)
        beta = dual_ridge(Xwm, y, alpha)
        gf_maps[f"G_{target}"] = gf_map(wm * beta, p, m)
        gf_maps[f"G_{target}"].attrs = {"alpha": alpha}
        series[f"{target}_true"] = ("year", y)
        series[f"{target}_hat"] = ("year", Xwm @ beta)

    (DERIVED_DIR / "gf").mkdir(parents=True, exist_ok=True)
    xr.Dataset(gf_maps, attrs={"model": model}).to_netcdf(DERIVED_DIR / "gf" / f"GF_{model}.nc")
    (DERIVED_DIR / "series").mkdir(parents=True, exist_ok=True)
    xr.Dataset(series, coords={"year": Xm.year.values}).to_netcdf(
        DERIVED_DIR / "series" / f"amip_{model}.nc")
    return skill


all_skill = []
for model in analysis_models():
    all_skill += fit_gf(model)
    print(f"fit {model}")

skill_table = pd.DataFrame(all_skill)
skill_table.to_csv(DERIVED_DIR / "gf" / "holdout_skill.csv", index=False)
print("\nmean held-out r² by target/segment:")
print(skill_table.groupby(["target", "segment"]).r2.mean().round(2).to_string())

# %% [markdown]
# ## Figures: all GF maps and the held-out skill
#
# The tropical-Pacific fingerprint across models, and the honest skill numbers: T is
# reconstructed well from the SST pattern; N (noisier, cloud-dominated) is harder.

# %%
import cartopy.crs as ccrs
models = analysis_models()
ncols = 4
nrows = int(np.ceil(len(models) / ncols))          # grid sized to the model count
fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
for i, model in enumerate(models):
    g = xr.open_dataset(DERIVED_DIR / "gf" / f"GF_{model}.nc").G_toa
    ax = plt.subplot(nrows, ncols, i + 1, projection=ccrs.Robinson(central_longitude=180))
    plotting.map_plot(g, ax=ax, robust=True, cbar=False, title=model)
fig.suptitle("$G_{toa}$ Green's function per model (each scaled to its own P98)")
config.FIGURES_DIR.mkdir(exist_ok=True)
fig.savefig(config.FIGURES_DIR / "gf_toa_maps.png", dpi=110, bbox_inches="tight");

# %%
pivot = skill_table.pivot_table(index="segment", columns="target", values="r2", aggfunc="mean")
print(pivot.round(2))

# %%
