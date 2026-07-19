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
import cartopy.crs as ccrs
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, root_mean_squared_error

from obsgf import config
from obsgf.config import (ANALYSIS_YEARS, BASELINE_YEARS, DERIVED_DIR,
                          MODELS, find_preprocessed)
from obsgf.masks import ocean_mask, global_mean

# --- knobs for this notebook (edit freely) ---
ALPHAS = np.logspace(0, 6, 25)   # ridge penalties to search over
N_CV_BLOCKS = 5                  # blocked cross-validation folds (contiguous in time)
HOLDOUT_LENGTH = 20              # length (yr) of each held-out validation segment
PREDICTOR_LAT_BOUND = 55         # keep SST predictors within ±this latitude (avoid sea ice)
WALKTHROUGH_MODEL = "CanESM5"

# %%
# A small map helper (Robinson projection, coastlines, optional 98th-pct scaling), used
# for the mask and GF maps below — the cartopy boilerplate kept out of the science cells.
def map_plot(da, ax=None, title=None, vmin=None, vmax=None, cmap="RdBu_r",
             robust=False, cbar=True, cbar_label=None):
    """Plot a lat/lon DataArray on a Robinson map with coastlines; robust=True scales to ±P98."""
    if ax is None:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
    if robust and vmin is None:
        v = float(np.nanpercentile(np.abs(da.values), 98))
        vmin, vmax = -v, v
    da = da.sortby(["lat", "lon"])          # guard against descending / wrapped coords
    p = da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                add_colorbar=cbar, cbar_kwargs={"label": cbar_label, "shrink": 0.7} if cbar else None)
    ax.coastlines(lw=0.4)
    if title:
        ax.set_title(title)
    return p

# %% [markdown]
# ## Global-means
#
# Load tas and toa, take anomalies vs 1870–1919, and form the two things the GFs predict:
# global-mean net TOA **N** and global-\mean temperature **T** (both area-weighted over the
# *full* globe — λ is defined per unit global surface-air temperature).

# %%
def anomaly(da):
    """Anomaly relative to the 1870–1919 baseline climatology."""
    return da - da.sel(year=slice(*BASELINE_YEARS)).mean("year")


def load_amip(var, model):
    """Load one amip-piForcing variable for a model, clipped to the common analysis window."""
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
    """The (lat, lon) index of a mask's True cells — a fixed column order for the design matrix."""
    m = mask.stack(pt=("lat", "lon"))
    return m.pt[m.values]           # the ocean-point index (fixes column order)


def ocean_sst_matrix(anom, pts):
    """Stack an anomaly field into a (year, ocean-point) matrix on `pts`; missing SST → 0."""
    return anom.stack(pt=("lat", "lon")).sel(pt=pts).fillna(0.0)   # (year, ocean-point)


def area_weights(X):
    """√cos(lat) column weights so the ridge penalty is area-fair over the sphere."""
    # cos clipped at 0: some regridded grids carry a latitude a hair beyond ±90
    return np.sqrt(np.clip(np.cos(np.deg2rad(X.lat.values)), 0, None))


ts_anom = anomaly(load_amip("ts", WALKTHROUGH_MODEL))   # SST proxy — the predictor field
mask = predictor_mask(WALKTHROUGH_MODEL)
pts = ocean_points(mask)
X = ocean_sst_matrix(ts_anom, pts)       # (year, ocean-point) DataArray
w = area_weights(X)                      # √cos(lat) column scaling
Xw = X.values * w                        # area-fair design matrix
print("design matrix:", Xw.shape, "= (years, ocean points within ±%d°)" % PREDICTOR_LAT_BOUND)
map_plot(mask.astype(float), title=f"{WALKTHROUGH_MODEL} predictor mask (ocean, ±{PREDICTOR_LAT_BOUND}°)",
                  cmap="Blues", cbar=False);

# %% [markdown]
# ## Ridge regression
#
# The GF is a linear map from the ~5000-point SST pattern to a scalar response. With far
# more points than years (145), plain least squares would overfit wildly, so we use **ridge
# regression** — least squares with an α‖β‖² penalty that shrinks the coefficients. We treat
# the estimator itself as trusted machinery: scikit-learn's `KernelRidge` with a linear
# kernel, which is ridge regression in the form that stays cheap when the predictors vastly
# outnumber the samples. Two modeling choices stay in view: it carries **no intercept** (zero
# SST anomaly must give zero response), and α is chosen by blocked cross-validation (next cell).

# %%
def fit_ridge(Xw, y):
    """CV-select the ridge penalty (blocked in time) and refit on all years; returns the fitted search."""
    search = GridSearchCV(KernelRidge(kernel="linear"),
                          {"alpha": ALPHAS}, cv=KFold(N_CV_BLOCKS, shuffle=False),
                          scoring="neg_mean_squared_error")
    return search.fit(Xw, y)

# %% [markdown]
# ## Choosing α by blocked cross-validation
#
# `KFold(shuffle=False)` holds out *contiguous* blocks of years — blocked because SST is
# autocorrelated, so shuffled folds would leak information from train into test. The α with
# the lowest cross-validated MSE wins. The curve below (read off the search's `cv_results_`)
# shows the trade-off; the star marks the choice for N.

# %%
search_N = fit_ridge(Xw, N)
alpha_N = search_N.best_params_["alpha"]
cv_mse_curve = -search_N.cv_results_["mean_test_score"]     # stored as neg-MSE → flip sign
plt.figure(figsize=(6, 3.2))
plt.loglog(ALPHAS, cv_mse_curve, "o-")
plt.loglog(alpha_N, cv_mse_curve.min(), "r*", ms=15)
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
    """Index arrays for the early / middle / late held-out validation segments of the record."""
    mid = n_years // 2 - length // 2
    return {"early": np.arange(0, length),
            "middle": np.arange(mid, mid + length),
            "late": np.arange(n_years - length, n_years)}


# validate N reconstruction on each held-out segment, and plot it
fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=True)
for ax, (name, test) in zip(axes, holdout_segments(len(N)).items()):
    train = np.setdiff1d(np.arange(len(N)), test)
    N_hat = fit_ridge(Xw[train], N[train]).predict(Xw)      # α re-selected within the train years
    r2 = r2_score(N[test], N_hat[test])
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


beta_N = Xw.T @ fit_ridge(Xw, N).best_estimator_.dual_coef_   # primal coefficients (linear kernel)
G_toa = gf_map(w * beta_N, pts, mask)          # physical units (applies to unscaled SST)
fig = plt.figure(figsize=(8, 4))
map_plot(G_toa, robust=True, cbar_label="∂N/∂T(x)  [W m⁻² K⁻¹ per cell]",
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
            search = fit_ridge(Xwm[train], y[train])
            pred = search.predict(Xwm[test])
            skill.append({"model": model, "target": target, "segment": name,
                          "alpha": search.best_params_["alpha"],
                          "r2": r2_score(y[test], pred),
                          "rmse": root_mean_squared_error(y[test], pred)})

    gf_maps, series = {}, {}                                    # production fit on full record
    for target, y in targets.items():
        search = fit_ridge(Xwm, y)
        beta = Xwm.T @ search.best_estimator_.dual_coef_       # primal coefficients (linear kernel)
        gf_maps[f"G_{target}"] = gf_map(wm * beta, p, m)
        gf_maps[f"G_{target}"].attrs = {"alpha": search.best_params_["alpha"]}
        series[f"{target}_true"] = ("year", y)
        series[f"{target}_hat"] = ("year", search.predict(Xwm))

    (DERIVED_DIR / "gf").mkdir(parents=True, exist_ok=True)
    xr.Dataset(gf_maps, attrs={"model": model}).to_netcdf(DERIVED_DIR / "gf" / f"GF_{model}.nc")
    (DERIVED_DIR / "series").mkdir(parents=True, exist_ok=True)
    xr.Dataset(series, coords={"year": Xm.year.values}).to_netcdf(
        DERIVED_DIR / "series" / f"amip_{model}.nc")
    return skill


all_skill = []
for model in MODELS:
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
models = MODELS
ncols = 4
nrows = int(np.ceil(len(models) / ncols))          # grid sized to the model count
fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
for i, model in enumerate(models):
    g = xr.open_dataset(DERIVED_DIR / "gf" / f"GF_{model}.nc").G_toa
    ax = plt.subplot(nrows, ncols, i + 1, projection=ccrs.Robinson(central_longitude=180))
    map_plot(g, ax=ax, robust=True, cbar=False, title=model)
fig.suptitle("$G_{toa}$ Green's function per model (each scaled to its own P98)")
config.FIGURES_DIR.mkdir(exist_ok=True)
fig.savefig(config.FIGURES_DIR / "gf_toa_maps.png", dpi=110, bbox_inches="tight");

# %%
pivot = skill_table.pivot_table(index="segment", columns="target", values="r2", aggfunc="mean")
print(pivot.round(2))
