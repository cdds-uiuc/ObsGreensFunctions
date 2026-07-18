# %% [markdown]
# # 03 · Radiative feedbacks: amip-piForcing vs historical
#
# **Why we need the GFs.** In a historical run the net TOA anomaly mixes forcing and
# response, **N′ = F′ + λ·T′**, and F′ is unknown — so you *cannot* just regress N′ on T′
# to get λ. The Green's functions fit in notebook 02 came from the zero-forcing
# amip-piForcing world, so a GF reconstruction of N′ is the **radiative response alone**.
# Applying the GFs to historical SST therefore gives a forcing-clean feedback.
#
# Three estimates per model:
# - **amip_true** — the model's own N′, T′ in amip-piForcing (the truth; valid because F′=0)
# - **amip_gf**   — GF reconstruction of the same run (the method vs truth)
# - **hist_gf**   — GFs applied to historical tos, one value per ensemble member
#
# Two feedback **definitions**: a 30-yr sliding-window slope, and a cumulative ratio
# N′(t)/T′(t). We walk through CanESM5, then batch over all models.

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from collections import namedtuple

from obsgf import config, plotting
from obsgf.config import ANALYSIS_YEARS, BASELINE_YEARS, DERIVED_DIR, FIGURES_DIR, analysis_models, representative_member
from obsgf.regrid import regrid_model

# --- knobs for this notebook (edit freely) ---
WINDOW_LENGTH = 30        # yr, sliding-window feedback regression
RATIO_START_YEAR = 1940   # cumulative ratio starts here (so T' is safely positive)
RATIO_SMOOTH = 5          # yr, centered running mean before the ratio
RATIO_TMIN = 0.2          # K, ratio undefined where smoothed T' <= this (denominator guard)
WALKTHROUGH_MODEL = "CanESM5"

# %% [markdown]
# ## The two feedback definitions
#
# **Window:** the OLS slope of N′ on T′ in each 30-yr window (a local, decadal feedback).
# **Ratio:** cumulative N′(t)/T′(t) vs the 1870–1919 baseline. The ratio blows up when T′
# nears zero, so we smooth first and only define it where smoothed T′ > 0.2 K. The next
# cell shows why that guard is needed.

# %%
def _ols_slope(x, y):
    xc = x - x.mean()
    denom = np.sum(xc * xc)
    return np.sum(xc * (y - y.mean())) / denom if denom > 0 else np.nan


def _smooth(a, w):
    """Centered running mean of width w, NaN-padded at the edges to keep length."""
    if w <= 1:
        return np.asarray(a, dtype=float)
    valid = np.convolve(a, np.ones(w) / w, mode="valid")
    out = np.full(len(a), np.nan)
    out[w // 2: w // 2 + len(valid)] = valid
    return out


def window_centres(years, win=WINDOW_LENGTH):
    return np.array([float(np.mean(years[i:i + win])) for i in range(len(years) - win + 1)])


def window_slopes(T, N, years, win=WINDOW_LENGTH):
    return np.array([_ols_slope(T[i:i + win], N[i:i + win]) for i in range(len(years) - win + 1)])


def cumulative_ratio(T, N, years, start=RATIO_START_YEAR, smooth=RATIO_SMOOTH, tmin=RATIO_TMIN):
    """N'(t)/T'(t) from `start` onward; NaN where smoothed T' <= tmin (denominator guard)."""
    Ts = _smooth(T, smooth)
    lam = np.where(Ts > tmin, _smooth(N, smooth) / Ts, np.nan)
    return lam[years >= start]

# %% [markdown]
# ## Assemble the three estimates' N′, T′ series

# %%
def anomaly(da):
    return da - da.sel(year=slice(*BASELINE_YEARS)).mean("year")


def reconstruct(G, anom):
    """N'(t) = sum_x G(x) anom(x, t); land & missing-SST cells (NaN) skipped."""
    return (G * anom).sum(("lat", "lon"))


def amip_series(model):
    """{estimate: (years, T', N')} for amip_true and amip_gf, from notebook 02's output."""
    a = xr.open_dataset(DERIVED_DIR / "series" / f"amip_{model}.nc")
    yr = a.year.values
    return {"amip_true": (yr, a.tas_true.values, a.toa_true.values),
            "amip_gf": (yr, a.tas_hat.values, a.toa_hat.values)}


def hist_series(gf, tos):
    """(years, T', N') for one historical member's regridded tos through the model GF."""
    tos_a = anomaly(tos.sel(year=slice(*ANALYSIS_YEARS)))
    return (tos_a.year.values, reconstruct(gf.G_tas, tos_a).values, reconstruct(gf.G_toa, tos_a).values)


gf = xr.open_dataset(DERIVED_DIR / "gf" / f"GF_{WALKTHROUGH_MODEL}.nc")
tos_by_member = regrid_model(WALKTHROUGH_MODEL)
rep_member = representative_member(WALKTHROUGH_MODEL)
yr_hist, T_hist, N_hist = hist_series(gf, tos_by_member[rep_member])

# guard demo: the raw ratio without the T'>0.2 guard blows up when T' crosses zero
raw_ratio = _smooth(N_hist, RATIO_SMOOTH) / _smooth(T_hist, RATIO_SMOOTH)
plt.figure(figsize=(9, 3))
plt.plot(yr_hist, raw_ratio, "0.6", label="no guard (blows up)")
plt.plot(yr_hist[yr_hist >= RATIO_START_YEAR],
         cumulative_ratio(T_hist, N_hist, yr_hist), "tab:blue", lw=2, label="guarded ratio")
plt.ylim(-6, 4); plt.axhline(0, color="k", lw=0.4); plt.legend()
plt.title(f"{WALKTHROUGH_MODEL} {rep_member}: why the ratio needs a denominator guard");

# %% [markdown]
# ## Walkthrough: amip feedback, truth vs GF
#
# Where the GF reconstruction (red) tracks the truth (black), the method works — including
# the late-century trend to more-negative λ, the pattern effect.

# %%
year, T_true, N_true = amip_series(WALKTHROUGH_MODEL)["amip_true"]
_, T_gf, N_gf = amip_series(WALKTHROUGH_MODEL)["amip_gf"]
centres = window_centres(year)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.4))
a1.plot(centres, window_slopes(T_true, N_true, year), "k", label="amip true")
a1.plot(centres, window_slopes(T_gf, N_gf, year), "tab:red", label="amip GF")
a1.set_title(f"{WALKTHROUGH_MODEL}: 30-yr window λ"); a1.set_xlabel("window centre"); a1.legend(fontsize=8)
ry = year[year >= RATIO_START_YEAR]
a2.plot(ry, cumulative_ratio(T_true, N_true, year), "k", label="amip true")
a2.plot(ry, cumulative_ratio(T_gf, N_gf, year), "tab:red", label="amip GF")
a2.set_title("cumulative λ = N′/T′"); a2.set_xlabel("year"); a2.legend(fontsize=8)
for a in (a1, a2): a.axhline(0, color="0.7", lw=0.5); a.set_ylabel("λ [W m⁻² K⁻¹]");

# %% [markdown]
# ## Walkthrough: the historical ensemble for this model
#
# Apply the GFs to every historical member's SST and overlay the spread (blue) on the
# amip trajectory (black). The blue band is the coupled model's natural variability of λ.

# %%
hist_windows = []
for tos in tos_by_member.values():
    yr_h, T_h, N_h = hist_series(gf, tos)
    hist_windows.append(window_slopes(T_h, N_h, yr_h))
hist_windows = np.array(hist_windows)

fig, ax = plt.subplots(figsize=(9, 3.4))
plotting.ensemble_band(ax, centres, hist_windows, "tab:blue", f"hist GF (n={len(tos_by_member)})")
ax.plot(centres, window_slopes(T_true, N_true, year), "k", lw=2, label="amip true")
ax.axhline(0, color="0.7", lw=0.5); ax.legend(fontsize=8)
ax.set_title(f"{WALKTHROUGH_MODEL}: amip vs historical ensemble"); ax.set_ylabel("λ [W m⁻² K⁻¹]");

# %% [markdown]
# ## Batch: both definitions, three estimates, every model
#
# For each model we turn every estimate's (T′, N′) into a `Feedback` (window + ratio).
# amip has one series each; historical is a per-member ensemble. Two datasets come out:
# `feedbacks.nc` (representative member) and `feedbacks_hist_ensemble.nc` (all members).

# %%
ESTIMATES = ["amip_true", "amip_gf", "hist_gf"]
Feedback = namedtuple("Feedback", ["window", "ratio"])
Run = namedtuple("Run", ["model", "member", "feedback"])


def feedback_of(years, T, N):
    return Feedback(window_slopes(T, N, years), cumulative_ratio(T, N, years))


models = analysis_models()
shared_years = amip_series(models[0])["amip_true"][0]
center_year = window_centres(shared_years)
ratio_year = shared_years[shared_years >= RATIO_START_YEAR]

rep, ensemble = {}, []
for model in models:
    gfm = xr.open_dataset(DERIVED_DIR / "gf" / f"GF_{model}.nc")
    for estimate, (yr, T, N) in amip_series(model).items():
        rep[(model, estimate)] = feedback_of(yr, T, N)
    rep_m = representative_member(model)
    for member, tos in regrid_model(model).items():
        fb = feedback_of(*hist_series(gfm, tos))
        ensemble.append(Run(model, member, fb))
        if member == rep_m:
            rep[(model, "hist_gf")] = fb
    print(f"{model:16s} {sum(r.model == model for r in ensemble):3d} members")

# %%
# representative-member dataset: (model, estimate, time)
window = np.array([[rep[(m, e)].window for e in ESTIMATES] for m in models])
ratio = np.array([[rep[(m, e)].ratio for e in ESTIMATES] for m in models])
feedbacks = xr.Dataset(
    {"feedback_window": (("model", "estimate", "center_year"), window),
     "feedback_ratio": (("model", "estimate", "year"), ratio)},
    coords={"model": models, "estimate": ESTIMATES, "center_year": center_year, "year": ratio_year})
feedbacks.to_netcdf(DERIVED_DIR / "feedbacks.nc")

# ensemble dataset: run-indexed over every historical member
ens = xr.Dataset(
    {"feedback_window": (("run", "center_year"), np.array([r.feedback.window for r in ensemble])),
     "feedback_ratio": (("run", "year"), np.array([r.feedback.ratio for r in ensemble]))},
    coords={"run": np.arange(len(ensemble)),
            "model": ("run", [r.model for r in ensemble]),
            "member": ("run", [r.member for r in ensemble]),
            "center_year": center_year, "year": ratio_year})
ens.to_netcdf(DERIVED_DIR / "feedbacks_hist_ensemble.nc")
print(f"saved feedbacks.nc and feedbacks_hist_ensemble.nc ({len(ensemble)} runs)")

# %% [markdown]
# ## Figures: the deliverable, per model
#
# amip_true (black) & amip_gf (red) against the CMIP6-historical ensemble band (blue).
# The late-century amip swing to more-negative λ reaches the edge of the coupled spread.

# %%
def hist_of(var, m):
    return ens[var].values[ens.model.values.astype(str) == m]


def multipanel(var, x, xlabel, title, fname, ylim=None):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, models):
        plotting.ensemble_band(ax, x, hist_of(var, m), "tab:blue", f"CMIP6 hist (n={hist_of(var, m).shape[0]})")
        for e, c in [("amip_true", "k"), ("amip_gf", "tab:red")]:
            ax.plot(x, feedbacks[var].sel(model=m, estimate=e), color=c, lw=1.8, label=e.replace("_", " "))
        ax.set_title(m); ax.axhline(0, color="0.7", lw=0.5); ax.legend(fontsize=6.5)
    ax = axes.flat[7]
    plotting.ensemble_band(ax, x, ens[var].values, "tab:blue", f"CMIP6 hist (all {ens.sizes['run']})")
    for e, c in [("amip_true", "k"), ("amip_gf", "tab:red")]:
        ax.plot(x, feedbacks[var].sel(estimate=e).mean("model"), color=c, lw=2, label=e.replace("_", " "))
    ax.set_title("all models pooled"); ax.axhline(0, color="0.7", lw=0.5); ax.legend(fontsize=7)
    if ylim: ax.set_ylim(*ylim)
    for a in axes[:, 0]: a.set_ylabel("λ [W m⁻² K⁻¹]")
    for a in axes[1, :]: a.set_xlabel(xlabel)
    fig.suptitle(title); fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fname, dpi=110)


multipanel("feedback_window", center_year, "window centre year",
           "30-yr window radiative feedback: amip-piForcing vs CMIP6-historical ensemble", "feedbacks.png")
multipanel("feedback_ratio", ratio_year, "year",
           "Cumulative feedback λ = N′/T′: amip-piForcing vs CMIP6-historical ensemble",
           "feedbacks_ratio.png", ylim=(-4, 1))

# %% [markdown]
# ## Summary: early vs late, both definitions

# %%
import pandas as pd
rows = []
for name, da, axis in [("window", feedbacks.feedback_window, center_year),
                       ("ratio", feedbacks.feedback_ratio, ratio_year)]:
    arr = da.values
    fin = np.where(np.isfinite(arr).all(axis=(0, 1)))[0]
    for j, e in enumerate(ESTIMATES):
        rows.append({"definition": name, "estimate": e,
                     f"early~{axis[fin[0]]:.0f}": round(float(np.nanmean(arr[:, j, fin[0]])), 2),
                     f"late~{axis[fin[-1]]:.0f}": round(float(np.nanmean(arr[:, j, fin[-1]])), 2)})
pd.DataFrame(rows)
