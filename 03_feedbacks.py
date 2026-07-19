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
# The feedback is a **sliding-window** estimate, computed two ways (below). We show the
# full time series only for the direct slope, and compare both methods over the single last
# window (ending 2014), where each is stable. `WINDOW_LENGTH` carries into every filename.

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.stats import linregress, gaussian_kde

from obsgf import config
from obsgf.config import ANALYSIS_YEARS, BASELINE_YEARS, DERIVED_DIR, FIGURES_DIR, MODELS, representative_member
from obsgf.regrid import regrid_model

# --- knobs for this notebook (edit freely) ---
WINDOW_LENGTH = 35        # yr, sliding-window length (carried into output filenames)
WALKTHROUGH_MODEL = "CanESM5"


# %%
# A small plotting helper used by the figures below: a member ensemble as a shaded
# 5-95% band + mean line (a single member falls back to a plain line).
def ensemble_band(ax, x, arr, color, label, pct=(5, 95)):
    """Draw a member ensemble as a shaded pct band + mean line; a single member is a plain line."""
    arr = np.atleast_2d(arr)
    if arr.shape[0] == 1:
        ax.plot(x, arr[0], color=color, lw=1.5, label=label)
        return
    lo, hi = np.nanpercentile(arr, pct[0], 0), np.nanpercentile(arr, pct[1], 0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.plot(x, np.nanmean(arr, 0), color=color, lw=2, label=label)

# %% [markdown]
# ## Two ways to get λ within a window
#
# Both slide a `WINDOW_LENGTH`-year window along the record and return one λ per window,
# plotted against the window's **end year**.
#
# **(1) slope:** the OLS slope of N′ on T′ inside the window (with intercept — the window
# means of N′, T′ are nonzero, being anomalies vs pre-industrial). The local feedback, but
# internal covariance (ENSO) between N′ and T′ enters it.
#
# **(2) trend ratio:** regress N′ and T′ separately on *time* within the window and take the
# ratio of the two trends. The linear-in-time fit is a low-pass filter, so high-frequency
# internal covariance does not alias into λ.

# %%
def window_end_years(years, win=WINDOW_LENGTH):
    """End year of each length-`win` sliding window — the x-axis for the windowed feedbacks."""
    return np.asarray(years[win - 1:], dtype=float)


def window_slopes(T, N, years, win=WINDOW_LENGTH):
    """Feedback λ per window (method 1): OLS slope of N' on T' over `win` years."""
    return np.array([linregress(T[i:i + win], N[i:i + win]).slope
                     for i in range(len(years) - win + 1)])


def window_trend_ratio(T, N, years, win=WINDOW_LENGTH):
    """Feedback λ per window (method 2): ratio of the time-trends of N' and T' (each
    regressed on year within the window). Low-pass — internal covariance doesn't alias in."""
    return np.array([linregress(years[i:i + win], N[i:i + win]).slope
                     / linregress(years[i:i + win], T[i:i + win]).slope
                     for i in range(len(years) - win + 1)])

# %% [markdown]
# ## Assemble the three estimates' N′, T′ series

# %%
def anomaly(da):
    """Anomaly relative to the 1870–1919 baseline climatology."""
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

# %% [markdown]
# ## Walkthrough: amip feedback (slope method), truth vs GF, vs the historical ensemble
#
# The slope-method λ time series for CanESM5: the GF reconstruction (red) tracks the truth
# (black) — including the late-century trend to more-negative λ, the pattern effect — and the
# historical ensemble band (blue) is the coupled model's natural variability of λ.

# %%
year, T_true, N_true = amip_series(WALKTHROUGH_MODEL)["amip_true"]
_, T_gf, N_gf = amip_series(WALKTHROUGH_MODEL)["amip_gf"]
ends = window_end_years(year)

gf = xr.open_dataset(DERIVED_DIR / "gf" / f"GF_{WALKTHROUGH_MODEL}.nc")
tos_by_member = regrid_model(WALKTHROUGH_MODEL)
hist_slopes = []
for tos in tos_by_member.values():
    yr_h, T_h, N_h = hist_series(gf, tos)
    hist_slopes.append(window_slopes(T_h, N_h, yr_h))

fig, ax = plt.subplots(figsize=(9, 3.4))
ensemble_band(ax, ends, np.array(hist_slopes), "tab:blue", f"hist GF (n={len(tos_by_member)})")
ax.plot(ends, window_slopes(T_true, N_true, year), "k", lw=2, label="amip true")
ax.plot(ends, window_slopes(T_gf, N_gf, year), "tab:red", lw=1.5, label="amip GF")
ax.axhline(0, color="0.7", lw=0.5); ax.set_ylim(-4, 1.5); ax.legend(fontsize=8)
ax.set_title(f"{WALKTHROUGH_MODEL}: window λ (slope of N′ on T′)")
ax.set_xlabel(f"{WINDOW_LENGTH}-yr window end year"); ax.set_ylabel("λ [W m⁻² K⁻¹]");

# %% [markdown]
# ## Batch: both methods, three estimates, every model
#
# For each model we turn every estimate's (T′, N′) into a `Feedback` (both window methods).
# amip has one series each; historical is a per-member ensemble. Two datasets come out, the
# window length in their name: `feedbacks_win{W}.nc` (representative member) and
# `feedbacks_hist_ensemble_win{W}.nc` (all members).

# %%
ESTIMATES = ["amip_true", "amip_gf", "hist_gf"]
Feedback = namedtuple("Feedback", ["slope", "trend_ratio"])
Run = namedtuple("Run", ["model", "member", "feedback"])


def feedback_of(years, T, N):
    """Both windowed feedback estimates for one (T', N') series: slope, and trend ratio."""
    return Feedback(window_slopes(T, N, years), window_trend_ratio(T, N, years))


models = MODELS
shared_years = amip_series(models[0])["amip_true"][0]
end_year = window_end_years(shared_years)

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
# representative-member dataset: (model, estimate, end_year)
slope = np.array([[rep[(m, e)].slope for e in ESTIMATES] for m in models])
tratio = np.array([[rep[(m, e)].trend_ratio for e in ESTIMATES] for m in models])
feedbacks = xr.Dataset(
    {"feedback_slope": (("model", "estimate", "end_year"), slope),
     "feedback_trend_ratio": (("model", "estimate", "end_year"), tratio)},
    coords={"model": models, "estimate": ESTIMATES, "end_year": end_year})
feedbacks.to_netcdf(DERIVED_DIR / f"feedbacks_win{WINDOW_LENGTH}.nc")

# ensemble dataset: run-indexed over every historical member
ens = xr.Dataset(
    {"feedback_slope": (("run", "end_year"), np.array([r.feedback.slope for r in ensemble])),
     "feedback_trend_ratio": (("run", "end_year"), np.array([r.feedback.trend_ratio for r in ensemble]))},
    coords={"run": np.arange(len(ensemble)),
            "model": ("run", [r.model for r in ensemble]),
            "member": ("run", [r.member for r in ensemble]),
            "end_year": end_year})
ens.to_netcdf(DERIVED_DIR / f"feedbacks_hist_ensemble_win{WINDOW_LENGTH}.nc")
print(f"saved feedbacks_win{WINDOW_LENGTH}.nc and feedbacks_hist_ensemble_win{WINDOW_LENGTH}.nc ({len(ensemble)} runs)")

# %% [markdown]
# ## Figure: slope-method feedback time series, per model
#
# amip_true (black) & amip_gf (red) against the CMIP6-historical ensemble band (blue), per
# model. Only the direct slope of N′ on T′ gets a time series (the trend ratio is unstable
# in early windows); both methods are compared over the last window in the next figure.

# %%
def hist_of(var, m):
    """The historical-ensemble rows of `var` belonging to model `m`."""
    return ens[var].values[ens.model.values.astype(str) == m]


def multipanel(var, x, xlabel, title, fname, ylim=None):
    """Grid of per-model panels: amip true/gf lines over the historical band; saves `fname`."""
    ncols = 4
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), sharey=True)
    axf = axes.flat
    for ax, name in zip(axf, models):
        band = hist_of(var, name)
        true_ = feedbacks[var].sel(model=name, estimate="amip_true")
        gf_ = feedbacks[var].sel(model=name, estimate="amip_gf")
        ensemble_band(ax, x, band, "tab:blue", f"CMIP6 hist (n={np.atleast_2d(band).shape[0]})")
        ax.plot(x, true_, "k", lw=1.8, label="amip true")
        ax.plot(x, gf_, "tab:red", lw=1.8, label="amip gf")
        ax.set_title(name); ax.axhline(0, color="0.7", lw=0.5); ax.legend(fontsize=6.5); ax.set_xlabel(xlabel)
    if ylim:
        axf[0].set_ylim(*ylim)                 # sharey -> applies to all panels
    for ax in axf[len(models):]:               # hide any unused panels
        ax.axis("off")
    for r in range(nrows):
        axf[r * ncols].set_ylabel("λ [W m⁻² K⁻¹]")
    fig.suptitle(title); fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fname, dpi=110)


multipanel("feedback_slope", end_year, f"{WINDOW_LENGTH}-yr window end year",
           f"Windowed feedback λ = slope of N′ on T′  ({WINDOW_LENGTH}-yr windows)",
           f"feedbacks_slope_win{WINDOW_LENGTH}.png", ylim=(-4, 1.5))

# %% [markdown]
# ## Figures: last-window feedback distribution, per model, one figure per method
#
# λ over the final window (ending 2014) — the recent, strong-pattern-effect interval where
# both methods are stable. One panel per model: a KDE of that model's historical members
# (each member a thin line), with its amip-piForcing truth (black) and GF reconstruction
# (red) as thick vertical lines. One figure per method.

# %%
def lastwindow_panels(var, title, fname, xlim=(-5, 0)):
    """Per-model grid of the last-window feedback: a KDE of the model's historical members
    with each member as a thin line, and the model's amip_true (black) and amip_gf (red) as
    thick vertical lines. Saves `fname`."""
    ncols = 4
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows))
    grid = np.linspace(*xlim, 300)
    axf = axes.flat
    for ax, name in zip(axf, models):
        memb = hist_of(var, name)[:, -1]
        at = float(feedbacks[var].sel(model=name, estimate="amip_true").values[-1])
        ag = float(feedbacks[var].sel(model=name, estimate="amip_gf").values[-1])
        for v in memb:                                               # each member: a thin line
            ax.axvline(v, color="tab:blue", lw=0.5, alpha=0.25)
        if len(memb) > 1 and np.ptp(memb) > 0:                       # KDE needs >1 point with spread
            pdf = gaussian_kde(memb)(grid)
            ax.fill_between(grid, pdf, color="tab:blue", alpha=0.15)
            ax.plot(grid, pdf, color="tab:blue", lw=1.8, label=f"hist GF (n={len(memb)})")
        ax.axvline(at, color="k", lw=2.8, label="amip true")
        ax.axvline(ag, color="tab:red", lw=2.4, ls="--", label="amip gf")
        ax.set_title(f"{name} (n={len(memb)})"); ax.set_xlim(*xlim); ax.set_ylim(bottom=0)
        ax.set_xlabel("λ [W m⁻² K⁻¹]"); ax.legend(fontsize=6.5)
    for r in range(nrows):
        axf[r * ncols].set_ylabel("density")
    for ax in axf[len(models):]:
        ax.axis("off")
    fig.suptitle(title); fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / fname, dpi=110)


lw_end = int(end_year[-1])
lastwindow_panels("feedback_slope",
                  f"Last-window λ = slope of N′ on T′  ({WINDOW_LENGTH}-yr window ending {lw_end})",
                  f"feedbacks_lastwindow_slope_win{WINDOW_LENGTH}.png")
lastwindow_panels("feedback_trend_ratio",
                  f"Last-window λ = ratio of N′, T′ trends  ({WINDOW_LENGTH}-yr window ending {lw_end})",
                  f"feedbacks_lastwindow_trendratio_win{WINDOW_LENGTH}.png")
