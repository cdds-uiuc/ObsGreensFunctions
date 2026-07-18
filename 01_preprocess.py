# %% [markdown]
# # 01 · Pre-processing: CMIP6 catalog → clean annual-mean files
#
# **Goal.** Turn the raw CMIP6 pool into tidy per-model netCDFs we can analyse without
# ever touching the catalog again. For each (variable, experiment, model, member) we
#
# 1. load the harmonized *monthly* data on its native grid,
# 2. build `toa = rsdt − rsut − rlut` (net **down**ward top-of-atmosphere flux),
# 3. take a **month-length-weighted annual mean**,
# 4. save one file per member to `pre-processed_data/`.
#
# Variables: `tas` (2-m air temp), `toa`, `ts` (surface temp — over open ocean this *is*
# the prescribed SST, our Green's-function predictor), and historical `tos`. We also
# extract each model's static land-fraction field `sftlf` (for ocean masks later).
#
# **Why this is its own stage:** it is the *only* place that opens the CMIP6 catalog.
# Everything downstream (masks, Green's functions, feedbacks) reads `pre-processed_data/`
# only. The catalog plumbing lives in `obsgf/catalog.py` — you don't need to read it.
#
# We first walk through **one member** so every intermediate is visible to inspect and
# plot, then run the batch loop over everything.

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from collections import Counter

from obsgf import config
from obsgf.catalog import (open_catalog, list_members, load_member, raw_vars_for,
                           out_name, existing_output)

# --- knobs for this notebook (edit freely) ---
FORCE = False                    # rebuild files that already exist?
ONLY_MODELS = None               # e.g. ["CanESM5"] to build just one model; None = all
WALKTHROUGH_MODEL = "CanESM5"    # coarse grid → fast, legible

cat = open_catalog()

# %% [markdown]
# ## Walkthrough: load one member's monthly data
#
# We take CanESM5's amip-piForcing run and load monthly `tas` (near-surface air
# temperature). `load_member` returns it harmonized and on the native grid.

# %%
members, grid = list_members(cat, WALKTHROUGH_MODEL, "amip-piForcing", ["tas"])
member = members[0]
ds_month = load_member(cat, WALKTHROUGH_MODEL, "amip-piForcing", ["tas"], member, grid)
tas_month = ds_month["tas"]
print(member, grid, "| dims:", dict(tas_month.sizes))
tas_month.isel(time=0).sortby(["lat", "lon"]).plot(figsize=(8, 3));   # one month, sanity look

# %% [markdown]
# ## Annual means are **month-length weighted**
#
# A plain `groupby("time.year").mean()` weights February the same as July. Weighting by
# `days_in_month` is the correct annual mean. The function below does that; the next cell
# shows how much it matters.

# %%
def annual_mean(da):
    """Month-length-weighted annual mean; keeps only whole (12-month) years."""
    months_per_year = da.time.groupby("time.year").count()
    full_years = months_per_year.year.where(months_per_year == 12, drop=True)
    w = da.time.dt.days_in_month
    num = (da * w).groupby("time.year").sum("time", skipna=True)
    den = xr.where(da.notnull(), w, 0).groupby("time.year").sum("time")
    ann = (num / den).where(den > 0)
    return ann.sel(year=full_years)


tas_annual = annual_mean(tas_month)
tas_naive = tas_month.groupby("time.year").mean("time").sel(year=tas_annual.year)
diff = (tas_annual - tas_naive)
print("weighted − naive global-mean annual tas (K):",
      float(diff.weighted(np.cos(np.deg2rad(diff.lat))).mean().max()))   # small but nonzero

# %% [markdown]
# ## Net TOA flux: `toa = rsdt − rsut − rlut`
#
# Incoming solar minus reflected solar minus outgoing longwave = net **down** flux.
# Watch the global mean: it must be **area-weighted** (`cos lat`). An unweighted mean is
# dominated by the poles and comes out badly wrong.

# %%
flux = load_member(cat, WALKTHROUGH_MODEL, "amip-piForcing",
                   ["rsdt", "rsut", "rlut"], member, grid)
toa_month = flux["rsdt"] - flux["rsut"] - flux["rlut"]
toa_annual = annual_mean(toa_month)

toa_mean_field = toa_annual.mean("year")
print("global-mean net TOA, UNweighted: %6.1f W/m²  (wrong — pole-biased)"
      % float(toa_mean_field.mean()))
print("global-mean net TOA, cos-lat   : %6.2f W/m²  (right — small positive imbalance)"
      % float(toa_mean_field.weighted(np.cos(np.deg2rad(toa_annual.lat))).mean()))

# %% [markdown]
# ## Land fraction `sftlf` → ocean mask preview
#
# `sftlf` is time-invariant (the `fx` table). We save it per model so the mask stage
# never needs the catalog. Ocean = land fraction below `OCEAN_SFTLF_MAX`.

# %%
sftlf_members, sftlf_grid = list_members(cat, WALKTHROUGH_MODEL, config.SFTLF_SOURCE, ["sftlf"])
sftlf = load_member(cat, WALKTHROUGH_MODEL, config.SFTLF_SOURCE, ["sftlf"],
                    sftlf_members[0], sftlf_grid)["sftlf"].reset_coords(drop=True)
sftlf = sftlf.sortby(["lat", "lon"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3))
sftlf.plot(ax=a1); a1.set_title("sftlf (% land)")
(sftlf < config.OCEAN_SFTLF_MAX).plot(ax=a2); a2.set_title(f"ocean (sftlf < {config.OCEAN_SFTLF_MAX})");

# %% [markdown]
# ## Sanity checks
#
# A clean run *is* the verification, so each written file must pass physical checks:
# no NaNs where there shouldn't be, global means in a plausible range.

# %%
def sanity_check(da, var, label):
    years = da.year.values
    assert (years[1:] - years[:-1] == 1).all(), f"{label}: gap in years"
    gmean = float(da.weighted(np.cos(np.deg2rad(da.lat))).mean())
    if var in ("tas", "toa", "ts"):
        assert not da.isnull().any(), f"{label}: unexpected NaNs in {var}"
    if var in ("tas", "ts"):                              # both surface temperatures (K)
        assert 270 < gmean < 300, f"{label}: global {var} {gmean:.1f} K implausible"
    if var == "toa":
        assert abs(gmean) < 5, f"{label}: global toa {gmean:.2f} W/m² implausible"
    if var == "tos":
        assert da.isnull().any(), f"{label}: tos has no land mask?"
        assert -5 < gmean < 35, f"{label}: tos mean {gmean:.1f} °C implausible"


sanity_check(tas_annual, "tas", "walkthrough tas")
sanity_check(toa_annual, "toa", "walkthrough toa")
print("walkthrough passes sanity checks")

# %% [markdown]
# ## Batch: build every file
#
# Now package the trusted steps into two builders and loop over everything in the spec.
# Existing files are skipped (so re-running the notebook is cheap); a bad member is
# reported but doesn't halt the run. **Cold run is ~30–60 min; warm run is seconds.**

# %%
def build_annual_member(var, experiment, model, member, grid, force):
    """load → toa/annual-mean → sanity-check → save one member file. Returns an outcome."""
    label = f"{var} {model} {experiment} {member}"
    if not force and existing_output(var, model, experiment, member, grid):
        return "skip"
    try:
        ds = load_member(cat, model, experiment, raw_vars_for(var), member, grid)
        da = (ds["rsdt"] - ds["rsut"] - ds["rlut"]) if var == "toa" else ds[var]
        ann = annual_mean(da).rename(var).compute()
        sanity_check(ann, var, label)
        years = (int(ann.year[0]), int(ann.year[-1]))
        out = config.PREPROCESSED_DIR / out_name(var, model, experiment, member, grid, years)
        ann.to_dataset().assign_attrs(note="annual means (month-length weighted), native grid",
                                      harmonize_version=ds.attrs.get("harmonize_version", "?")
                                      ).to_netcdf(out, engine="netcdf4")
        print(f"done {out.name}  ({years[0]}-{years[1]})")
        return "done"
    except Exception as e:
        print(f"FAIL {label}: {e}")
        return "fail"


def build_sftlf(model, force):
    """Save the static land-fraction field to pre-processed_data/sftlf_<model>.nc."""
    out = config.PREPROCESSED_DIR / f"sftlf_{model}.nc"
    if out.exists() and not force:
        return "skip"
    try:
        mem, g = list_members(cat, model, config.SFTLF_SOURCE, ["sftlf"])
        if not mem:
            print(f"MISS {model} sftlf: not in pool"); return "miss"
        s = load_member(cat, model, config.SFTLF_SOURCE, ["sftlf"], mem[0], g)["sftlf"]
        if "time" in s.dims:
            s = s.isel(time=0, drop=True)
        s = s.reset_coords(drop=True)
        assert -0.1 <= float(s.min()) and float(s.max()) <= 100.1, f"{model} sftlf out of [0,100]"
        assert bool((s < 50).any()) and bool((s >= 50).any()), f"{model} sftlf all land or all ocean"
        s.rename("sftlf").to_dataset().to_netcdf(out, engine="netcdf4")
        print(f"done sftlf_{model}.nc"); return "done"
    except Exception as e:
        print(f"FAIL sftlf {model}: {e}"); return "fail"


# %%
config.PREPROCESSED_DIR.mkdir(exist_ok=True)
outcomes = Counter()

# annual-mean fields (tas, toa, tos): one file per experiment / model / member
for experiment, variables in config.PREPROCESS_SPEC.items():
    for model in config.MODELS:
        if ONLY_MODELS and model not in ONLY_MODELS:
            continue
        for var in variables:
            members, grid = list_members(cat, model, experiment, raw_vars_for(var))
            if not members:
                print(f"MISS {var} {model} {experiment}: not in pool"); continue
            for mem in members:
                outcomes[build_annual_member(var, experiment, model, mem, grid, FORCE)] += 1

# static land fraction (sftlf): one file per model
for model in config.MODELS:
    if ONLY_MODELS and model not in ONLY_MODELS:
        continue
    outcomes[build_sftlf(model, FORCE)] += 1

print(f"\n{outcomes['done']} written, {outcomes['skip']} skipped, {outcomes['fail']} failed")
