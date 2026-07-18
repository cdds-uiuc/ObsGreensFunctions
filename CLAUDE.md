# CLAUDE.md

Guidance for working in this repository.

## What this project is

Observationally-derived **Green's functions (GFs)** for net radiative feedback.
GFs are trained on **amip-piForcing** simulations (constant pre-industrial forcing,
so net TOA = radiative response only) and used to reconstruct feedback λ from SST
patterns. See `PLAN.md` for the full design and `obsGF_design-doc.md` for the
original goal.

**Core idea / why it works:** in historical runs, net TOA is N′ = F′ + λ·T′ with the
forcing F′ unknown, so λ can't be regressed directly. In amip-piForcing F′ = 0, so
N′ = λ·T′. Train GFs there (∂N/∂T(x) and ∂T_global/∂T(x) from SST patterns), then
apply them to any SST field to get a forcing-clean feedback estimate.

## Environment

Dedicated conda env **`obs-gf`** (`environment.yml`). Create/update with:

```bash
mamba env create -f environment.yml      # first time
mamba env update -f environment.yml      # after editing deps
```

Run project code with the env's interpreter. In this sandbox `conda run` and `conda
activate` are unreliable — **call the env binary by full path**:

```bash
/Users/cristi/miniforge3/envs/obs-gf/bin/python 01_preprocess.py
```

Key deps: xarray/dask/netcdf4/cftime (pangeo core), **intake-esm** + **xmip**
(catalog access + the pool's read-time harmonization hook — the hook imports xmip),
**xesmf/esmpy** (regridding), scikit-learn (ridge), cartopy, jupyterlab.

## Data

- **The catalog is touched ONLY in preprocessing.** `01_preprocess.py` (via
  `obsgf/catalog.py`) is the one and only place that opens the CMIP6 catalog; every
  downstream notebook reads `pre-processed_data/` exclusively. If a stage needs something
  from the pool, extract it in preprocessing and save it. (Verified: the analysis runs
  with the catalog path broken.)
- **Source:** local CMIP6 pool at `/Users/cristi/cmip6/`. Catalog:
  `/Users/cristi/cmip6/catalog/cmip6_local.json` (intake-esm). Read
  `/Users/cristi/cmip6/catalog/USING_THE_CATALOG.md` and `HOLDINGS.md` first.
  Always load with the pool's `preprocess.py` hook and `use_cftime=True`; files are
  byte-identical to ESGF and models stay on **native grids** (no regridding in storage).
- **`pre-processed_data/`** — stage-1 output: one annual-mean netcdf per
  variable/experiment/model/member, CMIP-named without the month field, e.g.
  `toa_Amon_CanESM5_amip-piForcing_r1i1p2f1_gn_1870-2014.nc`; plus one static
  `sftlf_<model>.nc` (land fraction) per model for the ocean masks.
- **`derived/`** — GFs, reconstructions, feedback time series, ocean masks (small netcdfs).
- Variables: `tas`, `toa` (= rsdt − rsut − rlut, net downward), `tos`, and static `sftlf`.
  Analysis uses tas-over-ocean as the SST proxy for training (tos not reported in
  amip-piForcing).

## Pipeline — three py-percent notebooks at the project root

The three stages are **notebooks**, not modules: py-percent scripts (`# %%` cells) that
open as notebooks in VS Code / Jupytext and also run top-to-bottom as plain scripts. Each
walks through **CanESM5** with every intermediate visible to plot/inspect, then batches
over all models, and ends with its figures. Run in order (each is idempotent — existing
outputs are skipped/overwritten identically):

```bash
python 01_preprocess.py   # catalog -> pre-processed_data/ (annual means + sftlf)
python 02_greens.py       # fit GFs + held-out validation -> derived/gf/, derived/series/, figures/
python 03_feedbacks.py    # both feedback definitions, 3 estimates -> derived/feedbacks{,_hist_ensemble}.nc, figures/
```

Knobs live in visible cells at the top of each notebook (`FORCE`, `ONLY_MODELS`, `ALPHAS`,
`WINDOW_LENGTH`, `RATIO_*`, `WALKTHROUGH_MODEL`) — no CLI.

**`obsgf/` is now helper modules only** (imported by the notebooks), per the
`earth-science-code-guidelines.md` split — boring, trusted machinery, not the science:
- `config.py` — paths, roster, file-finders, shared constants (`ANALYSIS_YEARS`, `BASELINE_YEARS`, `OCEAN_SFTLF_MAX`)
- `catalog.py` — all intake-esm / xmip catalog plumbing (imported only by 01)
- `masks.py` — ocean masks + area-weighted global means
- `regrid.py` — historical tos (ocean grid) → atmosphere grid (xesmf), cached
- `plotting.py` — `map_plot`, `series_vs_truth`, `ensemble_band`

The scientific "meat" (annual means, the ridge, held-out validation, the two feedback
definitions) lives **in the notebook cells**, defined right before use, so a first-year
grad student reads the method top-to-bottom without opening a module.

**Synthetic SST (extension point, currently tabled).** The GFs can be forced with any
external SST ensemble (synthetic or observed), not just historical tos. The seam is a
*directory of per-member `sst_*.nc` files* that a bridge regrids onto each model's GF grid
and runs through the same feedback code. A LIM-based synthetic-SST generator was built and
then set aside (the annual LIM overdamps low-frequency variability); all of it — the
`sstlim/` generator, the `apply_sst.py` bridge, ERSST input, and outputs — now lives in
`deprecated/` (see `deprecated/README.md`). The core analysis below is SST-source-agnostic
and does not depend on any of it.

**Ensemble members.** amip-piForcing has one member per model, so each GF is single.
Historical is a **member ensemble**: preprocessing emits one tos file per member, regrid
builds the xesmf weights once per model and applies to all members, and feedbacks loops
members. `historical_members(model)` / `representative_member(model)` (config) enumerate
them. Outputs: `feedbacks.nc` = 3-estimate comparison on the representative member
(dims `model, estimate, center_year`); `feedbacks_hist_ensemble.nc` = every member's
hist_gf λ, run-indexed (dims `run, center_year` with `model`/`member` coords) to handle
ragged per-model counts. To add members: drop new tos in the pool, rerun
preprocess→regrid→feedbacks; the roster/ensemble grow automatically.

The estimate dimension is named **`estimate`** (`amip_true`, `amip_gf`, `hist_gf`) —
*not* `method`, which collides with xarray's `.sel(method=...)` fill-method kwarg.

**Two feedback definitions**, both in the feedbacks files: `feedback_window` (30-yr
window slope of N′ on T′, coord `center_year`) and `feedback_ratio` (cumulative
N′(t)/T′(t) vs the 1870–1919 baseline, coord `year`, from 1940, 5-yr smoothed, NaN where
smoothed T′ ≤ 0.2 K so the denominator stays a robust positive warming signal).

## Conventions

- **Exploratory science, readability first** (see `earth-science-code-guidelines.md`):
  audience is a first-year atmospheric-sciences grad student. Keep the science at
  notebook-cell level with intermediates visible to plot/inspect; extract only boring,
  trusted machinery (I/O, catalog, regridding, plotting) into `obsgf/`. Don't collapse an
  inspectable multi-step computation into one opaque function. Not production code.
- **Keep it simple**: plain functions, no classes/frameworks. Don't add flexibility the
  project doesn't need.
- **Native grids everywhere.** Regridding, when unavoidable, is *model-internal*
  (a model's tos → that same model's atmosphere grid) — never onto a common grid.
- **Area-weight global means** (`cos(lat)`); an unweighted mean of net TOA is badly
  biased by the poles (~−27 vs correct ~+1.4 W/m²) — this bit us once.
- **Annual means are month-length weighted** (`days_in_month` on the native calendar),
  not plain `groupby('time.year').mean()`.
- **Anomalies** are relative to the 1870–1919 climatology (`config.BASELINE_YEARS`).
- Notebooks skip existing outputs unless `FORCE`, and carry sanity asserts — a clean
  run *is* the verification.

## Roster / data gaps (see `obsgf/config.py`)

- **Analysis roster is derived, not hardcoded.** `config.analysis_models()` scans
  `pre-processed_data/` and returns models that have all required amip-piForcing
  variables (tas, toa → a GF can be built) *and* historical tos (→ it can be applied).
  Both the GF and historical stages use this one roster, so the analysis auto-restricts
  to what preprocessing produced. Don't hardcode model lists in the analysis code —
  call `analysis_models()`.
- **Currently 7:** CESM2, CanESM5, HadGEM3-GC31-LL, IPSL-CM6A-LR, MIROC6, MRI-ESM2-0,
  TaiESM1.
- **Dropped from the roster:** GISS-E2-1-G (local amip-piForcing only 1950–1970),
  CNRM-CM6-1 (no amip-piForcing). A per-model GF must exist to reconstruct a historical
  run, so historical is *not* run on the wider tos-reporting set; their historical tos
  is not kept. Re-download a full GISS amip-piForcing run to include it.
- **`sftlf`** (land fraction, fx) is now in the catalog and extracted by preprocessing
  (`sftlf_<model>.nc`); no longer a manual download. Optionally `areacella` for exact
  area weights (currently cos-lat).
