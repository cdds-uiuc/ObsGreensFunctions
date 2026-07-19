# CLAUDE.md

Guidance for working in this repository.

## What this project is

Observationally-derived **Green's functions (GFs)** for net radiative feedback.
GFs are trained on **amip-piForcing** simulations (constant pre-industrial forcing,
so net TOA = radiative response only) and used to reconstruct feedback λ from SST
patterns. (The original design/goal notes are not in the repo — they live in the
early git history and locally.)

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
- Variables: `tas`, `toa` (= rsdt − rsut − rlut, net downward), `ts`, `tos`, and static
  `sftlf`.
- **GF predictor = ocean `ts`, within ±55° latitude.** Over open ocean `ts` (surface
  temperature) *is* the prescribed SST — a cleaner, causal predictor than `tas` (2-m air
  temp, which is itself part of the response, so N-on-tas has inflated response-response
  skill). The ±55° bound (`PREDICTOR_LAT_BOUND` in 02_greens) drops sea-ice cells, where
  `ts` is the ice-surface temperature, not the ocean beneath (AMIP can't give the ocean
  under ice). The **targets stay global `tas` (for T) and global `toa` (for N)**, full
  globe — λ is per global surface-air temperature. So there are still two GFs (`G_tas`,
  `G_toa`) sharing the ocean-`ts` predictor. Historical application is unchanged: the GF
  is NaN poleward of ±55°, so applying it to full-globe `tos` automatically uses only the
  ±55° cells. (Switching to `ts` lowers held-out reconstruction r² — partly healthy
  de-inflation, partly the bound's cost to global-T via polar amplification — but the
  *feedback* deliverable reproduces as well or better. `siconc` masking is a possible
  later refinement to recover high-latitude open ocean.)

## Pipeline — three py-percent notebooks at the project root

The three stages are **notebooks**, not modules: py-percent scripts (`# %%` cells) that
open as notebooks in VS Code / Jupytext and also run top-to-bottom as plain scripts. Each
walks through **CanESM5** with every intermediate visible to plot/inspect, then batches
over all models, and ends with its figures. Run in order (each is idempotent — existing
outputs are skipped/overwritten identically):

```bash
python 01_preprocess.py   # catalog -> pre-processed_data/ (annual means + sftlf)
python 02_greens.py       # fit GFs (ocean-ts predictor, ±55°) + held-out validation -> derived/gf/, series/, figures/
python 03_feedbacks.py    # windowed feedback (2 methods), 3 estimates -> derived/feedbacks_win{N}{,_hist_ensemble}.nc, figures/
```

Knobs live in visible cells at the top of each notebook (`FORCE`, `ONLY_MODELS`, `ALPHAS`,
`WINDOW_LENGTH`, `WALKTHROUGH_MODEL`) — no CLI.

**This project lives in Dropbox.** `derived/` and `pre-processed_data/` are marked
`com.dropbox.ignored` (`xattr -w com.dropbox.ignored 1 <dir>`) so Dropbox doesn't race the
pipeline's many netcdf writes — otherwise it intermittently locks a file mid-write, giving a
`PermissionError`/0-byte output (it bit us repeatedly, e.g. `GF_<model>.nc`). If a run still
hits it, quit Dropbox for the run and relaunch after.

**`obsgf/` is now helper modules only** (imported by the notebooks), per the
exploratory-science split (extract only boring machinery; keep the science in cells) —
trusted machinery, not the science:
- `config.py` — paths, roster, file-finders, shared constants (`ANALYSIS_YEARS`, `BASELINE_YEARS`, `OCEAN_SFTLF_MAX`)
- `catalog.py` — all intake-esm / xmip catalog plumbing (imported only by 01)
- `masks.py` — ocean masks + area-weighted global means
- `regrid.py` — historical tos (ocean grid) → atmosphere grid (xesmf), cached

Plotting helpers are *not* shared modules — each is used by only one notebook, so
`map_plot` lives in a setup cell of `02_greens.py` and `ensemble_band` in `03_feedbacks.py`
(a single-caller helper belongs with its caller; a one-function module is a smell).

The scientific "meat" (annual means, the ridge fit (scikit-learn `KernelRidge`, blocked-CV
α), held-out validation, the windowed feedback computed two ways) lives **in the notebook
cells**, defined right before use, so a first-year grad student reads the method
top-to-bottom without opening a module.

The three stages also open as real Jupyter notebooks: `jupytext --to notebook 0X.py`
(kernel `python3` from the env); convert back with `jupytext --to py:percent 0X.ipynb`. The
`.py` are the tracked source of truth; the `.ipynb` are gitignored/regenerable. They are
*not* auto-paired, so edits made in the `.ipynb` must be synced back (`jupytext --to
py:percent 0X.ipynb`) or they diverge from the `.py`.

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
them. Outputs (window length in the name): `feedbacks_win{N}.nc` = 3-estimate comparison on
the representative member (dims `model, estimate, end_year`); `feedbacks_hist_ensemble_win{N}.nc`
= every member's hist_gf λ, run-indexed (dims `run, end_year` with `model`/`member` coords)
to handle ragged per-model counts. To add members: drop new tos in the pool, rerun
preprocess→regrid→feedbacks; the roster/ensemble grow automatically.

The estimate dimension is named **`estimate`** (`amip_true`, `amip_gf`, `hist_gf`) —
*not* `method`, which collides with xarray's `.sel(method=...)` fill-method kwarg.

**Windowed feedback, two ways.** λ is a sliding-window estimate (default 35-yr,
`WINDOW_LENGTH`), computed two ways within each window and stored in the feedbacks files,
indexed by window **end year** (`end_year`):
- `feedback_slope` — OLS slope of N′ on T′ in the window (Gregory-style, *with* intercept:
  the window means of N′,T′ are nonzero, being anomalies vs pre-industrial). The direct,
  local feedback; internal N′–T′ covariance (ENSO) enters it.
- `feedback_trend_ratio` — ratio of the time-trends of N′ and T′ (each regressed on year in
  the window). A low-pass estimate — high-frequency internal covariance doesn't alias in —
  but only stable where the window has a solid warming trend (it blows up in early windows
  where the T′ trend crosses zero).

Figures: only the **slope** gets a per-model time series (`feedbacks_slope_win{N}.png`,
x = window end year). **Both** methods are compared over the single last window (ending 2014,
where each is stable) as per-model distributions — a KDE of the historical member ensemble
with the amip truth (black) and GF reconstruction (red, dashed) as vertical lines
(`feedbacks_lastwindow_{slope,trendratio}_win{N}.png`). `WINDOW_LENGTH` is in every output
filename, so re-running at 30/35/40 accumulates rather than overwrites. (The earlier
cumulative-ratio definition was dropped.)

## Conventions

- **Exploratory science, readability first**:
  audience is a first-year atmospheric-sciences grad student. Keep the science at
  notebook-cell level with intermediates visible to plot/inspect; extract only boring,
  trusted machinery (I/O, catalog, masking, regridding) into `obsgf/`. Don't collapse an
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

- **One hardcoded roster, `config.MODELS`.** Every model in it has a full amip-piForcing
  predictor/target set (tas, toa, ts → a GF can be built) *and* historical tos (→ it can be
  applied); all stages — preprocess (01), GF fitting (02), historical analysis (03) — iterate
  this one list. It's kept in `sorted()` order so the output model dimension is deterministic.
  Every model in `MODELS` is assumed fully present in `pre-processed_data/`; a missing file is
  a loud error, not a silent drop. (Earlier this was *derived* from a disk-scan, `analysis_models()`,
  which auto-dropped incomplete models — removed as unused flexibility once the roster stabilised.)
  To add a model: download its amip-piForcing (tas/toa/ts) + historical tos into the pool, add
  its name to `MODELS`, and rerun 01 → 02 → 03.
- **Currently 8:** CESM2, CNRM-CM6-1, CanESM5, HadGEM3-GC31-LL, IPSL-CM6A-LR, MIROC6,
  MRI-ESM2-0, TaiESM1. (CNRM-CM6-1 added once its amip-piForcing was downloaded; 30
  historical members.)
- **GISS-E2-1-G is not in the roster:** its local amip-piForcing covers only 1950–1970 —
  too short to build a GF — so it can't reconstruct a historical run. Re-download a full
  GISS amip-piForcing run and add it to `MODELS` to include it.
- **`sftlf`** (land fraction, fx) is now in the catalog and extracted by preprocessing
  (`sftlf_<model>.nc`); no longer a manual download. Optionally `areacella` for exact
  area weights (currently cos-lat).
