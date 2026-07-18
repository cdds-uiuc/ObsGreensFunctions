# Observational Green's Functions — implementation plan

Goal (from [obsGF_design-doc.md](obsGF_design-doc.md)): assess variations in net radiative
feedbacks over historical simulations using Green's functions derived from amip-piForcing
simulations, trained on tas-over-ocean and applied to historical tos.

## 0. What exists today, and what carries over

The current code lives in `newbooks/` as exploratory notebooks:

- `1_preprocess.ipynb` reads already-processed files from a (no longer accessible) cluster
  path, regrids all models to a common grid, and bundles them into `data_*deg.nc`
  (tas maps + global-mean TOA/TAS scalars, amip-piForcing + abrupt-4xCO2 concatenated in time).
- `2_regression.ipynb` / `4_regression_precip.ipynb` explore GF estimation with
  OLS/Ridge/Lasso and EOF truncation on those bundles.
- `3_Bayes.ipynb` + `stan/` explore a Bayesian (Stan) formulation with multi-model priors.

Per the design doc this is refactored, not extended. What carries over conceptually:
the tas-over-ocean masking trick, the stack-to-`z`/Ridge regression pattern, and the
lesson that plain OLS on ~10⁴ grid points × 145 years is hopeless without regularization
(the notebooks converged on Ridge, α ≈ 5000 at 1°). The Stan/Bayesian branch, EOF
experiments, abrupt-4xCO2, precip, and CRE variables are all **out of scope** for v1.

## 1. Data inventory (local CMIP6 pool, checked 2026-07-16)

Catalog: `/Users/cristi/cmip6/catalog/cmip6_local.json` (intake-esm), harmonize on read
with the pool's `preprocess.py` hook and `use_cftime=True`.

- **amip-piForcing** (Amon tas/rlut/rsut/rsdt): CESM2, CanESM5 (r1i1p2f1),
  HadGEM3-GC31-LL (r1i1p1f3), IPSL-CM6A-LR (gr), MIROC6, MRI-ESM2-0, TaiESM1 — all
  1870–2014 (TaiESM1 starts 1850; trim to 1870). **GISS-E2-1-G only covers 1950–1970 —
  unusable for GF training; exclude or re-download the full run.**
- **historical** (Amon tas/rlut/rsut/rsdt + Omon tos): all 9 models, 1850–2014.
- **tos in amip-piForcing**: only 1 model — this is exactly why the design doc uses
  tas-over-ocean as the SST proxy for GF training.
- **Not in the pool: `sftlf`** (land fraction). See §3.

**Files to download (user action):** `sftlf`, table `fx`, one per model — e.g.
`sftlf_fx_CESM2_historical_r1i1p1f1_gn.nc`. Grab it for the historical (or amip, or
piControl — it's time-invariant) experiment, any member, for each of the 8 usable models.
Optional but nice: `areacella_fx_*` for exact area weights (otherwise we use cos(lat),
which is fine on these regular/gaussian atmosphere grids).

## 2. Repository layout (simple, flat)

```
obsgf/                  # small importable package, plain functions
  config.py             # model/member table, paths, constants (baseline years, window length)
  preprocess.py         # stage 1: catalog -> pre-processed_data/   (runnable: python -m obsgf.preprocess)
  masks.py              # sftlf -> ocean mask; area weights; global means
  regrid.py             # per-model tos (ocean grid) -> atmosphere grid
  greens.py             # anomalies, GF estimation (ridge), GF application/reconstruction
  feedbacks.py          # 30-yr sliding-window regression
notebooks/
  01_greens_functions.ipynb    # fit GFs, maps, closure checks
  02_feedbacks.ipynb           # direct vs GF feedbacks, final figure
pre-processed_data/     # stage-1 output (gitignored if repo is ever git-init'd)
derived/                # GFs, reconstructions, feedback time series (small netcdfs)
figures/
environment.yml         # conda env `obs-gf` (pangeo stack + intake-esm, xmip, xesmf)
CLAUDE.md               # agent onboarding: env, commands, conventions
```

No classes, no plugin systems, no CLI framework — each module is a handful of functions;
each stage is idempotent and writes netcdf, so any stage can be rerun alone.

**Environment.** Dedicated conda env `obs-gf` (`environment.yml`), created with
`mamba env create -f environment.yml`. Pangeo core (xarray/dask/netcdf4/cftime) +
intake-esm and xmip (catalog access + the pool's read-time harmonization hook) +
xesmf/esmpy (per-model regridding) + scikit-learn (ridge) + cartopy + jupyterlab.
The pool's `preprocess.py` hook depends on **xmip**, so it must be in the env.

## 3. Stage 1 — Pre-processing (`obsgf/preprocess.py`)

For each (model, experiment, member) in the config table:

1. `cat.search(...)` → `to_dataset_dict(preprocess=pool_preprocess, use_cftime=True)`;
   intake-esm already concatenates the multi-file time splits.
2. Build `toa = rsdt − rsut − rlut` (net downward TOA flux, W m⁻²).
3. Annual means weighted by month length (`ds.time.dt.days_in_month`, native calendar) —
   plain `groupby('time.year').mean()` is biased on real-world calendars.
4. Keep native grid, minimal harmonized coords (drop bounds/vertices to keep files lean).
5. Write one file per variable/experiment/model/member, CMIP-style naming without months:
   `{var}_{table}_{model}_{experiment}_{member}_{grid}_{y0}-{y1}.nc`
   e.g. `toa_Amon_CanESM5_amip-piForcing_r1i1p2f1_gn_1870-2014.nc`,
   `tos_Omon_CanESM5_historical_r1i1p1f1_gn_1850-2014.nc`.

Scope: tas + toa for amip-piForcing and historical; tos for historical (and where else
reported). Verification: for each output assert expected year count, no NaN in tas/toa,
toa global-mean magnitude sanity (historical net TOA imbalance ~0–1.5 W m⁻²).

## 4. Stage 2 — Masks and per-model regridding

- `masks.py`: ocean mask = `sftlf < 50%` on each model's atmosphere grid; area weights
  = cos(lat) (or areacella if downloaded); `global_mean(da, weights)`.
- `regrid.py`: **necessary addition the design doc doesn't spell out** — historical `tos`
  lives on each model's native *ocean* (curvilinear) grid, but the GFs are defined on
  that model's *atmosphere* grid. Regrid tos → atmosphere grid **within each model**
  (xesmf bilinear + nearest-neighbour fill at coastlines, then apply the ocean mask).
  This stays true to "native grids" in spirit: nothing is put on a common grid;
  regridding is model-internal and unavoidable. The xesmf weight matrix depends only
  on the (ocean grid, atmosphere grid) pair, so it is computed **once per model** and
  reused across all ensemble members — adding large-ensemble tos members costs no
  extra regridder setup. (If we ever apply a GF to a *different* model's tos — or to
  observed SSTs — the same machinery regrids that source onto the GF's grid; only the
  weight cache key changes.)

## 5. Stage 3 — Green's functions from amip-piForcing (`greens.py`)

Per model (7–8 models, GISS excluded unless re-downloaded):

1. Anomalies of tas(x), tas_global, toa_global relative to the **1870–1919 climatology**
   (first 50 years, per design doc).
2. Predictor matrix: tas anomalies at ocean points only (mask from §4), each column
   scaled by its area weight so the GF has a clean ∂(global mean)/∂T(x) interpretation.
3. Two GFs per model by ridge regression over the full 145 years:
   - `G_toa(x)`: toa_global′(t) = Σₓ G_toa(x) · T′(x,t)
   - `G_tas(x)`: tas_global′(t) = Σₓ G_tas(x) · T′(x,t)
4. Ridge α chosen by **blocked cross-validation in time** (leave-out contiguous blocks,
   within the training data only) to respect autocorrelation; report the CV curve
   rather than hand-picking α.
5. **Train–test–validation within amip-piForcing:** all validation of the GF
   methodology happens inside amip-piForcing, and the skill target is the
   reconstruction of **N (toa_global′) and T (tas_global′) separately** — not λ.
   Withhold a contiguous **20-yr segment**, train the GF on the remaining ~125 years
   (α chosen by blocked CV within the training years), and score the reconstructions
   N̂ and T̂ against truth in the withheld segment (r², RMSE). Repeat for several
   segment positions (early/middle/late record) and per model. This step settles the
   methodology — α range, ocean mask, area weighting — before λ is ever computed.
6. **Final GFs**: once the methodology is validated, retrain on the **full 145-yr
   record** with the selected α. These full-record GFs are what get used for the
   λ prediction (§6) and the historical application.
7. Outputs: `derived/GF_{toa,tas}_{model}.nc` (maps on the native atmosphere grid) +
   held-out N/T reconstruction skill tables.

Checks: GF maps should show the known pattern (strongly negative ∂R̄/∂T over the
West Pacific warm pool, weak/positive in East Pacific — cf. Dong et al. 2019);
held-out-segment skill is the honest number, in-sample r² only a diagnostic.

## 6. Stage 4 — Feedbacks (`feedbacks.py`)

**Why there is no "direct" historical feedback.** In historical simulations the TOA
imbalance mixes forcing and response, N′ = F′ + λ·T′, and F′(t) is not known, so
regressing N′ on T′ does not give λ. In amip-piForcing the forcing is held at
pre-industrial (F′ = 0), so N′ = λ·T′ and the regression is valid. Crucially, the GFs
are trained in that zero-forcing world, so **GF-reconstructed toa is the radiative
response alone** — applying the GFs to historical SST patterns yields a λ estimate
that is uncontaminated by the unknown historical forcing. That is the point of the
method, and it means GF validation can only happen inside amip-piForcing (§5.5).

**Two feedback definitions**, each computed for all three estimates below:
- **window** (`feedback_window`): OLS slope of N′ on T′ in a 30-yr sliding window,
  at the window center — the local/decadal feedback.
- **ratio** (`feedback_ratio`): cumulative/effective feedback λ(t) = N′(t)/T′(t)
  relative to the 1870–1919 baseline, from 1940 onward (5-yr smoothed), defined only
  where smoothed T′ > 0.2 K so the denominator is a robust positive warming signal
  (guards the mid-century aerosol plateau where T′ crosses zero and the ratio would
  blow up). Both are well-posed because N′ is forcing-free (amip) or the GF-reconstructed
  radiative response alone (hist) — never the raw historical N = F′ + λT′.

λ(t) via OLS slope of toa_global′ on tas_global′ in 30-yr sliding windows (stride 1 yr,
timestamped at window center), computed three ways:

1. **Direct amip-piForcing**: model's own toa_global′ vs tas_global′ (truth; valid
   because F′ = 0).
2. **GF-predicted amip-piForcing**: apply the **full-record** GFs (§5.6) to the
   amip-piForcing tas-over-ocean anomalies to predict N̂(t) and T̂(t), compute λ̂(t)
   in 30-yr windows, and compare with the true λ(t) from (1). This is the λ-level
   test of the (already N/T-validated) GF method.
3. **GF-derived historical**: apply each model's full-record GFs to its own historical
   tos anomalies (regridded per §4; baseline 1870–1919 for consistency with training —
   tos is in °C, tas in K, but anomalies make the offset irrelevant). Sea-ice-covered
   cells where tos is frozen/masked: fill anomaly with 0 (GF contributes nothing
   there), same convention as training-mask consistency requires.

**Role of the historical runs:** they are not a validation target — they serve as a
**large ensemble** characterizing the natural (internal + forced-pattern) variability
of λ in coupled models: the spread of 30-yr-window λ across the historical runs
(restricted to the analysis roster — the models that have a GF, currently 7)
tells us how big λ variations "naturally" are in coupled models, against which the
amip-piForcing λ trajectory (driven by the observed SST pattern) is compared.
A useful property: extending this ensemble with more members only requires
downloading **tos** (plus nothing else), since both N and T are GF-reconstructed —
e.g. CanESM5/MIROC6 large-ensemble members are cheap to add later.

Output: `derived/feedbacks.nc` — λ(window-center-year) per model per method.

## 7. Stage 5 — Final figure (`notebooks/02_feedbacks.ipynb`)

The plot from the design doc: time series of 30-yr-window λ:
- amip-piForcing true (black) and GF-predicted (colored) per model — how well the GF
  captures the pattern effect;
- the historical GF-derived ensemble (thin lines and/or spread envelope) — the natural
  variability of λ in coupled models, against which the amip-piForcing trajectory is
  judged (does the observed-SST-driven λ excursion fall outside coupled-model
  variability?).
  Secondary diagnostics (notebook 01): GF maps, and the held-out train/test skill of
  N̂ and T̂ (time series overlays + r²/RMSE tables per model and segment position).

## 8. Decisions taken / open questions

- **Per-model GFs** applied to the same model's historical run (no common grid, no
  cross-model transfer). A multi-model-mean GF would need a common grid — possible
  later extension, not v1.
- **Endgame (post-v1):** apply the GFs to other SST fields — observed (HadISST/ERSST)
  or statistically generated — via the same regrid-to-GF-grid machinery (§4). The
  GF-application code takes "an SST anomaly field on some grid" as input, with no
  assumption that it came from the same model.
- **Analysis roster is derived, not hardcoded.** `config.analysis_models()` scans
  `pre-processed_data/` and returns the models that have every required amip-piForcing
  variable (so a GF can be built) *and* historical tos (so it can be applied). Both the
  GF and historical stages read this one roster, so the analysis auto-restricts to what
  preprocessing actually produced (currently 7). Historical is **not** run on the wider
  set of models that merely report tos — a per-model GF must exist to reconstruct a run.
- **GISS-E2-1-G**: preprocessed for historical tos but dropped from the roster
  (amip-piForcing only 1950–1970 locally → no GF). Re-download the full 1870–2014
  amip-piForcing run to include it.
- **CNRM-CM6-1**: has no amip-piForcing in the pool at all → no GF → dropped from the
  roster (historical tos preprocessed but unused).
- **Baseline for historical anomalies**: 1870–1919 (mirrors GF training). Alternative
  (first 50 yrs of historical, 1850–1899) is a one-line config change.
- **Ensemble members**: amip-piForcing has one member per model (single GF each).
  Historical is a member ensemble — **226 runs** (CanESM5 65, HadGEM3-GC31-LL 55,
  MIROC6 50, IPSL-CM6A-LR 33, CESM2 11, MRI-ESM2-0 11, TaiESM1 1). `feedbacks.py` loops
  members; `feedbacks_hist_ensemble.nc` holds every member's λ (run-indexed to handle
  ragged counts), while `feedbacks.nc` keeps the representative member for the 3-estimate
  plot. Deepening the ensemble just needs more tos dropped in the pool + a rerun of
  preprocess→regrid→feedbacks (GFs are amip-only, so unaffected).
- **Multi-grid tos**: some models publish historical tos on both `gn` (native) and `gr`
  (regridded); preprocessing keeps `gn` so every member of a model shares one grid.
  Members are loaded one at a time (robust to large ensembles with messy file splits;
  skip-before-load makes reruns cheap).
- Windows shorter than 30 yr at the record edges are simply not computed (no padding).

## 9. Build order & verification

0. ✅ `environment.yml` → `mamba env create` (env `obs-gf`).
1. ✅ `config.py` + `preprocess.py`; CanESM5 verified end-to-end (λ_full = −1.47,
   early→late window −0.61 → −1.88 W/m²/K, the expected pattern-effect steepening);
   full roster preprocessing run.
2. ✅ `sftlf` now in the catalog (fx, all 9 models); `masks.py` + `regrid.py` built,
   ocean fractions ~0.66, tos regridded to each atmosphere grid.
3. ✅ `greens.py` on all 7 models. Held-out skill: tas r² ≈ 0.78–0.91, toa r² ≈ 0.3–0.5
   (toa intrinsically noisier — expected). Area-fair (√cos φ) predictor weighting;
   GF maps show the tropical-Pacific pattern-effect fingerprint.
4. ✅ `feedbacks.py`, all three estimates. Method-1 λ(t) reproduces the pattern effect
   (multi-model mean −1.63 → −2.37 W/m²/K, early→late); method-2 (GF) tracks it
   (−1.67 → −2.32); historical GF feedback varies far less (−1.39 → −1.53).
5. ✅ `notebooks/01_greens_functions.ipynb`, `02_feedbacks.ipynb` (executed, figures
   embedded); standalone PNGs in `figures/`.

### Results snapshot

The GF reproduces the amip-piForcing feedback evolution (amip_gf ≈ amip_true, r high
across all windows/models), confirming the method. Applied to coupled historical SSTs,
the GF-derived feedback shows a much weaker trend than amip-piForcing — i.e. the
observed/amip SST *pattern* drives a stronger feedback swing (pattern effect) than the
coupled models' own historical SST evolution does. Caveats surfaced by the run:
toa reconstruction skill is modest (single amip realization; noisy annual global TOA),
and historical GF feedback is noisy in windows with low within-window SST-pattern
variance (CESM2 early decades the clearest case).
