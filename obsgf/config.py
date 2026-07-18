"""Project configuration: paths, model roster, and analysis constants."""

from pathlib import Path

# --- paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = Path("/Users/cristi/cmip6/catalog")
CATALOG_JSON = CATALOG_DIR / "cmip6_local.json"

PREPROCESSED_DIR = PROJECT_ROOT / "pre-processed_data"
DERIVED_DIR = PROJECT_ROOT / "derived"
FIGURES_DIR = PROJECT_ROOT / "figures"

# --- candidate models to preprocess ----------------------------------------
# These are *attempted* during preprocessing. The analysis roster is NOT this list
# — it is derived from what preprocessing actually produced (see analysis_models()),
# so any model missing a required variable is dropped automatically.
CANDIDATE_AMIP_MODELS = [
    "CESM2",
    "CanESM5",
    "CNRM-CM6-1",
    "HadGEM3-GC31-LL",
    "IPSL-CM6A-LR",
    "MIROC6",
    "MRI-ESM2-0",
    "TaiESM1",
]
# GISS-E2-1-G has local amip-piForcing for 1950-1970 only, so it can't build a GF; it can
# still supply historical tos, so we preprocess it, but it won't enter the analysis.
CANDIDATE_HIST_MODELS = CANDIDATE_AMIP_MODELS + ["GISS-E2-1-G"]

# --- what to preprocess ----------------------------------------------------
# experiment -> (variables, candidate models). "toa" is derived from rsdt-rsut-rlut.
# "ts" (surface temperature) over open ocean IS the prescribed SST — a cleaner GF
# predictor than tas (2-m air temp). Historical runs only feed the GF reconstruction,
# so only tos is needed from them.
PREPROCESS_SPEC = {
    "amip-piForcing": (["tas", "toa", "ts"], CANDIDATE_AMIP_MODELS),
    "historical": (["tos"], CANDIDATE_HIST_MODELS),
}

TOA_INPUTS = ["rsdt", "rsut", "rlut"]  # toa = rsdt - rsut - rlut (net downward)

# sftlf (land-area fraction) is time-invariant (fx table); the pool carries it under
# piControl. Preprocessing extracts it once per model so the analysis (ocean masks)
# never needs the catalog. Written as pre-processed_data/sftlf_<model>.nc.
SFTLF_SOURCE = "piControl"

# --- analysis roster (derived from preprocessing output) -------------------
# A model is a per-model GF requires every amip-piForcing predictor/target variable;
# applying that GF to the model's own historical run requires historical tos. So the
# analysis roster is the set of models that have BOTH. Because GFs are per-model, the
# historical analysis uses this same roster — not the broader set of models that merely
# happen to report historical tos.
REQUIRED_AMIP_VARS = ["tas", "toa", "ts"]   # ts = SST predictor; tas/toa = global targets


def _preprocessed_models(experiment, variables):
    """Models in pre-processed_data/ that have every one of `variables` for `experiment`.

    Filenames are `{var}_{table}_{model}_{experiment}_{member}_{grid}_{years}.nc`;
    no field except the model name contains a hyphen and none contains an underscore,
    so positional split is safe.
    """
    have = {}
    for f in PREPROCESSED_DIR.glob("*.nc"):
        parts = f.stem.split("_")
        if len(parts) < 4:
            continue
        var, model, exp = parts[0], parts[2], parts[3]
        if exp == experiment:
            have.setdefault(model, set()).add(var)
    return {m for m, vs in have.items() if set(variables) <= vs}


def analysis_models():
    """The analysis roster: models with a full amip-piForcing predictor set AND
    historical tos, so a per-model GF can be both built and applied. Computed from
    what preprocessing produced, so it shrinks/grows with the data on disk."""
    amip = _preprocessed_models("amip-piForcing", REQUIRED_AMIP_VARS)
    hist = _preprocessed_models("historical", ["tos"])
    return sorted(amip & hist)


def find_preprocessed(var, experiment, model, member=None):
    """The pre-processed file for (var, experiment, model[, member]); error if not unique.

    With `member` omitted the match must be unique (the amip-piForcing case, one member
    per model); pass `member` to disambiguate when several exist (historical tos).
    """
    pat = (f"{var}_*_{model}_{experiment}_{member}_*.nc" if member
           else f"{var}_*_{model}_{experiment}_*.nc")
    hits = sorted(PREPROCESSED_DIR.glob(pat))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected 1 file for {var}/{experiment}/{model}/{member}, got {hits}")
    return hits[0]


def find_sftlf(model):
    """The pre-processed static sftlf (land fraction) file for a model."""
    path = PREPROCESSED_DIR / f"sftlf_{model}.nc"
    if not path.exists():
        raise FileNotFoundError(f"no pre-processed sftlf for {model}; run 01_preprocess.py")
    return path


def _member_of(path):
    """Member id from a pre-processed filename (5th underscore-separated field)."""
    return path.name.split("_")[4]


def historical_members(model):
    """Sorted historical tos members that have been pre-processed for a model."""
    return sorted(_member_of(f) for f in PREPROCESSED_DIR.glob(f"tos_*_{model}_historical_*.nc"))


def representative_member(model):
    """The historical member used for the headline 3-estimate comparison: the one that
    matches the model's amip-piForcing (GF) member if present, else the first."""
    amip_member = _member_of(find_preprocessed("tas", "amip-piForcing", model))
    members = historical_members(model)
    return amip_member if amip_member in members else members[0]


# --- shared analysis constants ---------------------------------------------
# Only constants used by more than one place live here; single-notebook knobs (ridge
# alphas, window length, ratio settings) sit in visible cells at the top of their
# notebook, where they belong for exploratory work.
#
# Common analysis window: models differ at the edges (TaiESM1 starts 1850, CESM2 runs
# to 2015); clip everything to this shared span so windows and holdouts align.
ANALYSIS_YEARS = (1870, 2014)  # used by 02_greens and 03_feedbacks
BASELINE_YEARS = (1870, 1919)  # anomaly reference climatology (first 50 yrs); both notebooks
OCEAN_SFTLF_MAX = 50.0         # cell is "ocean" if land fraction < this (%); used by masks.py
