"""CMIP6-pool access + preprocessing I/O plumbing.  Imported ONLY by 01_preprocess.

You do not need to read this file to follow the science. It is the intake-esm / xmip
machinery that turns the CMIP6 catalog into harmonized monthly xarray datasets, one
member at a time, on native grids. Keeping it here is what lets every later stage read
`pre-processed_data/` without ever opening the catalog.
"""

import sys

import intake
import xarray as xr

from .config import CATALOG_DIR, CATALOG_JSON, PREPROCESSED_DIR, TOA_INPUTS

sys.path.insert(0, str(CATALOG_DIR))
from preprocess import preprocess as pool_harmonize  # noqa: E402  (the pool's read-time hook)

# CMIP table each variable lives in (Amon = atmosphere monthly, Omon = ocean, fx = static)
TABLE_FOR_VAR = {"tas": "Amon", "toa": "Amon", "ts": "Amon", "tos": "Omon", "sftlf": "fx"}

_DROP_PATTERNS = ("bnds", "bounds", "vertices")   # bounds/vertex vars we don't keep
_REGULAR_TOL = 1e-2                                # deg; atmosphere grids are regular within this


def open_catalog():
    """Open the local CMIP6 intake-esm catalog."""
    return intake.open_esm_datastore(str(CATALOG_JSON))


def raw_vars_for(var):
    """The raw CMIP variables a target needs (toa is built from three fluxes)."""
    return TOA_INPUTS if var == "toa" else [var]


def list_members(cat, model, experiment, variables):
    """(sorted members, grid_label) available for a model/experiment/variable set.

    Members are loaded one at a time (see load_member), which is robust to large
    ensembles whose per-member file splits don't concatenate cleanly. If a variable is
    published on several grids (e.g. tos on gn + gr), native `gn` is preferred so all
    members of a model share one grid.
    """
    tables = sorted({TABLE_FOR_VAR.get(v, "Amon") for v in variables})
    sub = cat.search(source_id=model, experiment_id=experiment,
                     variable_id=variables, table_id=tables)
    if sub.df.empty:
        return [], None
    grids = sorted(sub.df.grid_label.unique())
    grid = "gn" if "gn" in grids else grids[0]
    members = sorted(sub.df[sub.df.grid_label == grid].member_id.unique())
    return members, grid


def load_member(cat, model, experiment, variables, member, grid):
    """Harmonized *monthly* dataset for one member (its time-file splits concatenated)."""
    tables = sorted({TABLE_FOR_VAR.get(v, "Amon") for v in variables})
    sub = cat.search(source_id=model, experiment_id=experiment, member_id=member,
                     variable_id=variables, table_id=tables, grid_label=grid)
    dsets = sub.to_dataset_dict(
        preprocess=pool_harmonize,
        xarray_open_kwargs={"use_cftime": True, "chunks": {"time": 600}},
        progressbar=False,
    )
    assert len(dsets) == 1, f"expected one dataset for {model}/{experiment}/{member}, got {list(dsets)}"
    key, ds = dsets.popitem()
    if "member_id" in ds.dims:
        ds = ds.isel(member_id=0)
    atmos = "Omon" not in key   # Omon = ocean (curvilinear); Amon/fx = atmosphere (regular)
    return tidy(ds, atmos)


def tidy(ds, atmos):
    """Drop bounds/vertices; give atmosphere grids plain 1-D lat/lon dims."""
    drop = [v for v in list(ds.coords) + list(ds.data_vars)
            if any(p in v for p in _DROP_PATTERNS)]
    ds = ds.drop_vars(drop, errors="ignore")
    return _to_1d_latlon(ds) if atmos else ds


def _to_1d_latlon(ds):
    """Recover 1-D lat/lon dims on a regular grid (xmip broadcasts them to 2-D on x/y)."""
    if "lat" in ds.dims and "lon" in ds.dims:
        return ds
    lon, lat = ds.lon, ds.lat
    if lon.ndim == 2:
        if not (float(lon.std("y").max()) < _REGULAR_TOL and float(lat.std("x").max()) < _REGULAR_TOL):
            raise ValueError("grid is not separable to 1-D (curvilinear?)")
        ds = ds.assign_coords(lon=lon.isel(y=0), lat=lat.isel(x=0))
    return ds.swap_dims({"x": "lon", "y": "lat"}).drop_vars(["x", "y"], errors="ignore")


def out_name(var, model, experiment, member, grid, years):
    """Pre-processed filename for an annual-mean member, CMIP-style without the month."""
    return f"{var}_{TABLE_FOR_VAR[var]}_{model}_{experiment}_{member}_{grid}_{years[0]}-{years[1]}.nc"


def existing_output(var, model, experiment, member, grid):
    """An already-written file for this member (year range unknown until load), or None."""
    hits = list(PREPROCESSED_DIR.glob(
        f"{var}_{TABLE_FOR_VAR[var]}_{model}_{experiment}_{member}_{grid}_*.nc"))
    return hits[0] if hits else None
