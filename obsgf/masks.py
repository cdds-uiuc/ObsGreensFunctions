"""Ocean masks, area weights, and area-weighted global means.

Reads only pre-processed_data/ — no catalog access. The ocean mask lives on each model's
native *atmosphere* grid (the grid the GFs are defined on): threshold the pre-processed
land-fraction field, `sftlf < OCEAN_SFTLF_MAX`, and cache to derived/masks/. sftlf itself
is extracted from the catalog once, in the preprocessing stage.
"""

import numpy as np
import xarray as xr

from .config import DERIVED_DIR, OCEAN_SFTLF_MAX, find_preprocessed, find_sftlf


def _atmos_grid(model):
    """Reference atmosphere lat/lon (1D) from the model's preprocessed tas file."""
    tas = xr.open_dataset(find_preprocessed("tas", "amip-piForcing", model))
    return tas[["lat", "lon"]]


def ocean_mask(model, rebuild=False):
    """Boolean DataArray (True = ocean) on the model's atmosphere grid.

    Cached to derived/masks/ocean_{model}.nc.
    """
    out = DERIVED_DIR / "masks" / f"ocean_{model}.nc"
    if out.exists() and not rebuild:
        return xr.open_dataarray(out)

    grid = _atmos_grid(model)
    sftlf = xr.open_dataset(find_sftlf(model)).sftlf
    # align sftlf onto the exact tas grid (nearest — same native grid, guards float drift)
    sftlf = sftlf.interp(lat=grid.lat, lon=grid.lon, method="nearest")
    mask = (sftlf < OCEAN_SFTLF_MAX).rename("ocean")
    mask.attrs = {"description": f"ocean where sftlf < {OCEAN_SFTLF_MAX}%", "model": model}

    out.parent.mkdir(parents=True, exist_ok=True)
    mask.to_netcdf(out)
    return mask


def _coslat(lat):
    """cos(lat), clipped at 0 to absorb latitudes a hair beyond ±90 (float noise)."""
    return np.clip(np.cos(np.deg2rad(lat)), 0, None)


def area_weights(da):
    """cos(lat) weights broadcast over da's lat (unnormalized)."""
    w = _coslat(da.lat)
    return w.broadcast_like(da.isel({d: 0 for d in da.dims if d not in ("lat", "lon")}, missing_dims="ignore"))


def global_mean(da):
    """Area-weighted mean over lat/lon (cos-lat), skipping NaNs."""
    return da.weighted(_coslat(da.lat)).mean(("lat", "lon"))


if __name__ == "__main__":
    from .config import MODELS

    for m in MODELS:
        mask = ocean_mask(m, rebuild=True)
        frac = float(mask.mean())
        print(f"{m}: ocean fraction {frac:.2f}  grid {mask.sizes['lat']}x{mask.sizes['lon']}")
