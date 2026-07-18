"""Stage 2b: regrid historical tos (native ocean grid) -> model's atmosphere grid.

The GFs live on each model's atmosphere grid, so the historical SST field they are
applied to must be moved there first. This is a *model-internal* regrid (a model's own
ocean grid -> its own atmosphere grid); nothing is put on a common cross-model grid.

The xesmf weight matrix depends only on the (ocean, atmosphere) grid pair, so it is
built ONCE per model and reused across all ensemble members — adding large-ensemble
members costs no extra regridder setup.

Run:  python -m obsgf.regrid [--model M] [--force]
"""

import argparse

import xarray as xr
import xesmf as xe

from .config import DERIVED_DIR, MODELS, find_preprocessed, historical_members
from .masks import ocean_mask


def _out_path(model, member):
    return DERIVED_DIR / "regridded" / f"tos_{model}_historical_{member}.nc"


def regrid_tos(model, member, rebuild=False):
    """Historical tos for one member on the model's atmosphere grid, ocean-masked. Cached."""
    out = _out_path(model, member)
    if out.exists() and not rebuild:
        return xr.open_dataarray(out)
    return _regrid_members(model, [member], rebuild=True)[member]


def _regrid_members(model, members, rebuild=False):
    """Regrid a list of members, building the xesmf weights once for the model."""
    mask = ocean_mask(model)
    grid_out = xr.Dataset({"lat": mask.lat, "lon": mask.lon})
    regridder = None
    result = {}
    for member in members:
        out = _out_path(model, member)
        if out.exists() and not rebuild:
            result[member] = xr.open_dataarray(out)
            continue
        tos = xr.open_dataset(find_preprocessed("tos", "historical", model, member)).tos
        if regridder is None:  # same source grid for every member -> build once
            regridder = xe.Regridder(tos, grid_out, "bilinear", periodic=True, ignore_degenerate=True)
        tos_rg = regridder(tos, skipna=True, na_thres=0.5).where(mask).rename("tos")
        tos_rg.attrs = {"model": model, "member": member,
                        "note": "historical tos regridded to atmosphere grid, ocean-masked"}
        out.parent.mkdir(parents=True, exist_ok=True)
        tos_rg.to_netcdf(out)
        result[member] = tos_rg
    return result


def regrid_model(model, rebuild=False):
    """Regrid every pre-processed historical member of a model. Returns {member: da}."""
    return _regrid_members(model, historical_members(model), rebuild=rebuild)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    models = [args.model] if args.model else MODELS
    for m in models:
        res = regrid_model(m, rebuild=args.force)
        one = next(iter(res.values()))
        print(f"{m}: {len(res):3d} member(s) regridded to {one.sizes['lat']}x{one.sizes['lon']}")


if __name__ == "__main__":
    raise SystemExit(main())
