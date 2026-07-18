"""Reusable plotting helpers for the analysis notebooks.

These wrap the repeated cartopy / matplotlib boilerplate so notebook cells stay about
the science. The DATA you pass in stays in the notebook namespace, so you can still
inspect it (`.min()`, `.max()`, a suspicious cell) right next to the plot.
"""

import matplotlib.pyplot as plt
import numpy as np


def map_plot(da, ax=None, title=None, vmin=None, vmax=None, cmap="RdBu_r",
             robust=False, cbar=True, cbar_label=None):
    """Plot a (lat, lon) field on a Robinson map (Pacific-centred), coastlines added.

    `robust=True` sets symmetric limits at the 98th percentile of |da| — handy for GF
    maps whose per-cell magnitude varies with grid resolution.
    """
    import cartopy.crs as ccrs

    if ax is None:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
    if robust and vmin is None:
        v = float(np.nanpercentile(np.abs(da.values), 98))
        vmin, vmax = -v, v
    da = da.sortby(["lat", "lon"])          # guard against descending / wrapped coords
    p = da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                add_colorbar=cbar, cbar_kwargs={"label": cbar_label, "shrink": 0.7} if cbar else None)
    ax.coastlines(lw=0.4)
    if title:
        ax.set_title(title)
    return p


def series_vs_truth(year, truth, recon, ax=None, ylabel="", title="",
                    truth_label="truth", recon_label="GF"):
    """Overlay a reconstructed series (red) on the truth (black) — the skill eyeball."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(year, truth, "k", lw=1.6, label=truth_label)
    ax.plot(year, recon, "tab:red", lw=1.4, label=recon_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def ensemble_band(ax, x, arr, color, label, pct=(5, 95)):
    """A member ensemble as a shaded percentile band + mean line; a single member -> line.

    `arr` is (n_member, n_time). NaNs are ignored (some ratio years are undefined).
    """
    arr = np.atleast_2d(arr)
    if arr.shape[0] == 1:
        ax.plot(x, arr[0], color=color, lw=1.5, label=label)
        return
    lo, hi = np.nanpercentile(arr, pct[0], 0), np.nanpercentile(arr, pct[1], 0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.plot(x, np.nanmean(arr, 0), color=color, lw=2, label=label)
