# Refactoring Guidelines: Exploratory Earth-Science Analysis Code

**Purpose:** Guidance for refactoring an earth-science analysis project (Python, Jupyter-centric, visualization-heavy). This project is deliberately in the **early, exploratory phase** of its lifecycle: the priority is understanding the science and inspecting/visualizing intermediate results, **not** production hardening. Refactor toward clarity and inspectability, not toward maximum encapsulation.

**Read this before restructuring anything.** The default software instinct — "break everything into many small functions" — is only partially right here. Over-encapsulating hides the intermediate values that are the whole point of exploratory analysis. Follow the split lines below.

---

## Guiding principle

Intermediate values *are* the deliverable during exploration. Regridded fields, masks before/after, anomalies, unit conversions — the analyst needs these sitting in the namespace to plot, inspect, and sanity-check (`.min()`, `.max()`, units, a suspicious grid cell). A function that swallows these and returns only a final result destroys the thing we care most about right now. Optimize for **"can I see and plot every meaningful step?"** over **"is this maximally abstracted?"**

## What to extract into functions / helper modules

Pull out the **stable, boring, reusable machinery** — code that no longer needs inspecting and only adds noise when repeated inline:

- **I/O and loading:** opening CMIP/netCDF/Zarr files, path handling, catalog queries.
- **Standard transforms you trust:** applying a standard land/ocean mask, standard regridding, standard unit conversions.
- **Plotting routines:** map plots, time series, panel figures — anything called repeatedly with different data. A well-named `plot_map(da, ...)` keeps notebooks readable *and* preserves inspectability, because its **input** stays visible in the namespace.
- Move this machinery into an importable `.py` module (e.g. `utils.py`, `io.py`, `plotting.py`) rather than redefining in notebooks. Import it at the top of the notebook.

## What to keep at cell level (do NOT over-extract)

- **The scientific "meat"** currently being reasoned about — the transformations whose correctness the analyst is actively checking. Keep these linear, one meaningful step per cell, with intermediates left in the namespace.
- **The step currently being debugged.** Encapsulate it *after* it's understood and trusted, not before.
- Do not collapse an inspectable multi-step computation into one opaque function just to reduce line count.

## Functions should return intermediates, not hide them

When you do extract a function in this phase, prefer returning the intermediate(s) the analyst cares about — either directly, or as a small `dict` / `xarray.Dataset` bundling the useful stages — rather than only the final product. This keeps encapsulation *and* inspectability.

```python
# Prefer this in exploratory phase:
def compute_anomaly(da, climatology):
    deseasoned = da.groupby("time.month") - climatology
    anomaly = deseasoned - deseasoned.mean("time")
    return {"deseasoned": deseasoned, "anomaly": anomaly}  # intermediates visible

# Over this, which hides the middle step:
def compute_anomaly(da, climatology):
    return ((da.groupby("time.month") - climatology)
            - (da.groupby("time.month") - climatology).mean("time"))
```

## Notebook correctness hazards to fix while refactoring

These are real bugs, not style — address them even in exploratory code:

- **Out-of-order execution / hidden state:** long runs of cell-level code accumulate globals and produce silently wrong results when cells are re-run out of order. Where a block has grown fragile, wrapping it in a function to create a clean scope is a legitimate correctness fix (weigh against inspectability).
- **Accidental global reuse:** watch for variables silently carried between unrelated cells.
- **Mutation in place:** be explicit when operations mutate arrays/datasets vs. return copies.

## Naming and readability

- Name functions and key intermediates for **what they mean scientifically** (`sea_surface_temp_anomaly`, not `tmp2`). Names are the documentation.
- Keep cells short and single-purpose so the notebook reads as a narrative: load → transform → plot → inspect → next step.
- A bit of duplication is acceptable in this phase; avoid premature shared abstractions that lock callers into an interface that may turn out wrong.

## The lifecycle (where this project sits, and where it's heading)

1. **[WE ARE HERE] Explore:** linear cells, visible intermediates, inspect and plot freely.
2. **Stabilize:** once a step is understood and trusted, refactor it into a named function or helper module.
3. **Consolidate:** notebook becomes a readable sequence of helper calls with plots between them; messy scaffolding moves into `.py` modules.
4. **Harden (later, out of scope now):** anything that will outlive exploration or be run by others migrates to modules with real unit tests. A notebook full of globals is not shippable — but we are not shipping yet.

## Do / Don't summary for the refactor

**Do**
- Extract I/O, standard transforms, and plotting into importable helpers.
- Keep the active scientific computation and any step under debug at cell level.
- Have extracted functions return intermediates the analyst inspects.
- Fix out-of-order-execution and hidden-global hazards.
- Use scientifically meaningful names for functions and intermediates.

**Don't**
- Collapse inspectable multi-step science into one opaque function.
- Extract single-use blocks that aren't independently meaningful just to shrink cells.
- Introduce shared abstractions prematurely.
- Add production scaffolding (heavy testing, config frameworks, packaging) — wrong phase.
- Remove the ability to drop back to cell level and inspect a value when a result looks off.
