"""Robustness analysis for the matched-mu_s bilayer campaign.

The script reuses the validated loading and profile definitions from
``analisis_bicapa_matched_mus_final.ipynb`` and adds three analyses in which
the independent Monte Carlo replica, rather than an angular or temporal bin,
is the resampling unit:

1. bootstrap uncertainty for the failure of stationary single-scale collapse;
2. separate depth weights for the absolute background B and interference
   sector P=C-B;
3. bootstrap propagation through the late-time recovery of the substrate
   transport mean free path.

Run from ``plots/layers`` with the project virtual environment.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


CORE_NOTEBOOK = Path("analisis_bicapa_matched_mus_final.ipynb")
CORE_CELLS = (2, 4, 5, 6, 8, 9, 11)


def load_core_namespace() -> None:
    """Execute only the shared definitions and campaign-loading cells."""
    with CORE_NOTEBOOK.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    namespace = globals()
    for index in CORE_CELLS:
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"{CORE_NOTEBOOK.name}:cell-{index}", "exec"), namespace)


load_core_namespace()


from utils.analysis import Stokes  # noqa: E402  (core establishes sys.path)


# Value selected by the bias-versus-replica-noise calibration in cell 2f of
# the reference notebook.  The bootstrap must use the same apex estimator.
CONV["n_apex"] = 3

OUTDIR = Path("figs_tesis_bicapa_mus_robustness")
OUTDIR.mkdir(exist_ok=True)

ORDERS = ("normal", "inverse")
N_BOOT = 4000
SEED = 20260813
N_REP = 5
PLATEAU_FRACTIONS = (0.15, 0.20, 0.25)


def count_vectors(total: int = N_REP, categories: int = N_REP) -> np.ndarray:
    """All multinomial count vectors with ``sum(counts) == total``."""
    rows = []
    for cuts in itertools.combinations(range(total + categories - 1), categories - 1):
        bounds = (-1, *cuts, total + categories - 1)
        rows.append([bounds[i + 1] - bounds[i] - 1 for i in range(categories)])
    return np.asarray(rows, dtype=int)


COUNTS = count_vectors()
COUNT_INDEX = {tuple(row): i for i, row in enumerate(COUNTS)}
RNG = np.random.default_rng(SEED)


def bootstrap_count_indices(n: int = N_BOOT) -> np.ndarray:
    draws = RNG.multinomial(N_REP, np.full(N_REP, 1.0 / N_REP), size=n)
    return np.asarray([COUNT_INDEX[tuple(row)] for row in draws], dtype=int)


def channel_stack(proc, c, which: str) -> np.ndarray:
    """Selected polarization channel for all stored time bins."""
    prefix = "coh" if which == "coherent" else "inc"
    stokes = Stokes(
        np.asarray(getattr(proc, f"{prefix}_s0"), float),
        np.asarray(getattr(proc, f"{prefix}_s1"), float),
        np.asarray(getattr(proc, f"{prefix}_s2"), float),
        np.asarray(getattr(proc, f"{prefix}_s3"), float),
    )
    values = _basis_for(c.pol)(stokes)[c.channel]
    values = np.asarray(values, float)
    return values[..., CONV["phi_main"]] if values.ndim == 3 else values


RAW = {}


def raw_stack(c, key):
    """Return q, C(time,q), B(time,q), loading each H5 sensor only once."""
    cache_key = (c.order, c.pol, key)
    if cache_key in RAW:
        return RAW[cache_key]

    loader = c.sweep[key]
    p1 = loader.processed_cbs("farfield_cbs_1")
    p2 = loader.processed_cbs("farfield_cbs_2")
    th1, th2 = np.asarray(p1.theta, float), np.asarray(p2.theta, float)
    use2 = th2 > th1[-1]
    theta = np.concatenate([th1, th2[use2]])
    coherent = np.concatenate(
        [channel_stack(p1, c, "coherent"), channel_stack(p2, c, "coherent")[..., use2]],
        axis=-1,
    )
    background = np.concatenate(
        [channel_stack(p1, c, "incoherent"), channel_stack(p2, c, "incoherent")[..., use2]],
        axis=-1,
    )
    result = (k * c.l_anchor * theta, coherent, background)
    RAW[cache_key] = result
    return result


def group_arrays(c, keys):
    """Stack the five independent replicas as (replica,time,q)."""
    loaded = [raw_stack(c, key) for key in keys]
    q = loaded[0][0]
    if len(loaded) != N_REP:
        raise ValueError(f"{c.tag}: expected {N_REP} replicas, found {len(loaded)}")
    if not all(np.allclose(item[0], q) for item in loaded[1:]):
        raise ValueError(f"{c.tag}: inconsistent angular grids within a configuration")
    coherent = np.stack([item[1] for item in loaded])
    background = np.stack([item[2] for item in loaded])
    return q, coherent, background


def ratio_profile(q, coherent, background) -> np.ndarray:
    enhancement = (coherent + eps) / (background + eps)
    return enhancement - tail_baseline(q, enhancement) + 1.0


def profile_for_counts(q, coherent, background, counts, time_index=0):
    csum = np.tensordot(counts, coherent[:, time_index], axes=(0, 0))
    bsum = np.tensordot(counts, background[:, time_index], axes=(0, 0))
    return ratio_profile(q, csum, bsum)


def central_profile(group, time_index=0):
    q, coherent, background = group
    return q, ratio_profile(q, coherent[:, time_index].sum(0), background[:, time_index].sum(0))


# ---------------------------------------------------------------------------
# 1. Replica bootstrap for stationary single-scale failure
# ---------------------------------------------------------------------------


LINEAR_GROUPS = {}
for order in ORDERS:
    c = CAMP[(order, "linear")]
    for layer in c.layers:
        LINEAR_GROUPS[(order, layer.z)] = group_arrays(c, layer.keys)
    for role in ("entry", "back"):
        LINEAR_GROUPS[(order, role)] = group_arrays(c, c.ctrl[role])


def precompute_shapes(group):
    q, coherent, background = group
    shapes = []
    for counts in COUNTS:
        profile = profile_for_counts(q, coherent, background, counts)
        shapes.append(normalized_shape(q, profile))
    return np.asarray(shapes)


SHAPE_LOOKUP = {name: precompute_shapes(group) for name, group in LINEAR_GROUPS.items()}
SHAPE_DRAWS = {name: bootstrap_count_indices() for name in LINEAR_GROUPS}


def spread_from_rows(rows) -> float:
    return collapse_spread(rows)


SHAPE_ROWS = []
SHAPE_BOOT = {}
control_names = [(order, role) for order in ORDERS for role in ("entry", "back")]

central_control_spread = spread_from_rows(
    [normalized_shape(*central_profile(LINEAR_GROUPS[name])) for name in control_names]
)

for order in ORDERS:
    c = CAMP[(order, "linear")]
    layer_names = [(order, layer.z) for layer in c.layers]
    central_layer_spread = spread_from_rows(
        [normalized_shape(*central_profile(LINEAR_GROUPS[name])) for name in layer_names]
    )

    layer_boot = np.empty(N_BOOT)
    control_boot = np.empty(N_BOOT)
    for ib in range(N_BOOT):
        layer_boot[ib] = spread_from_rows(
            [SHAPE_LOOKUP[name][SHAPE_DRAWS[name][ib]] for name in layer_names]
        )
        control_boot[ib] = spread_from_rows(
            [SHAPE_LOOKUP[name][SHAPE_DRAWS[name][ib]] for name in control_names]
        )

    difference = layer_boot - control_boot
    ratio = layer_boot / control_boot
    SHAPE_BOOT[order] = {"layer": layer_boot, "control": control_boot,
                         "difference": difference, "ratio": ratio}
    SHAPE_ROWS.append({
        "order": order,
        "bilayer_spread": central_layer_spread,
        "control_spread": central_control_spread,
        "spread_ratio": central_layer_spread / central_control_spread,
        "difference_ci_low": np.quantile(difference, 0.025),
        "difference_ci_high": np.quantile(difference, 0.975),
        "ratio_ci_low": np.quantile(ratio, 0.025),
        "ratio_ci_high": np.quantile(ratio, 0.975),
        "bootstrap_probability_difference_le_zero": float(np.mean(difference <= 0.0)),
    })


SHAPE_SUMMARY = pd.DataFrame(SHAPE_ROWS)
SHAPE_SUMMARY.to_csv(OUTDIR / "bootstrap_single_scale.csv", index=False)


apply(width_frac=0.72)
fig, ax = plt.subplots(figsize=(0.72 * TEXTWIDTH_IN, 0.42 * TEXTWIDTH_IN))
for xpos, order in enumerate(ORDERS):
    row = SHAPE_SUMMARY[SHAPE_SUMMARY["order"] == order].iloc[0]
    st = STYLE[order]
    ax.errorbar(
        xpos,
        row["spread_ratio"],
        yerr=[[row["spread_ratio"] - row["ratio_ci_low"]],
              [row["ratio_ci_high"] - row["spread_ratio"]]],
        fmt=st["marker"], color=st["color"], ms=5, capsize=3, lw=1.0,
    )
ax.axhline(1.0, color="black", ls="--", lw=0.8)
ax.set_xticks(range(len(ORDERS)), ORDERS)
ax.set_ylabel("bilayer/control spread")
ax.set_title("Replica-bootstrap test of single-scale collapse", loc="left", fontsize="small")
ax.grid(axis="y", alpha=0.18)
fig.savefig(OUTDIR / "F6_bootstrap_single_scale.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "F6_bootstrap_single_scale.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Separate stationary depth weights for B and P=C-B
# ---------------------------------------------------------------------------


def component_curve(group, counts, component: str):
    q, coherent, background = group
    cmean = np.tensordot(counts, coherent[:, 0], axes=(0, 0)) / counts.sum()
    bmean = np.tensordot(counts, background[:, 0], axes=(0, 0)) / counts.sum()
    if component == "B":
        return q, bmean
    if component == "P":
        return q, cmean - bmean
    raise ValueError(component)


def common_weight(q, layer, entry, back, q_max=CONV["q_fit"]):
    use = q <= q_max
    direction = (entry - back)[use]
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-30:
        return np.nan
    return float(np.dot((layer - back)[use], direction) / denominator)


WEIGHT_ROWS = []
WEIGHT_BOOT = {}
equal_counts = np.ones(N_REP, dtype=int)

for order in ORDERS:
    c = CAMP[(order, "linear")]
    entry_name, back_name = (order, "entry"), (order, "back")
    for layer in c.layers:
        layer_name = (order, layer.z)
        central = {}
        samples = {}
        for component in ("B", "P"):
            q, layer_curve = component_curve(LINEAR_GROUPS[layer_name], equal_counts, component)
            _, entry_curve = component_curve(LINEAR_GROUPS[entry_name], equal_counts, component)
            _, back_curve = component_curve(LINEAR_GROUPS[back_name], equal_counts, component)
            central[component] = common_weight(q, layer_curve, entry_curve, back_curve)

            values = np.empty(N_BOOT)
            for ib in range(N_BOOT):
                lc = COUNTS[SHAPE_DRAWS[layer_name][ib]]
                ec = COUNTS[SHAPE_DRAWS[entry_name][ib]]
                bc = COUNTS[SHAPE_DRAWS[back_name][ib]]
                _, lv = component_curve(LINEAR_GROUPS[layer_name], lc, component)
                _, ev = component_curve(LINEAR_GROUPS[entry_name], ec, component)
                _, bv = component_curve(LINEAR_GROUPS[back_name], bc, component)
                values[ib] = common_weight(q, lv, ev, bv)
            samples[component] = values

        delta = samples["P"] - samples["B"]
        WEIGHT_BOOT[(order, layer.z)] = samples | {"delta": delta}
        WEIGHT_ROWS.append({
            "order": order,
            "d_over_ls": layer.m_s,
            "w_B": central["B"],
            "w_B_ci_low": np.nanquantile(samples["B"], 0.025),
            "w_B_ci_high": np.nanquantile(samples["B"], 0.975),
            "w_P": central["P"],
            "w_P_ci_low": np.nanquantile(samples["P"], 0.025),
            "w_P_ci_high": np.nanquantile(samples["P"], 0.975),
            "delta_w_P_minus_B": central["P"] - central["B"],
            "delta_ci_low": np.nanquantile(delta, 0.025),
            "delta_ci_high": np.nanquantile(delta, 0.975),
            "bootstrap_probability_delta_le_zero": float(np.nanmean(delta <= 0.0)),
        })


WEIGHT_SUMMARY = pd.DataFrame(WEIGHT_ROWS)
WEIGHT_SUMMARY.to_csv(OUTDIR / "bootstrap_component_weights.csv", index=False)


apply(width_frac=1.0)
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.40 * TEXTWIDTH_IN), sharey=True)
for ax, order, panel in zip(axes, ORDERS, ("a", "b")):
    sub = WEIGHT_SUMMARY[WEIGHT_SUMMARY["order"] == order].sort_values("d_over_ls")
    sub = sub[sub["d_over_ls"] >= 0.099]
    for component, marker, color in (("B", "o", COL[0]), ("P", "s", COL[1])):
        y = sub[f"w_{component}"].to_numpy()
        low = sub[f"w_{component}_ci_low"].to_numpy()
        high = sub[f"w_{component}_ci_high"].to_numpy()
        ax.errorbar(
            sub["d_over_ls"], y, yerr=np.vstack([y - low, high - y]),
            marker=marker, ms=3.7, lw=0.85, capsize=1.7, color=color,
            label=(r"background $B$" if component == "B" else r"interference $P=C-B$"),
        )
    ax.axhspan(0.0, 1.0, color="0.94", zorder=-5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$d/\ell_s$")
    ax.set_title(f"({panel}) {order}", loc="left", fontsize="small")
    ax.grid(alpha=0.16, which="both")
axes[0].set_ylabel("entry-layer interpolation weight")
axes[0].legend(fontsize=6, frameon=False)
fig.savefig(OUTDIR / "F7_component_depth_weights.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "F7_component_depth_weights.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Replica-bootstrap propagation for late-time substrate recovery
# ---------------------------------------------------------------------------


CIRCULAR_GROUPS = {}
for order in ORDERS:
    c = CAMP[(order, "circular")]
    CIRCULAR_GROUPS[(order, "entry")] = group_arrays(c, c.ctrl["entry"])
    for layer in c.layers:
        if 0.099 <= layer.m_s <= 0.501:
            CIRCULAR_GROUPS[(order, layer.z)] = group_arrays(c, layer.keys)


def width_curve_for_counts(c, group, counts):
    q, coherent, background = group
    csum = np.tensordot(counts, coherent, axes=(0, 0))
    bsum = np.tensordot(counts, background, axes=(0, 0))
    times, widths = [], []
    for time_index in range(1, c.n_time + 1):
        profile = ratio_profile(q, csum[time_index], bsum[time_index])
        apex = apex_value(q, profile)
        tail = profile[q > CONV["q_view"]]
        noise = float(np.std(tail)) if tail.size > 5 else np.inf
        if not np.isfinite(apex) or (apex - 1.0) < CONV["snr_min_time"] * noise:
            continue
        width = level_q(q, profile, 0.5)
        if np.isfinite(width) and width > 0.0:
            times.append((time_index - 0.5) * c.dt_sim / c.tau_entry)
            widths.append(width)
    return np.asarray(times), np.asarray(widths)


def late_recovery(c, layer_curve, entry_curve, fraction):
    td, qd = layer_curve
    te, qe = entry_curve
    if min(len(td), len(te)) < 8:
        return np.nan
    low, high = max(td.min(), te.min()), min(td.max(), te.max())
    if not high > low:
        return np.nan
    grid = np.geomspace(low, high, 55)
    ratio = np.interp(grid, td, qd) / np.interp(grid, te, qe)
    nlate = max(5, int(np.ceil(fraction * len(ratio))))
    plateau = float(np.nanmedian(ratio[-nlate:]))
    return float(c.l_entry / plateau**2) if plateau > 0.0 else np.nan


WIDTH_LOOKUP = {}
for name, group in CIRCULAR_GROUPS.items():
    c = CAMP[(name[0], "circular")]
    WIDTH_LOOKUP[name] = [width_curve_for_counts(c, group, counts) for counts in COUNTS]

TIME_DRAWS = {name: bootstrap_count_indices() for name in CIRCULAR_GROUPS}
TIME_ROWS = []
POOL_ROWS = []
TIME_BOOT = {}

for order in ORDERS:
    c = CAMP[(order, "circular")]
    selected = [layer for layer in c.layers if 0.099 <= layer.m_s <= 0.501]
    central_entry = WIDTH_LOOKUP[(order, "entry")][COUNT_INDEX[tuple(equal_counts)]]
    per_layer_boot = {}

    for layer in selected:
        name = (order, layer.z)
        central_layer = WIDTH_LOOKUP[name][COUNT_INDEX[tuple(equal_counts)]]
        central = {
            fraction: late_recovery(c, central_layer, central_entry, fraction)
            for fraction in PLATEAU_FRACTIONS
        }
        samples = {fraction: np.empty(N_BOOT) for fraction in PLATEAU_FRACTIONS}
        for ib in range(N_BOOT):
            layer_curve = WIDTH_LOOKUP[name][TIME_DRAWS[name][ib]]
            entry_curve = WIDTH_LOOKUP[(order, "entry")][TIME_DRAWS[(order, "entry")][ib]]
            for fraction in PLATEAU_FRACTIONS:
                samples[fraction][ib] = late_recovery(c, layer_curve, entry_curve, fraction)
        per_layer_boot[layer.z] = samples
        primary = samples[0.20]
        TIME_ROWS.append({
            "order": order,
            "d_over_ls": layer.m_s,
            "l_star_substrate_true": c.l_back,
            "l_star_substrate_hat": central[0.20],
            "ci_low": np.nanquantile(primary, 0.025),
            "ci_high": np.nanquantile(primary, 0.975),
            "relative_error": central[0.20] / c.l_back - 1.0,
            "valid_bootstrap_fraction": float(np.mean(np.isfinite(primary))),
            "hat_plateau_15pct": central[0.15],
            "hat_plateau_25pct": central[0.25],
        })

    pooled = {}
    central_pooled = {}
    for fraction in PLATEAU_FRACTIONS:
        matrix = np.vstack([per_layer_boot[layer.z][fraction] for layer in selected])
        pooled[fraction] = np.nanmedian(matrix, axis=0)
        central_values = [
            late_recovery(
                c,
                WIDTH_LOOKUP[(order, layer.z)][COUNT_INDEX[tuple(equal_counts)]],
                central_entry,
                fraction,
            )
            for layer in selected
        ]
        central_pooled[fraction] = float(np.nanmedian(central_values))

    TIME_BOOT[order] = pooled
    primary = pooled[0.20]
    POOL_ROWS.append({
        "order": order,
        "n_thicknesses": len(selected),
        "l_star_substrate_true": c.l_back,
        "pooled_hat": central_pooled[0.20],
        "pooled_ci_low": np.nanquantile(primary, 0.025),
        "pooled_ci_high": np.nanquantile(primary, 0.975),
        "pooled_relative_error": central_pooled[0.20] / c.l_back - 1.0,
        "pooled_hat_plateau_15pct": central_pooled[0.15],
        "pooled_hat_plateau_25pct": central_pooled[0.25],
        "plateau_window_relative_range": (
            max(central_pooled.values()) - min(central_pooled.values())
        ) / central_pooled[0.20],
    })


TIME_SUMMARY = pd.DataFrame(TIME_ROWS)
POOL_SUMMARY = pd.DataFrame(POOL_ROWS)
TIME_SUMMARY.to_csv(OUTDIR / "bootstrap_temporal_recovery_by_thickness.csv", index=False)
POOL_SUMMARY.to_csv(OUTDIR / "bootstrap_temporal_recovery_pooled.csv", index=False)


apply(width_frac=1.0)
fig, axes = plt.subplots(
    1, 3, figsize=(TEXTWIDTH_IN, 0.39 * TEXTWIDTH_IN),
    gridspec_kw={"wspace": 0.38, "width_ratios": (1.0, 1.0, 1.15)},
)

for ax, order, panel in zip(axes[:2], ORDERS, ("a", "b")):
    c = CAMP[(order, "circular")]
    selected = [layer for layer in c.layers if 0.099 <= layer.m_s <= 0.501]
    entry_curve = WIDTH_LOOKUP[(order, "entry")][COUNT_INDEX[tuple(equal_counts)]]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(selected)))
    for color, layer in zip(colors, selected):
        layer_curve = WIDTH_LOOKUP[(order, layer.z)][COUNT_INDEX[tuple(equal_counts)]]
        td, qd = layer_curve
        te, qe = entry_curve
        low, high = max(td.min(), te.min()), min(td.max(), te.max())
        grid = np.geomspace(low, high, 55)
        ratio = np.interp(grid, td, qd) / np.interp(grid, te, qe)
        ax.plot(grid, ratio, color=color, lw=0.85, label=rf"$d/\ell_s={layer.m_s:.2f}$")
    ax.axhline(np.sqrt(c.l_entry / c.l_back), color="black", ls="--", lw=0.9,
               label="substrate prediction")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t/\tau^*_{\rm entry}$")
    ax.set_ylabel(r"$q_{1/2}^{d}/q_{1/2}^{\rm entry}$")
    ax.set_title(f"({panel}) {order}", loc="left", fontsize="small")
    ax.grid(alpha=0.16, which="both")
    ax.legend(fontsize=5.2, frameon=False)

ax = axes[2]
xbase = np.arange(len(ORDERS), dtype=float)
for io, order in enumerate(ORDERS):
    st = STYLE[order]
    sub = TIME_SUMMARY[TIME_SUMMARY["order"] == order].sort_values("d_over_ls")
    offsets = np.linspace(-0.15, 0.15, len(sub))
    for offset, (_, row) in zip(offsets, sub.iterrows()):
        ax.errorbar(
            xbase[io] + offset,
            row["l_star_substrate_hat"],
            yerr=[[row["l_star_substrate_hat"] - row["ci_low"]],
                  [row["ci_high"] - row["l_star_substrate_hat"]]],
            fmt="o", color=st["color"], alpha=0.55, ms=3.2, capsize=1.7, lw=0.7,
        )
    pool = POOL_SUMMARY[POOL_SUMMARY["order"] == order].iloc[0]
    ax.errorbar(
        xbase[io], pool["pooled_hat"],
        yerr=[[pool["pooled_hat"] - pool["pooled_ci_low"]],
              [pool["pooled_ci_high"] - pool["pooled_hat"]]],
        fmt=st["marker"], color=st["color"], ms=6, capsize=3, lw=1.2,
        label=f"{order}: pooled",
    )
    ax.hlines(pool["l_star_substrate_true"], io - 0.30, io + 0.30,
              color=st["color"], ls="--", lw=0.9)
ax.set_xticks(xbase, ORDERS)
ax.tick_params(axis="x", labelsize=7)
ax.set_ylabel(r"recovered $\ell^*_{\rm substrate}$ [$\mu$m]")
ax.set_title("(c) substrate recovery", loc="left", fontsize="small")
ax.grid(axis="y", alpha=0.18)
fig.savefig(OUTDIR / "F8_temporal_recovery_bootstrap.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "F8_temporal_recovery_bootstrap.png", dpi=180, bbox_inches="tight")
plt.close(fig)


print("\nStationary single-scale bootstrap")
print(SHAPE_SUMMARY.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print("\nAbsolute-component depth weights")
print(WEIGHT_SUMMARY.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print("\nTemporal recovery by thickness")
print(TIME_SUMMARY.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print("\nPooled temporal recovery")
print(POOL_SUMMARY.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print(f"\nArtifacts written to {OUTDIR.resolve()}")
