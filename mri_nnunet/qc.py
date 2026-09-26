"""Quality-control report: what the pipeline did to ten random cases, and to the dataset.

    python -m mri_nnunet qc --n 10

For each sampled case, one PNG with the intensity histogram **before** (the native scan) and
**after** (normalised inside the organ mask), and three orthogonal slices through the lesion
with the pseudo-mask overlaid. Then the aggregate statistics of every built case
(``aggregate.json``, plus a histogram figure): spacing, shape, lesion volume, the
box-to-pseudo-mask ratio, the enhancement contrast, the registration transforms and the anomalies.

Nothing here is a verdict. A picture that looks wrong is the reason to look at the case; the
report is what makes looking cheap.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import pipeline, registry


def _sample(cases, n, seed):
    rng = np.random.default_rng(seed)
    cases = sorted(cases)
    return sorted(rng.choice(cases, size=min(n, len(cases)), replace=False).tolist())


def aggregate(silver_dir, cases):
    """Dataset-level statistics from the per-case summaries."""
    rows = []
    for pid in cases:
        record = pipeline._load_json(os.path.join(silver_dir, "cases", pid, "case.json"))
        rows.append({"patient_id": pid, **record["summary"]})

    def column(key, default=np.nan):
        return np.array([r.get(key) if r.get(key) is not None else default for r in rows], dtype=float)

    def describe(values):
        values = values[~np.isnan(values)]
        if values.size == 0:
            return None
        return {"n": int(values.size), "min": float(values.min()), "p05": float(np.percentile(values, 5)),
                "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "max": float(values.max())}

    registrations = [v for r in rows for v in r.get("registration", {}).values() if v]
    registry_rows = registry.read_csv(os.path.join(silver_dir, "registry.csv"))
    scanners = {}
    for row in registry_rows:
        key = f"{row['scanner']} {row['field_strength_t']}T".strip()
        scanners[key] = scanners.get(key, 0) + 1
    return {
        "n_cases": len(rows),
        "n_excluded": len(registry.read_csv(os.path.join(silver_dir, "exclusions.csv"))),
        "lesion_volume_mm3": describe(column("lesion_volume_mm3")),
        "smallest_lesion_axis_voxels": describe(column("smallest_lesion_axis_voxels")),
        "box_to_pseudo_ratio": describe(column("box_to_pseudo_ratio")),
        "organ_mask_fraction_of_fov": describe(column("organ_mask_fraction_of_fov")),
        "lesion_tissue_erased": describe(column("lesion_tissue_erased")),
        "lesion_in_air": describe(column("lesion_in_air")),
        "contrast": describe(column("contrast")),
        "registration_translation_mm": describe(np.array([v.get("translation_mm", np.nan) for v in registrations], float)),
        "registration_rotation_deg": describe(np.array([v.get("rotation_deg", np.nan) for v in registrations], float)),
        "registrations_refused": int(sum(1 for v in registrations if v.get("accepted") is False)),
        "cases_with_anomalies": int(sum(1 for r in rows if r.get("n_anomalies", 0) > 0)),
        "scanners": scanners,
        "_rows": rows,
    }


def run(silver_dir, out_dir, settings, n=10, seed=None):
    """Write the report for ``n`` random built cases (seeded) and return its summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import sitk_io

    seed = settings["seed"] if seed is None else seed
    cases = sorted(pipeline.held_cases(silver_dir, settings))
    if not cases:
        raise FileNotFoundError(f"no built case under {silver_dir}")
    os.makedirs(out_dir, exist_ok=True)
    chosen = _sample(cases, n, seed)
    channels = settings["channels"]
    # A rebuild that changes the set of built cases changes the draw: a figure of a case no
    # longer sampled would otherwise sit beside the new ones and read as part of this report.
    for name in os.listdir(out_dir):
        if name.endswith(".png") and name != "aggregate.png" and name[:-4] not in chosen:
            os.remove(os.path.join(out_dir, name))

    for pid in chosen:
        native, case_dir = pipeline._paths(silver_dir, pid)
        outputs = pipeline._outputs(settings, silver_dir, pid)
        label = sitk_io.to_array(sitk_io.read_nifti(outputs[-2]))
        organ = sitk_io.to_array(sitk_io.read_nifti(outputs[-1]))
        after = [sitk_io.to_array(sitk_io.read_nifti(outputs[i])) for i in range(len(channels))]
        before = [sitk_io.to_array(sitk_io.read_nifti(os.path.join(native, f"{pid}_ph{c['phase']}.nii.gz")))
                  for c in channels]

        fig, axes = plt.subplots(len(channels) + 1, 3, figsize=(13, 4 * (len(channels) + 1)))
        for i, c in enumerate(channels):
            ax = axes[i, 0]
            ax.hist(before[i].ravel(), bins=200, color="tab:gray", alpha=0.8, log=True)
            ax.set_title(f"{c['name']}: native intensities (log counts)")
            ax = axes[i, 1]
            ax.hist(after[i][organ > 0].ravel(), bins=200, color="tab:blue", alpha=0.8, log=True)
            inside = after[i][organ > 0]
            ax.set_title(f"{c['name']}: after, inside the mask (mean {inside.mean():.2f}, std {inside.std():.2f})")
            axes[i, 2].axis("off")

        centre = np.round(np.argwhere(label > 0).mean(axis=0)).astype(int)
        image = after[-1]
        views = [("axial", image[centre[0]], label[centre[0]]),
                 ("coronal", image[:, centre[1]], label[:, centre[1]]),
                 ("sagittal", image[:, :, centre[2]], label[:, :, centre[2]])]
        for ax, (title, slice_, mask_) in zip(axes[-1], views):
            ax.imshow(slice_, cmap="gray", origin="lower")
            overlay = np.ma.masked_where(mask_ == 0, mask_)
            ax.imshow(overlay, cmap="autumn", alpha=0.45, origin="lower")
            ax.set_title(f"{pid} {title}, {channels[-1]['name']} + pseudo-mask")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{pid}.png"), dpi=90)
        plt.close(fig)

    stats = aggregate(silver_dir, cases)
    rows = stats.pop("_rows")
    with open(os.path.join(out_dir, "aggregate.json"), "w", encoding="utf-8") as handle:
        json.dump({"sampled": chosen, "seed": seed, **stats}, handle, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(axes, ("lesion_volume_mm3", "box_to_pseudo_ratio", "contrast"),
                              ("lesion volume (mm3)", "box / pseudo-mask volume",
                               "pseudo-mask contrast (sd above the organ)")):
        values = np.array([r[key] for r in rows if r.get(key) is not None], float)
        ax.hist(values[~np.isnan(values)], bins=30, color="tab:blue")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aggregate.png"), dpi=90)
    plt.close(fig)
    return {"sampled": chosen, **stats}
