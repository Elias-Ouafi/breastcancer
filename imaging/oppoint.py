"""Publish the operating point of the exam-level decision head.

`imaging.examclf` cross-validates and reports an AUC. An AUC is not a decision: a tool
that answers "is there a cancer in this exam?" answers at **one threshold**, and the
pair it must be judged on is the one the national screening programme publishes --
sensitivity 82.8 %, specificity 91.4 % (DOCUMENTATION.md, "Cible chiffrée"). This module turns
the stored out-of-fold predictions into that pair, with its confidence intervals and
the prevalence PPV was measured at.

It reads `cv_predictions.csv` rather than re-running anything: the scores are already
out-of-fold, re-training to re-derive them would burn hours and, worse, would let a
threshold be retried until the number looked better.

**The threshold is chosen out of fold too.** Picking it on the same 272 scores it is
then scored on reports how well a rule fitted to these patients describes these
patients. Each fold's threshold therefore comes from the other four folds only -- the
same discipline one level down, and the reason `--naive` exists: to print both and let
the gap be read rather than asserted.

    python -m imaging.oppoint
    python -m imaging.oppoint --predictions models/examclf/relative/cv_predictions.csv

Writes `reports/examclf_operating_point.json` (every number, machine-readable) and
prints the same figures as a Markdown section. That section is not written to a
file of its own: the project keeps one documentation file, and the published copy
lives in DOCUMENTATION.md (§4.11), pasted from this output.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np

try:  # allow both "python -m imaging.oppoint" and direct execution
    from .metrics import (
        bootstrap_auc,
        bootstrap_operating_point,
        operating_point,
        threshold_for_sensitivity,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from metrics import (
        bootstrap_auc,
        bootstrap_operating_point,
        operating_point,
        threshold_for_sensitivity,
    )

import config  # noqa: E402 - repo root, importable under both invocations
from lineage import relative_path  # noqa: E402
from logging_setup import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

# Santé publique France, organised screening 50-74. Not a round number chosen here:
# see DOCUMENTATION.md, "Cible chiffrée et voie retenue".
TARGET_SENSITIVITY = 0.828
TARGET_SPECIFICITY = 0.914

DEFAULT_PREDICTIONS = os.path.join(config.MODELS_DIR, "examclf", "cv_predictions.csv")
DEFAULT_OUTPUT_DIR = config.REPORTS_DIR
REQUIRED_COLUMNS = ("patient", "label", "score", "fold")


def read_predictions(path):
    """Read a `cv_predictions.csv` into ``(patients, labels, scores, folds)``.

    A missing column is refused rather than defaulted: a file without ``fold`` cannot
    support an out-of-fold threshold, and silently falling back to the naive one would
    publish the optimistic number under the honest one's name.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} holds no prediction rows.")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; found {list(rows[0])}.")

    patients = [r["patient"] for r in rows]
    if len(set(patients)) != len(patients):
        raise ValueError(f"{path} names a patient twice; one row is one patient here.")

    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])
    return patients, labels, scores, folds


def out_of_fold_decisions(labels, scores, folds, target):
    """Decide each fold with a threshold taken from the *other* folds.

    Returns ``(decisions, thresholds)``. A fold whose complement holds no positive
    gets no threshold and no positive call -- it is counted, not guessed at.
    """
    decisions = np.zeros(labels.size, dtype=bool)
    thresholds = {}
    for fold in sorted(set(int(f) for f in folds)):
        held = folds == fold
        thr = threshold_for_sensitivity(labels[~held], scores[~held], target)
        thresholds[fold] = thr
        if np.isnan(thr):
            log.warning(f"Fold {fold}: no positive outside it, no threshold, no positive call.")
            continue
        decisions[held] = scores[held] >= thr
    return decisions, thresholds


def summarise(labels, decisions):
    """The counts and rates of one set of decisions, without a threshold to quote."""
    labels = np.asarray(labels) > 0
    decisions = np.asarray(decisions) > 0
    tp = int((decisions & labels).sum())
    fp = int((decisions & ~labels).sum())
    tn = int((~decisions & ~labels).sum())
    fn = int((~decisions & labels).sum())

    def ratio(num, den):
        return float(num / den) if den else float("nan")

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, labels.size),
        "prevalence": ratio(tp + fn, labels.size),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def build_report(labels, scores, folds, target=TARGET_SENSITIVITY, n_resamples=10000, seed=42):
    """Everything the published table needs, as one dict."""
    auc = bootstrap_auc(labels, scores, n_resamples=n_resamples, seed=seed)

    decisions, thresholds = out_of_fold_decisions(labels, scores, folds, target)
    honest = summarise(labels, decisions)
    honest["bootstrap"] = bootstrap_operating_point(
        labels, decisions, n_resamples=n_resamples, seed=seed)
    honest["thresholds_by_fold"] = {str(k): v for k, v in thresholds.items()}

    naive_threshold = threshold_for_sensitivity(labels, scores, target)
    naive = operating_point(labels, scores, naive_threshold)

    # What a coin flip reaches at the sensitivity this model actually achieved. A
    # chance classifier trades one for the other exactly: specificity = 1 - sensitivity.
    # Quoting it here is what stops "20 % specificity" from sounding like a weak result
    # rather than what it is.
    chance_specificity = 1.0 - honest["sensitivity"]

    return {
        "target": {"sensitivity": target, "specificity": TARGET_SPECIFICITY,
                   "source": "Santé publique France, dépistage organisé 50-74"},
        "corpus": {"patients": int(labels.size), "cancers": int((labels > 0).sum()),
                   "prevalence": float((labels > 0).mean()),
                   "folds": sorted(set(int(f) for f in folds))},
        "auc": auc,
        "out_of_fold_threshold": honest,
        "naive_threshold": naive,
        "chance_reference": {
            "specificity_at_achieved_sensitivity": float(chance_specificity),
            "beats_chance": bool(honest["specificity"] > chance_specificity),
        },
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def _pct(x):
    """A percentage in the document's own language: comma decimal, non-breaking space."""
    return "n/a" if x != x else f"{100 * x:.1f}".replace(".", ",") + " %"


def _num(x, digits=3):
    return "n/a" if x != x else f"{x:.{digits}f}".replace(".", ",")


def render_markdown(report, predictions_path):
    """The report as a table meant to be read, not parsed."""
    corpus, honest = report["corpus"], report["out_of_fold_threshold"]
    boot, auc = honest["bootstrap"], report["auc"]
    naive, chance = report["naive_threshold"], report["chance_reference"]
    c = honest["counts"]

    def ci(key):
        b = boot[key]
        return "n/a" if b["lo"] != b["lo"] else f"{_pct(b['lo'])} – {_pct(b['hi'])}"

    verdict = (
        "La VPP est **au niveau de la prévalence** : savoir que le modèle a répondu "
        "« cancer » ne change pas la probabilité qu'il y en ait un."
        if abs(honest["ppv"] - honest["prevalence"]) < 0.02 else
        f"VPP {_pct(honest['ppv'])} contre une prévalence de {_pct(honest['prevalence'])}."
    )

    return f"""#### Rapport de point de fonctionnement — tête de décision au niveau examen

> **Research Use Only — Not for diagnostic use.**

Généré par `python -m imaging.oppoint` depuis `{predictions_path}`. Ne réentraîne rien : les scores sont ceux de la validation croisée d'`imaging.examclf`.

**Corpus** : {corpus['patients']} patients, {corpus['cancers']} cancers, prévalence {_pct(corpus['prevalence'])}, {len(corpus['folds'])} plis.

**ROC-AUC patient** : {_num(auc['auc'])} [{_num(auc['lo'])} – {_num(auc['hi'])}]

##### Au seuil visé (sensibilité {_pct(report['target']['sensitivity'])})

Seuil pris **hors du pli noté** : chaque patient est jugé par un seuil calé sur les
quatre autres plis, jamais sur le sien.

| Mesure | Valeur | IC 95 % | Cible |
|---|---:|---|---:|
| Sensibilité | {_pct(honest['sensitivity'])} | {ci('sensitivity')} | {_pct(report['target']['sensitivity'])} |
| Spécificité | {_pct(honest['specificity'])} | {ci('specificity')} | {_pct(report['target']['specificity'])} |
| VPP | {_pct(honest['ppv'])} | {ci('ppv')} | — |
| NPV | {_pct(honest['npv'])} | {ci('npv')} | — |
| Prévalence du jeu | {_pct(honest['prevalence'])} | — | — |

TP {c['tp']} · FP {c['fp']} · TN {c['tn']} · FN {c['fn']}

##### Ce que ces chiffres disent

{verdict}

À la sensibilité réellement atteinte ({_pct(honest['sensitivity'])}), **le hasard donnerait {_pct(chance['specificity_at_achieved_sensitivity'])} de spécificité** — un classifieur aléatoire échange l'une contre l'autre exactement. Le modèle en donne {_pct(honest['specificity'])} : {'au-dessus' if chance['beats_chance'] else '**en dessous**'}.

La cible de sensibilité n'est pas atteinte ({_pct(honest['sensitivity'])} contre {_pct(report['target']['sensitivity'])}) : le seuil calé sur quatre plis ne transporte pas jusqu'au cinquième, ce qui est en soi une mesure — celle d'un score dont l'échelle ne veut rien dire d'un groupe de patients à l'autre (§4.9).

##### Seuil naïf, pour comparaison

Calé sur les scores mêmes qu'il note ensuite — publié pour que l'écart soit lisible, pas pour être cité : sensibilité {_pct(naive['sensitivity'])}, spécificité {_pct(naive['specificity'])}, VPP {_pct(naive['ppv'])}, seuil {_num(naive['threshold'], 4)}.

L'écart ne raconte pas l'histoire habituelle de l'optimisme, et il faut le dire : quand un modèle est au niveau du hasard, il n'y a rien à sur-estimer.
"""


def write_report(report, output_dir, predictions_path):
    """Write the JSON report. The source path is stored repo-relative -- a versioned report
    should not name someone's home directory (the rule ``lineage.relative_path`` already
    applies to manifests)."""
    predictions_path = relative_path(predictions_path)
    report = dict(report, predictions=predictions_path)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "examclf_operating_point.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    return json_path


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", default=DEFAULT_PREDICTIONS,
                   help="cv_predictions.csv written by imaging.examclf.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-sensitivity", type=float, default=TARGET_SENSITIVITY,
                   help="The sensitivity the threshold is fixed at, not an accuracy to "
                        "maximise (DOCUMENTATION.md, 'Cible chiffrée').")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    setup_logging(logfile="oppoint.log")
    args = build_arg_parser().parse_args(argv)

    _, labels, scores, folds = read_predictions(args.predictions)
    report = build_report(labels, scores, folds, target=args.target_sensitivity,
                          n_resamples=args.bootstrap, seed=args.seed)
    honest = report["out_of_fold_threshold"]
    log.info(f"{report['corpus']['patients']} patients, {report['corpus']['cancers']} cancers, "
             f"AUC {report['auc']['auc']:.3f} [{report['auc']['lo']:.3f}-{report['auc']['hi']:.3f}]")
    log.info(f"Out-of-fold threshold: sensitivity {honest['sensitivity']:.3f}, "
             f"specificity {honest['specificity']:.3f}, ppv {honest['ppv']:.3f}, "
             f"prevalence {honest['prevalence']:.3f}")
    if not report["chance_reference"]["beats_chance"]:
        log.warning("Specificity is at or below what chance gives at this sensitivity.")

    json_path = write_report(report, args.output_dir, args.predictions)
    log.info(f"Wrote {json_path}; the section below goes in DOCUMENTATION.md §4.11")
    if hasattr(sys.stdout, "reconfigure"):  # French text on a cp1252 Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    print(render_markdown(report, relative_path(args.predictions)))
    return report


if __name__ == "__main__":
    main()
