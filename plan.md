# plan.md — De « breastcancer » à un MVP démontrable devant investisseurs

> **Modalité** : IRM mammaire multiphase (DCE-MRI, DICOM).
> **Cible du MVP** : démo investisseurs / pitch. **Pas d'usage clinique, pas de certification** à ce stade.
> **Mention obligatoire, partout** : *Research Use Only — Not for diagnostic use*.
> **Lecture** : ~10 min. Ce document est la référence partagée (tech + produit + design).

---

## Synthèse — où en est-on (mise à jour du 2026-07-21)

Ce document a été rédigé sur l'hypothèse initiale « pas d'accès au code ». En pratique, l'agent a
obtenu un accès réel au dépôt et a exécuté le plan plutôt que de simplement l'écrire : téléchargement
et préprocessing de données DCE-MRI réelles, premier entraînement, branchement dans l'app de démo,
overlay visuel, charte graphique. Détail complet dans la grille d'audit (§1.1, colonnes mises à jour
en conditions réelles) et dans **Partie 4 — Prochaines étapes vers le MVP**.

**Fait, vérifié** : pipeline DCE-MRI bout-en-bout sur un échantillon de 61 patients (Duke-Breast-Cancer-MRI,
20 Go), premier modèle entraîné (Dice test 0,859), branché dans l'app Flask avec overlay visuel et
charte graphique appliquée.

**Pas encore fait** : échantillon trop petit pour un chiffre présentable en pitch, pas de test navigateur
interactif réel (seulement API/HTML via requêtes directes), 3 cas de démo garantis pas encore figés,
nom de produit non tranché. Détail priorisé en **Partie 4**.

---

## Partie 1 — État des lieux (audit actionnable)

Grille remplie à l'origine via un script d'inventaire (§1.2, toujours utile pour un futur audit à froid
sur un autre poste), puis mise à jour en continu à partir d'un accès réel au dépôt (voir statuts ✅/⚠️/❌
et colonne Preuve ci-dessous, datés).

### 1.1 Grille d'audit

Statut : ✅ présent & sain · ⚠️ présent mais à durcir · ❌ absent · ❓ à vérifier.
Grille remplie via audit direct du dépôt le 2026-07-20 (accès réel obtenu en session, contrairement à l'hypothèse initiale « pas d'accès »).

| # | Axe | Question de contrôle | Statut | Preuve | Priorité |
|---|-----|----------------------|:------:|--------|:--------:|
| A1 | Données | Un jeu DCE-MRI DICOM identifié, avec ≥3 phases (pré + post-contraste) ? | ✅ | Collection TCIA **Duke-Breast-Cancer-MRI** (922 patients au total). Échantillon élargi à **20 Go** dans `tciaDownload/duke_mri/` (267 séries, patients Breast_MRI_001 à 064) via `ExtractData.download_dce_mri_series`. Protocole confirmé : `ax dyn pre` + `1st`..`4th pass` (3-4 phases post-contraste selon le patient). | P0 |
| A2 | Données | Séries correctement triées (SeriesInstanceUID, temporalité des phases) ? | ✅ | `TransformData.group_dce_series_by_patient` + `_dce_phase_rank` : **62/62** patients de l'échantillon élargi correctement regroupés, **61/62** avec paire pré/post exploitable. Gère les deux conventions de nommage de la collection (`ax dyn pre/1st pass...` vs `ax 3d dyn` bare + `Ph1.../Ph4...`). | P0 |
| A3 | Données | Nombre de patients / volumes réellement exploitables ? | ✅ | **61 patients préprocessés avec succès, 1 seul échec** (`preprocessed_data_mri/`, 33 Mo, masques de lésion réels depuis `Annotation_Boxes.xlsx`). Échantillon suffisant pour un premier split train/val/test significatif (Jalon 2). Reste 922−64 ≈ 858 patients non téléchargés si besoin d'agrandir davantage plus tard. | P0 |
| A4 | Anonymisation | Les DICOM sont-ils dé-identifiés (tags PHI, burned-in pixels) ? | ✅ | Spot-check `pydicom` réel sur 15 séries MRI téléchargées : aucun `PatientBirthDate`, aucun `BurnedInAnnotation=YES`, aucun nom d'institution/médecin résiduel. `PatientName` = identifiant de recherche pseudonymisé (`Breast_MRI_00X`), identique au `PatientID` — schéma standard TCIA, pas une fuite. BCS-DBT (DBT) déjà confirmé pré-anonymisé par la source. | P0 |
| A5 | Anonymisation | Traçabilité du consentement / base légale RGPD documentée ? | ⚠️ | Base légale = données publiques de recherche (TCIA, licence CC). À documenter explicitement dans le registre de traitement (aucune collecte primaire de données patient par le projet). | P0 |
| B1 | Préproc | `TransformData.py` : quelles transfos ? (resampling, N4, recalage, normalisation) | ⚠️ | Nouvelle fonction `preprocess_dce_mri_with_boxes` écrite et le chargement multiphase validé sur données réelles (10 patients). Pas de recalage inter-phase pour l'instant (acquisitions Duke faites sans repositionnement patient — hypothèse à vérifier visuellement sur plus de cas), pas de N4 bias field (P2 si le bruit de champ s'avère un problème visible). L'ancien `preprocess_mri_data` (SEG/RTSTRUCT) reste inutilisé — Duke fournit des boxes, pas des SEG pour la majorité des patients. | P0 |
| B2 | Préproc | Sortie déterministe et versionnée (mêmes entrées → mêmes sorties) ? | ✅ | `save_preprocessed` déterministe, seed fixe (`--seed 42` côté train). | P1 |
| B3 | Préproc | Gestion multiphase / soustraction (post − pré) implémentée ? | ✅ | Implémentée et validée sur `Breast_MRI_001` réel : soustraction post−pré cohérente physiologiquement (réhaussement croissant avec les passes), normalisée (`normalize_intensity`), compatible telle quelle avec `imaging/dataset.py`/`train.py` (volume mono-canal). | P1 |
| C1 | Modèle | `imaging/train.py` : tâche réelle (segmentation ? classification ?) | ✅ | Segmentation/localisation de lésion (U-Net 2D, perte Focal Tversky/Dice+BCE), actuellement entraîné sur boxes DBT, pas MRI. | P0 |
| C2 | Modèle | Architecture, framework, dépendances figées (`requirements.txt`) ? | ✅ | PyTorch, `segmentation-models-pytorch` (encodeur pré-entraîné optionnel), versions bornées dans `requirements.txt`. | P1 |
| C3 | Modèle | Un checkpoint entraîné existe-t-il ? Reproductible ? | ✅ (DBT) / ✅ (MRI) | `results/unet_best.pt` (DBT, 31 Mo) et `results_mri/unet_best.pt` (DCE-MRI, 31 Mo, 21/07) — premier entraînement réel sur 61 patients Duke, 30 époques. **Dice test = 0,859, IoU test = 0,763** (`results_mri/segmentation_test_metrics.csv`). Bon signal pour un premier passage ; à confirmer sur un échantillon plus large avant tout chiffre présenté aux investisseurs (61 patients = risque de surestimation). | P0 |
| C4 | Modèle | Split train/val/test **par patient** (pas de fuite) ? | ✅ | `imaging/dataset.py:split_npz_by_patient`, utilise `case_id` = PatientID réel. Réutilisable tel quel pour MRI. | P0 |
| D1 | Éval | Métriques calculées (Dice, AUC, sensibilité/spécificité) et loggées ? | ✅ | Dice + IoU loggés par epoch (`results/segmentation_metrics.csv`) + test final (`segmentation_test_metrics.csv`). | P1 |
| D2 | Éval | Résultats sauvegardés interprétables ? | ✅ | CSV clairs, exploitables tels quels pour un rapport. | P1 |
| E1 | Inférence | `inference.py` tourne sur un volume nu, bout-en-bout ? | ✅ (DBT) / ✅ (MRI) | `predict_dce_mri` ajouté (partage `_localize_lesion` avec `predict_dbt`) et branché dans l'app de démo via `DceMriUNetPredictor` (`MRI_APP_BACKEND=dce_mri`). Testé bout-en-bout réellement : app lancée, upload de 2 volumes `.npz` réels via `/api/predict`, réponses valides (lésion détectée, box, backend correct). | P0 |
| E2 | Inférence | Temps d'inférence par volume sur la machine GPU cible ? | ❓ | Non mesuré. À chronométrer une fois le modèle MRI entraîné. | P1 |
| F1 | Démo | Une UI/entrée démo existe-t-elle ? | ✅ | App Flask complète et fonctionnelle (`app/server.py`), backend `mock`/`unet` interchangeable via `MRI_APP_BACKEND`, bind loopback uniquement (127.0.0.1), upload nettoyé après scoring. | P0 |
| F2 | Démo | Rendu visuel exploitable pour un pitch (overlay lisible) ? | ✅ | `inference.render_overlay_png` + intégration dans `app/server.py`/`result.html` : la coupe détectée s'affiche en niveaux de gris avec un cadre rouge sur la zone de réhaussement, encodée en base64 (aucun fichier dérivé laissé sur disque). Vérifié visuellement sur `Breast_MRI_001` — cadre bien positionné sur la zone lumineuse. | P1 |
| G1 | Repro | README suffisant pour rejouer l'entraînement/inférence de zéro ? | ✅ | `README.md` documente le pipeline DBT bout-en-bout ; à compléter pour le futur pipeline MRI. | P2 |
| G2 | Repro | Env reproductible (venv/conda/Docker), pas de chemins en dur ? | ✅ | `.venv` présent, `requirements.txt` figé par bornes de version. Pas de Docker (P2, optionnel). | P2 |
| H1 | Conformité | Mention « Research Use Only » présente dans l'app et les sorties ? | ✅ | Déjà présent sur l'UI réelle : `app/templates/base.html` (bloc `.disclaimer`) + `app/README.md`. Rien à faire. | P0 |
| H2 | Conformité | Aucune donnée patient dans le repo git (images, CSV nominatifs) ? | ✅ | Vérifié (`git ls-files`) : seules des données publiques/tabulaires non nominatives (Wisconsin UCI) sont versionnées. `tciaDownload/`, `preprocessed_data/`, `preprocessed_data_full/`, `results_full/` correctement ignorés. | P0 |

> **Verdict global Jalon 0** (mise à jour post-implémentation) : le socle conformité/repro (H1, H2, G2, C4) est solide — rien à corriger. Les items A1–A4/B1/B3 sont désormais ✅, validés sur données réelles (10 patients Duke-Breast-Cancer-MRI téléchargés, triés, soustraits, dé-identification vérifiée). **Seul blocage restant : `Annotation_Boxes.xlsx`** (fichier d'annotations de tumeurs, distribué hors API NBIA — récupération manuelle sur la page TCIA requise). Sans lui, C3/E1 (checkpoint + inférence MRI) restent à ❌ : impossible de construire les masques de lésion, donc d'entraîner. C'est la seule action bloquante restante avant Jalon 1.

### 1.2 Script d'inventaire (à exécuter en local)

Déposer dans `tools/inventory.py` puis `python tools/inventory.py`. Aucune dépendance lourde ; `pydicom` optionnel pour l'inspection DICOM.

```python
#!/usr/bin/env python3
"""Inventaire du dépôt breastcancer — produit un état des lieux factuel.
Usage: python tools/inventory.py [--root .] [--dicom-dir <path>]
Sortie: audit_inventory.md + audit_inventory.json
"""
import argparse, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path

CODE_EXT = {".py", ".ipynb", ".yaml", ".yml", ".toml", ".cfg"}
DATA_EXT = {".dcm", ".nii", ".gz", ".npy", ".npz", ".png", ".jpg", ".csv", ".pt", ".pth", ".ckpt", ".onnx"}
PHI_HINTS = {".csv", ".xlsx", ".json"}  # à ouvrir à la main si présents

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

def scan(root: Path):
    files, big, code_lines = [], [], 0
    for p in root.rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        size = p.stat().st_size
        rel = p.relative_to(root).as_posix()
        files.append((rel, size, p.suffix.lower()))
        if size > 20_000_000:
            big.append((rel, size))
        if p.suffix.lower() == ".py":
            try:
                code_lines += sum(1 for _ in p.open("r", errors="ignore"))
            except Exception:
                pass
    return files, big, code_lines

def dicom_probe(dicom_dir: Path):
    try:
        import pydicom
    except ImportError:
        return {"pydicom": False, "note": "pip install pydicom pour l'inspection DICOM"}
    dcm = list(dicom_dir.rglob("*.dcm"))[:2000] if dicom_dir.exists() else []
    series, modalities, has_phi_pixel = {}, set(), []
    for f in dcm:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", "?")
        series[uid] = series.get(uid, 0) + 1
        modalities.add(str(getattr(ds, "Modality", "?")))
        if getattr(ds, "BurnedInAnnotation", "NO") == "YES":
            has_phi_pixel.append(f.name)
        # Tags PHI restants ?
        for tag in ("PatientName", "PatientID", "PatientBirthDate"):
            if getattr(ds, tag, "") not in ("", None):
                has_phi_pixel.append(f"{f.name}:{tag}")
                break
    return {
        "pydicom": True, "dcm_files_sampled": len(dcm),
        "series_count": len(series), "modalities": sorted(modalities),
        "phi_flags_sample": has_phi_pixel[:20],
        "phi_suspected": bool(has_phi_pixel),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dicom-dir", default="")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    files, big, code_lines = scan(root)
    by_ext = {}
    for _, size, ext in files:
        d = by_ext.setdefault(ext or "<none>", {"n": 0, "bytes": 0})
        d["n"] += 1; d["bytes"] += size

    data_on_disk = [f for f in files if f[2] in DATA_EXT]
    phi_risk = [f for f in files if f[2] in PHI_HINTS]

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "git_branch": sh("git rev-parse --abbrev-ref HEAD"),
        "git_last_commit": sh("git log -1 --oneline"),
        "git_tracked_data": [l for l in sh("git ls-files").splitlines()
                             if Path(l).suffix.lower() in DATA_EXT],  # ⚠ données versionnées = alerte
        "totals": {"files": len(files), "python_loc": code_lines},
        "by_extension": by_ext,
        "large_files_gt_20mb": big[:50],
        "data_files_on_disk": len(data_on_disk),
        "phi_risk_files": [f[0] for f in phi_risk][:50],
        "key_files_present": {
            k: (root / k).exists() for k in
            ["TransformData.py", "inference.py", "requirements.txt",
             "imaging/train.py", "README.md", "Dockerfile"]
        },
        "dicom": dicom_probe(Path(args.dicom_dir)) if args.dicom_dir else {"skipped": True},
    }

    Path("audit_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Inventaire automatique\n", f"_Généré : {report['generated']}_\n",
          f"- Branche : `{report['git_branch']}` — dernier commit : `{report['git_last_commit']}`",
          f"- Fichiers : {report['totals']['files']} — LOC Python : {report['totals']['python_loc']}",
          f"- ⚠️ Données **versionnées dans git** : {len(report['git_tracked_data'])} (doit être 0)",
          f"- Fichiers de données sur disque : {report['data_files_on_disk']}",
          f"- Fichiers à risque PHI (csv/xlsx/json) à ouvrir manuellement : {len(report['phi_risk_files'])}",
          "\n## Fichiers clés\n"]
    for k, v in report["key_files_present"].items():
        md.append(f"- {'✅' if v else '❌'} `{k}`")
    if not report["dicom"].get("skipped"):
        d = report["dicom"]
        md.append("\n## DICOM\n")
        md.append(f"- Séries échantillonnées : {d.get('series_count')} — modalités : {d.get('modalities')}")
        md.append(f"- 🚨 PHI suspecté : {d.get('phi_suspected')} (exemples : {d.get('phi_flags_sample')})")
    Path("audit_inventory.md").write_text("\n".join(md), encoding="utf-8")
    print("OK → audit_inventory.md / audit_inventory.json")
    if report["git_tracked_data"]:
        print("🚨 Des fichiers de données sont suivis par git. Purger avant tout push public.")

if __name__ == "__main__":
    sys.exit(main())
```

**Lecture des résultats** : le script alimente directement la colonne *Preuve* de la grille §1.1. Trois signaux d'arrêt immédiat : (1) données suivies par git ≠ 0 → H2 ; (2) `phi_suspected: true` → A4 ; (3) fichiers clés `❌` → réparer avant roadmap.

---

## Partie 2 — Roadmap MVP séquencée

**Hypothèse de capacité** : 1 dev/ML + design ponctuel, 1 machine GPU (≥16 Go VRAM). Estimations en **jours-homme (j)**. Priorités **P0** (bloquant démo) → **P2** (confort).

**Ligne d'arrivée (MVP)** : une app locale où l'on charge un examen DCE-MRI dé-identifié, où le modèle produit en <10 s un overlay de lésion + un score, le tout habillé de la charte, avec mention *Research Use Only* — rejouable en direct devant un investisseur.

### Jalon 0 — Socle & conformité (semaine 1)
| Tâche | Prio | Est. | Définition de « fait » (DoD) |
|-------|:---:|:---:|------|
| Exécuter l'inventaire, remplir la grille | P0 | 0.5 j | `audit_inventory.md` généré, grille §1.1 sans `❓` sur les P0 |
| Env reproductible : `requirements.txt` figé + `venv`/`conda`, README « run in 5 lines » | P0 | 1 j | `pip install -r` propre + inférence rejouée sur machine vierge |
| Pipeline de dé-identification DICOM (pydicom/**DICOM-Anonymizer** ou **Presidio Image Redactor**) | P0 | 1.5 j | 100 % des tags PHI vidés + 0 burned-in pixel sur l'échantillon |
| Purge git des données + `.gitignore` (`preprocessed_data_full/`, `results_full/`) + audit historique | P0 | 0.5 j | `git ls-files` ne renvoie aucun fichier de données |
| Bandeau « Research Use Only » (UI + entêtes de sorties) | P0 | 0.5 j | Mention visible sur chaque écran et chaque export |

**Sortie du jalon** : socle sain, RGPD-safe, reproductible.

### Jalon 1 — Pipeline data DCE-MRI fiable (semaine 2)
| Tâche | Prio | Est. | DoD |
|-------|:---:|:---:|------|
| Ingestion DICOM → volume : tri séries, ordre des phases, conversion (**dcm2niix**/**SimpleITK**) | P0 | 1.5 j | 1 examen brut → tenseur 4D (x,y,z,phase) déterministe |
| Préproc standard : resampling isotrope, **N4** bias field, recalage inter-phase (**SimpleITK/ANTs**), z-norm | P0 | 2 j | Sortie stable, visualisable, documentée dans `TransformData.py` |
| Image de soustraction (post − pré) + MIP pour l'affichage | P1 | 1 j | Soustraction correcte + rendu MIP exploitable |
| Split **par patient** train/val/test gelé (seed) | P0 | 0.5 j | Aucun patient partagé entre splits (test automatisé) |

**Sortie** : un `preprocess()` unique, testé, réutilisé à l'entraînement et à l'inférence.

### Jalon 2 — Modèle démontrable (semaines 3–4)
| Tâche | Prio | Est. | DoD |
|-------|:---:|:---:|------|
| Cadrer la tâche MVP : **segmentation de lésion** (recommandé, visuel fort) via **MONAI**/**nnU-Net** | P0 | 1 j | Tâche + métrique cible actées dans ce doc |
| Baseline entraînée (nnU-Net « out of the box » ou U-Net MONAI 3D) | P0 | 3 j | Checkpoint reproductible + Dice val loggé |
| Éval honnête : Dice, sensibilité, faux positifs/volume, courbes (**MLflow**/CSV) | P1 | 1.5 j | Rapport `results/` interprétable + 5 cas illustrés |
| Post-traitement lésion : seuillage, plus grande composante, volume estimé (ml) | P1 | 1 j | Overlay propre, volume affiché |
| Export **ONNX** + inférence <10 s/volume sur GPU cible | P1 | 1 j | Temps mesuré et loggé dans `inference.py` |

> **Garde-fou pitch** : ne pas survendre. Cadrer les chiffres comme « performance de recherche sur un jeu limité », jamais comme performance clinique. Prévoir un **cas de repli** (exemple pré-calculé) si le live échoue.

### Jalon 3 — App démo investisseurs (semaine 5)
| Tâche | Prio | Est. | DoD |
|-------|:---:|:---:|------|
| ✅ UI Flask existante : upload examen → « Analyser » → verdict + overlay | P0 | 2 j | Parcours complet cliquable en local — déjà présent (`app/`), pas besoin de Gradio/Streamlit |
| ✅ Overlay coupe annotée (cadre sur la zone détectée) | P0 | 2 j | `inference.render_overlay_png`, intégré dans `result.html`, vérifié visuellement sur données réelles. Reste en P1 : slider multi-coupes/MIP (aujourd'hui une seule coupe, la plus confiante) |
| ✅ Application des **tokens de charte** (Partie 3) : couleurs, typo, bandeau RUO | P0 | 1 j | `app/templates/base.html`/`index.html`/`result.html` migrés sur les tokens CSS §3 (Space Grotesk/Inter/IBM Plex Mono, palette sombre, `.ruo-banner`, `.btn-primary`), couleur de l'overlay alignée sur `--accent`. Vérifié : page chargée (200), classes présentes, overlay orange confirmé visuellement. Logo non fait (reste P2, nom de produit non validé). |
| 3 examens de démo dé-identifiés pré-chargés | P0 | 0.5 j | 3 cas « qui marchent » garantis à froid |
| Écran « Comment ça marche » (pipeline en 4 étapes, 1 visuel) | P1 | 1 j | Slide intégrée, compréhensible en 30 s |

**Sortie du MVP** : démo autonome, jolie, honnête, rejouable — prête pour un pitch.

### Jalon 4 — Finitions & pitch (semaine 6, P2)
Dockerisation (repro one-command) · script d'enregistrement GIF de la démo · 1-pager technique · check RGPD final (registre de traitement, DPA si données tierces). DoD : la démo tourne sur une 2ᵉ machine sans intervention.

### Chemin critique
`Anonymisation (J0) → Ingestion+Préproc (J1) → Baseline+Éval (J2) → UI+Charte (J3)`.
**Total ≈ 30–34 j-homme (~6 semaines)**. Tout P1/P2 est sacrifiable si le calendrier glisse ; les P0 ne le sont pas.

---

## Partie 3 — Brand guidelines (directement implémentables)

### 3.1 Positionnement & anti-cliché
- **Interdit** : ruban rose, dégradés « féminins » roses, cœurs, imagerie compassionnelle. Cela infantilise le sujet et sature le marché.
- **Direction retenue** : *diagnostic instrument* — rigueur scientifique, lisibilité radiologique, précision. On s'inspire du **vocabulaire de la perfusion DCE** (cinétique de rehaussement du contraste) : fonds sombres type station de lecture, une couleur froide « signal » et un accent chaud « rehaussement » emprunté aux colormaps de perfusion.
- **Nom de code produit** (à valider) : **Perfusio** / **Kinetix** / **Contra** — évoquent la dynamique du contraste, pas la maladie.
- **Ton** : sobre, factuel, jamais alarmiste. Toujours accompagné de *Research Use Only — Not for diagnostic use*.

### 3.2 Palette (hex)
Pensée pour un fond sombre (contexte imagerie) avec pendant clair pour les documents.

| Rôle | Token | Hex | Usage |
|------|-------|-----|-------|
| Fond principal (sombre) | `--bg` | `#0B0F14` | Canvas app / viewer |
| Surface | `--surface` | `#141A22` | Cartes, panneaux |
| Surface haute | `--surface-2` | `#1E2733` | Modales, hover |
| Bordure | `--border` | `#2A3644` | Séparateurs |
| Texte principal | `--text` | `#E8EDF2` | Contenu |
| Texte secondaire | `--text-muted` | `#93A1B0` | Légendes |
| **Primaire (signal froid)** | `--primary` | `#2FB6C9` | Actions, liens, marque |
| Primaire foncé | `--primary-700` | `#1B7F8E` | Hover/actif |
| **Accent (rehaussement)** | `--accent` | `#FF7A59` | Overlay lésion, CTA fort |
| Accent alt (perfusion haute) | `--accent-2` | `#F2C14E` | Pics cinétiques, highlights |
| Succès | `--success` | `#3FB98A` | États OK |
| Alerte | `--warning` | `#E4B34A` | Bandeau RUO |
| Danger | `--danger` | `#E5544B` | Erreurs |

**Colormap overlay lésion** (segmentation) : rampe froide→chaude `#1B7F8E → #2FB6C9 → #F2C14E → #FF7A59`, cohérente avec une lecture de perfusion. Opacité overlay recommandée : 45–60 %.

**Pendant clair** (rapports/PDF) : `--bg #F7F9FB`, `--surface #FFFFFF`, `--text #0B0F14`, `--border #DCE3EA`, primaire et accent inchangés.

### 3.3 Typographie (open source)
- **Titres / UI** : **Space Grotesk** (Google Fonts, OFL) — caractère technique, un peu instrument scientifique.
- **Corps / interface dense** : **Inter** (OFL) — lisibilité écran maximale.
- **Données / mono** (mesures, dimensions, volumes en ml) : **IBM Plex Mono** (OFL).

Échelle (rem, base 16 px) : `12 · 14 · 16 · 20 · 24 · 32 · 40`. Interlignage corps 1.5, titres 1.15. Graisses : 400 / 500 / 600 / 700.

### 3.4 Tokens CSS (à copier tel quel)

```css
:root {
  /* Couleurs — thème sombre (défaut app/viewer) */
  --bg: #0B0F14;
  --surface: #141A22;
  --surface-2: #1E2733;
  --border: #2A3644;
  --text: #E8EDF2;
  --text-muted: #93A1B0;

  --primary: #2FB6C9;
  --primary-700: #1B7F8E;
  --accent: #FF7A59;
  --accent-2: #F2C14E;

  --success: #3FB98A;
  --warning: #E4B34A;
  --danger: #E5544B;

  /* Overlay lésion (segmentation) */
  --overlay-alpha: 0.55;
  --overlay-stop-0: #1B7F8E;
  --overlay-stop-1: #2FB6C9;
  --overlay-stop-2: #F2C14E;
  --overlay-stop-3: #FF7A59;

  /* Typographie */
  --font-display: "Space Grotesk", system-ui, sans-serif;
  --font-body: "Inter", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, monospace;

  --fs-xs: 0.75rem; --fs-sm: 0.875rem; --fs-md: 1rem;
  --fs-lg: 1.25rem; --fs-xl: 1.5rem; --fs-2xl: 2rem; --fs-3xl: 2.5rem;
  --lh-body: 1.5; --lh-tight: 1.15;

  /* Espacement (échelle 4px) */
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px;
  --sp-5: 24px; --sp-6: 32px; --sp-8: 48px;

  /* Rayons & ombres */
  --radius-sm: 6px; --radius-md: 10px; --radius-lg: 16px;
  --shadow-1: 0 1px 2px rgba(0,0,0,.4);
  --shadow-2: 0 8px 24px rgba(0,0,0,.45);

  /* Focus accessible */
  --focus-ring: 0 0 0 2px var(--bg), 0 0 0 4px var(--primary);
}

:root[data-theme="light"] {
  --bg: #F7F9FB; --surface: #FFFFFF; --surface-2: #EEF2F6;
  --border: #DCE3EA; --text: #0B0F14; --text-muted: #566573;
}

body {
  background: var(--bg); color: var(--text);
  font-family: var(--font-body); font-size: var(--fs-md);
  line-height: var(--lh-body);
}
h1, h2, h3 { font-family: var(--font-display); line-height: var(--lh-tight); }
.metric, code, .dicom-value { font-family: var(--font-mono); }

.btn-primary {
  background: var(--primary); color: #04212A; border: none;
  padding: var(--sp-3) var(--sp-5); border-radius: var(--radius-md);
  font-weight: 600; cursor: pointer;
}
.btn-primary:hover { background: var(--primary-700); color: var(--text); }
:focus-visible { outline: none; box-shadow: var(--focus-ring); }

/* Bandeau conformité — présent sur chaque écran */
.ruo-banner {
  background: color-mix(in srgb, var(--warning) 15%, var(--surface));
  border: 1px solid var(--warning); color: var(--text);
  font-size: var(--fs-xs); padding: var(--sp-2) var(--sp-4);
  border-radius: var(--radius-sm); letter-spacing: .02em;
}
```

```html
<!-- À afficher en pied de chaque vue et en entête de chaque export -->
<div class="ruo-banner">
  Research Use Only — Not for diagnostic use. Aucune décision clinique ne doit
  reposer sur cet outil.
</div>
```

### 3.5 Règles d'usage (do / don't)
- **Do** : fonds sombres pour le viewer, accent chaud réservé au rehaussement/lésion et aux CTA, mono pour toute mesure chiffrée, contraste AA minimum (texte sur `--surface` ≥ 4.5:1).
- **Don't** : rose ruban, plus d'un accent chaud par écran, overlay opaque masquant l'anatomie, chiffres de performance présentés comme cliniques.
- **Logo** (piste) : glyphe abstrait = courbe de rehaussement (wash-in/wash-out) stylisée, monochrome `--primary`, jamais sur imagerie médicale réelle non anonymisée.

---

### Annexe — Stack open source de référence
`PyTorch` · `MONAI` / `nnU-Net` (modèle) · `pydicom` · `SimpleITK` / `dcm2niix` / `ANTs` (I/O & recalage) · `TorchIO` (augmentations) · `DICOM-Anonymizer` / `Microsoft Presidio` (dé-identification) · `MLflow` (suivi) · `ONNX Runtime` (inférence) · `Gradio` / `Streamlit` (démo) · `Docker` (repro). Le tout tient sur une machine GPU unique (≥16 Go VRAM).

**RGPD — check minimal MVP** : base légale + consentement documentés · dé-identification avant tout traitement · aucune donnée patient dans git · stockage chiffré local · registre de traitement tenu · DPA si les données proviennent d'un tiers (hôpital, dataset). Mention *Research Use Only* non optionnelle.

---

## Partie 4 — Prochaines étapes vers le MVP (état du 2026-07-21)

Backlog priorisé à partir de l'état réel du dépôt (voir synthèse en tête de document). Format identique
à la Partie 2 : chaque ligne a une priorité, une estimation et un critère de « fait ».

### P0 — bloquant avant tout pitch

| Tâche | Est. | Critère de « fait » |
|-------|:---:|------|
| ✅ Diagnostiquer la confiance à 1.0 | 0,5 j | **Cause identifiée** : `preprocess_dce_mri_with_boxes` utilise `crop=True` par défaut (comme le pipeline DBT), qui recadre chaque volume à la ROI de la lésion + marge de 16 voxels. Résultat : des volumes minuscules (ex. 45×72×70) où 29 % des coupes contiennent déjà la lésion — le modèle n'apprend qu'à localiser *dans un patch déjà centré sur la cible*, une tâche artificiellement facile. Confirmé sur les **9 patients du vrai split de test** (jamais vus à l'entraînement) : confiance = 1.0000 sur 9/9, pas un hasard sur 2 cas. C'est exactement la mise en garde déjà documentée dans le code DBT existant (`crop` docstring, `TransformData.py`) — elle s'applique aussi au MRI. **Fix retenu** : ré-entraîner avec `crop=False` (volumes pleine trame) en même temps que l'élargissement de l'échantillon (tâche suivante). |
| Élargir l'échantillon d'entraînement (64 → ~150-200 patients Duke-Breast-Cancer-MRI) et ré-entraîner | 1 j | Nouveau Dice/IoU test calculé sur un split plus large ; chiffre accompagné d'un intervalle approximatif (ex. bootstrap sur le split test), pas un point unique présenté comme définitif |
| Figer 3 cas de démo garantis à froid (sélectionnés, re-testés plusieurs fois, jamais d'échec) | 0,5 j | 3 `.npz` identifiés + scriptés (`scripts/demo_cases.py` ou équivalent), testés 3 fois de suite sans erreur, overlay visuellement convaincant sur chacun |
| Test réel dans un navigateur interactif (upload via le formulaire, pas seulement `curl`) | 0,5 j | Parcours complet (upload → overlay → détails) validé à la main ou via un outil d'automation fonctionnel, captures d'écran à l'appui ; aucun bug JS/CSS non détecté par les tests HTTP directs |

### P1 — renforce la crédibilité technique

| Tâche | Est. | Critère de « fait » |
|-------|:---:|------|
| Chronométrer l'inférence sur la machine GPU cible (item E2 encore non mesuré) | 0,25 j | Temps moyen par volume loggé et reporté dans la grille d'audit |
| Slider multi-coupes / vue MIP dans l'UI (aujourd'hui une seule coupe, la plus confiante, est affichée) | 1 j | Navigation entre coupes fonctionnelle dans le navigateur, sans lag perceptible |
| Écran « Comment ça marche » (pipeline en 4 étapes, 1 visuel) | 1 j | Écran intégré à l'app, compréhensible en 30 s |
| Rédiger le registre de traitement RGPD (aujourd'hui seulement mentionné, pas écrit) | 0,5 j | Document d'une page : base légale, nature des données, finalité, durée de conservation, mesures de sécurité |

### P2 — polish, pas indispensable pour un premier pitch

| Tâche | Est. | Critère de « fait » |
|-------|:---:|------|
| Trancher le nom de produit (Perfusio / Kinetix / Contra ou autre) + logo simple | 0,5 j | Nom choisi, logo abstrait intégré au header de l'app |
| Packaging Docker pour reproductibilité sur une autre machine | 1 j | `docker build && docker run` reproduit la démo sans configuration manuelle |
| N4 bias field / recalage inter-phase, si le Dice plafonne sur l'échantillon élargi | 1-2 j (conditionnel) | À déclencher seulement si le Dice à échantillon élargi (item P0 ci-dessus) déçoit ; sinon ne pas investir ce temps |

**Chemin critique vers un MVP pitchable** : diagnostic de la confiance → échantillon élargi + ré-entraînement →
3 cas de démo figés → test navigateur réel. Le reste (P1/P2) améliore la crédibilité et le confort mais
n'empêche pas un premier pitch fonctionnel.
