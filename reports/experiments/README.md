# Métriques des runs archivés

Ce que ces fichiers sont : les courbes d'entraînement (`segmentation_metrics.csv` —
epoch, train_loss, val_dice, val_iou) et le score final sur le split test
(`segmentation_test_metrics.csv`) de runs qui ne servent plus, mais dont les chiffres
sont cités dans `plan.md`.

Les checkpoints correspondants (271 Mo) et les volumes prétraités qui les ont produits
(30 Go) ont été supprimés le 2026-08-10 : ils sont reconstructibles depuis
`data/raw_data/` via `python -m pipelines.dce_mri`. Ces CSV, eux, pèsent 28 Ko et sont
la seule trace de ce qui a été mesuré. Une affirmation datée sans le fichier qui la
soutient n'est plus une mesure, c'est un souvenir.

| Dossier | Ce qu'il documente |
|---|---|
| `results_ablation_pretrained_p1` | Encodeur ImageNet pré-entraîné — effondrement (`plan.md` §4.1) |
| `results_ablation_scratch_p1` | Le même run from-scratch, la référence à laquelle il est comparé |
| `results_mri_bankbench` | Banque de coupes memmap, gain de 7,1× sur le temps d'époque (§4.1) |
| `results_mri`, `results_mri_full`, `results_mri_p2` | Itérations DCE-MRI avant le run servant la démo |
| `results_full` | Run DBT sur échantillon élargi |

Le run **courant** n'est pas ici : il vit dans `models/dce_mri_p2_negfix/`, versionné,
avec son `eval_report.json` et son détail par patient.
