# Point de fonctionnement — tête de décision au niveau examen

> **Research Use Only — Not for diagnostic use.**

Généré par `python -m imaging.oppoint` depuis `models/examclf/cv_predictions.csv`. Ne réentraîne rien : les scores sont ceux de la validation croisée d'`imaging.examclf`.

**Corpus** : 272 patients, 56 cancers, prévalence 20,6 %, 5 plis.

**ROC-AUC patient** : 0,457 [0,369 – 0,543]

## Au seuil visé (sensibilité 82,8 %)

Seuil pris **hors du pli noté** : chaque patient est jugé par un seuil calé sur les
quatre autres plis, jamais sur le sien.

| Mesure | Valeur | IC 95 % | Cible |
|---|---:|---|---:|
| Sensibilité | 78,6 % | 67,2 % – 88,9 % | 82,8 % |
| Spécificité | 20,4 % | 15,3 % – 25,9 % | 91,4 % |
| VPP | 20,4 % | 15,2 % – 25,7 % | — |
| NPV | 78,6 % | 67,3 % – 88,9 % | — |
| Prévalence du jeu | 20,6 % | — | — |

TP 44 · FP 172 · TN 44 · FN 12

## Ce que ces chiffres disent

La VPP est **au niveau de la prévalence** : savoir que le modèle a répondu « cancer » ne change pas la probabilité qu'il y en ait un.

À la sensibilité réellement atteinte (78,6 %), **le hasard donnerait 21,4 % de spécificité** — un classifieur aléatoire échange l'une contre l'autre exactement. Le modèle en donne 20,4 % : **en dessous**.

La cible de sensibilité n'est pas atteinte (78,6 % contre 82,8 %) : le seuil calé sur quatre plis ne transporte pas jusqu'au cinquième, ce qui est en soi une mesure — celle d'un score dont l'échelle ne veut rien dire d'un groupe de patients à l'autre (plan.md §4.9).

## Seuil naïf, pour comparaison

Calé sur les scores mêmes qu'il note ensuite — publié pour que l'écart soit lisible, pas pour être cité : sensibilité 83,9 %, spécificité 13,4 %, VPP 20,1 %, seuil 0,1428.

L'écart ne raconte pas l'histoire habituelle de l'optimisme, et il faut le dire : quand un modèle est au niveau du hasard, il n'y a rien à sur-estimer.
