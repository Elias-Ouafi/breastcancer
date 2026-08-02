# Démo — 3 commandes

> **Research Use Only — Not for diagnostic use.** Outil de recherche, pas un
> dispositif médical. Aucune décision clinique ne doit en dépendre.

```bash
git clone https://github.com/Elias-Ouafi/breastcancer && cd breastcancer
pip install -r requirements.txt
python run_demo.py
```

Puis ouvrir <http://127.0.0.1:5000> et cliquer sur **Cas 1**, **Cas 2** ou **Cas 3** —
pas de sélecteur de fichier à manipuler en plein pitch. (Le dépôt de fichier reste
disponible pour un examen à vous.)

Sur la page de résultat : le **curseur** fait défiler les 25 coupes autour de la zone
détectée, et **Vue MIP** affiche la projection d'intensité maximale. Le cadre n'est
tracé que sur la coupe réellement évaluée.

Rien d'autre à télécharger : le modèle (`results_mri_p2_negfix/unet_best.pt`) et les
trois cas de démo sont versionnés dans le dépôt. Le serveur écoute uniquement sur
`127.0.0.1` — rien n'est exposé sur le réseau.

`python run_demo.py --check` vérifie qu'une machine est prête sans occuper de port.

## Les trois cas

| Fichier | Patient | Coupe | IoU vérifié |
|---------|---------|:-----:|:-----------:|
| `demo_1_Breast_MRI_135.npz` | Breast_MRI_135 | 52 sur 176 | 0,830 |
| `demo_2_Breast_MRI_105.npz` | Breast_MRI_105 | 62 sur 156 | 0,738 |
| `demo_3_Breast_MRI_079.npz` | Breast_MRI_079 | 104 sur 154 | 0,728 |

Résultat attendu : verdict + confiance, coupe annotée avec le cadre sur la zone de
réhaussement, encart des limites connues, et le détail technique dépliable.

Chaque fichier contient un **pavé de 25 coupes** centré sur celle indiquée (~4,5 Mo
au lieu de 30 Mo pour le volume entier) : le modèle n'en évalue qu'une, les 24 autres
servent uniquement au curseur et au MIP. Régénération :
`python scripts/make_demo_cases.py` (nécessite les volumes complets, hors dépôt).

---

## À dire pendant la démo

Ces trois cas fonctionnent parce que **la coupe a été choisie à l'avance par un
humain**, pas par le modèle (clé `forced_slice` dans le `.npz`). Sur un volume
complet uploadé librement, la sélection automatique de coupe échoue encore
(vérifié 0/186 sur les patients de test) : le modèle segmente correctement une
lésion **quand on lui montre la bonne coupe**, il ne sait pas encore la trouver
seul. C'est un problème de *sélection*, pas de *segmentation* — détaillé dans
[plan.md](plan.md) §4.2.

Les chiffres à citer, mesurés sur les 28 patients de test (`imaging.evaluate`,
détail dans `results_mri_p2_negfix/eval_report.json`) :

| Mesure | Valeur | IC95 |
|--------|:------:|:----:|
| Dice (coupes avec lésion) | 0,53 | 0,47 – 0,59 |
| Sensibilité, lésion trouvée (IoU ≥ 0,1) | 88 % | 82 – 93 % |
| Sensibilité, centre visé juste | 81 % | 74 – 88 % |
| Faux positifs par examen | 222 | 205 – 237 |
| Coupes saines déclenchant une alarme | 99,97 % | 99,92 – 100 % |
| Temps de calcul par volume (RTX 5060) | 0,76 s | — |

Le Dice n'est pas comparable à la littérature (~0,80) : les masques d'entraînement
sont des **boîtes englobantes** TCIA, pas des contours experts, ce qui plafonne
mécaniquement le Dice atteignable. Les deux dernières lignes sont la formulation
chiffrée du problème de sélection de coupe — c'est la limite, autant la donner soi-même.

C'est écrit noir sur blanc dans l'app (encart « Limites connues » sur les deux
écrans) — autant l'assumer avant qu'on ne le demande.

## Si ça ne marche pas

| Symptôme | Cause probable | Correctif |
|----------|----------------|-----------|
| `run_demo.py` refuse de démarrer | Il dit lequel des trois prérequis manque | Suivre la ligne `->` qu'il affiche |
| Port déjà utilisé | Une instance tourne déjà | `python run_demo.py --port 5001` |
| Aucune coupe annotée affichée | Le fichier n'est pas un `.npz` prétraité | Utiliser un fichier de `demo_cases/` |
| Moteur affiché = `mock` | `run_demo.py` contourné | Relancer via `python run_demo.py` |

**Repli si le live échoue** : garder une capture d'écran d'un résultat réussi.
