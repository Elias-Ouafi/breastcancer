# Démo

> **Research Use Only — Not for diagnostic use.** Outil de recherche, pas un
> dispositif médical. Aucune décision clinique ne doit en dépendre.

```bash
git clone https://github.com/Elias-Ouafi/breastcancer && cd breastcancer
pip install -r requirements.txt
python run_demo.py
```

Rien d'autre à télécharger : le modèle (`results_mri_p2_negfix/unet_best.pt`) et les
trois cas de démo sont versionnés dans le dépôt. Le serveur écoute uniquement sur
`127.0.0.1` — rien n'est exposé sur le réseau.

## Le déroulé

**La veille**, vérifier que la machine est prête sans occuper de port :

```bash
python run_demo.py --check
```

**Le jour J :**

1. `python run_demo.py`
2. Ouvrir <http://127.0.0.1:5000>
3. Cliquer sur **Cas 1** — pas de sélecteur de fichier à manipuler en plein pitch.
   (Le dépôt de fichier reste disponible pour un examen à vous.)
4. Dérouler le résultat, dans cet ordre :
   - le verdict et la confiance, avec le temps de calcul (~110 ms)
   - la coupe annotée, cadre orange sur la zone de réhaussement
   - **faire glisser le curseur** — la lésion apparaît, culmine, disparaît sur les
     25 coupes. C'est le moment qui convainc.
   - **Vue MIP** — la projection d'intensité maximale, comme lit un radiologue
   - déplier *Détail technique* si on vous pose des questions
5. Enchaîner sur **Comment ça marche** (lien en bas) si votre interlocuteur veut le
   pipeline.

Le cadre n'est tracé que sur la coupe réellement évaluée : les voisines sont montrées
telles quelles, le modèle ne les a pas analysées.

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

Trois phrases, avant qu'on ne vous les demande :

> « La coupe a été choisie à l'avance par un humain. Le modèle segmente très bien une
> lésion **quand on lui montre la bonne coupe** — 88 % de sensibilité. Il ne sait pas
> encore la trouver seul : on est passé de 0 % à 43 % avec un classifieur dédié,
> c'est en cours. »

> « Le Dice de 0,53 n'est pas comparable au 0,80 de la littérature : nos masques
> d'entraînement sont des boîtes englobantes, pas des contours d'expert. »

> « Rien n'est validé cliniquement. C'est de la recherche sur un échantillon de
> 186 patients. »

Le détail : les trois cas fonctionnent grâce à la clé `forced_slice` dans le `.npz`.
Sur un volume complet uploadé librement, la sélection automatique reste peu fiable.
C'est un problème de *sélection*, pas de *segmentation* — détaillé dans
[plan.md](plan.md) §4.2 et §4.3.

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
