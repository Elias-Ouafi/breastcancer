# Documentation du projet

> Outil de recherche, pas un dispositif
> médical. Aucune décision clinique ne doit en dépendre.

## Sommaire
1. [Contexte et objectif](#contexte-et-objectif)
2. [Données](#données)
3. [Pipeline : fonctionnement et commandes](#pipeline--fonctionnement-et-commandes)
4. [Démo et application web](#démo-et-application-web)
5. [Partie 4 — Journal des mesures](#partie-4--journal-des-mesures)
6. [Prochaines pistes](#prochaines-pistes)
7. [État d'avancement et feuille de route](#état-davancement-et-feuille-de-route)
8. [Écarts doc ↔ code](#écarts-doc--code)
9. [Partie 3 — Charte graphique](#partie-3--charte-graphique)
10. [Développement](#développement)
11. [Licence et données](#licence-et-données)

---

## Contexte et objectif

### La question

Localiser les lésions cancéreuses sur une IRM mammaire dynamique (DCE-MRI) : où elles sont,
avec quelle sensibilité, et au prix de combien de fausses détections. Le projet utilise
uniquement des données publiques de [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) (TCIA).

### Deux objectifs

1. **Contribuer, à mon niveau, au secteur médical** en construisant l'outil de bout en
   bout, honnêtement mesuré.
2. **Un projet de portfolio de data engineering** : ingestion de DICOM, stockage en médaillon
   (bronze → silver → gold), validation, lignage, orchestration, CI, Docker. Les livrables de
   data engineering passent avant les nouvelles expériences de modélisation.

### Une modalité, deux corpus

Une seule collection est utilisée, **Duke-Breast-Cancer-MRI** : des IRM dynamiques de patientes
avec cancer, et une boîte de lésion par patient. Elle sert deux corpus, séparés :

- `dce_mri_p2` : le corpus du modèle de la démo (soustraction post − pré, un canal). Il est figé :
  le checkpoint servi a été entraîné dessus et un test le compare bit à bit aux DICOM.
- `dce_mri_nnunet` : le corpus préparé pour nnU-Net v2 (canaux pré et post-contraste,
  pseudo-masques issus des boîtes), avec son propre module de prétraitement (`mri_nnunet/`).

### Ce que le projet ne vise plus

Il a d'abord visé un point de fonctionnement de dépistage (sensibilité 82,8 %, spécificité 91,4 %,
au niveau patient) sur la tomosynthèse BCS-DBT. **Cette piste est abandonnée** : les mesures
sont négatives (ROC-AUC patient 0,457 ; l'archive du §4.4 à §4.15 les résume) et son code et ses
données ont été retirés le 2026-09-20. Le dernier état qui les contient est le commit `1a364d4`.

Duke est une cohorte de cancers (prévalence 100 %) : une **spécificité n'y est pas mesurable**, et
aucun résultat ne doit être présenté comme une comparaison au programme national de dépistage. La
mesure adaptée est la sensibilité lésionnelle en fonction du nombre de faux positifs par examen
(FROC), avec intervalle de confiance : elle n'est **pas encore mise en place** (`imaging/evaluate.py`
donne la sensibilité et les faux positifs à un seul seuil, par coupe).

---

## Données

### Collection

| Collection | Contenu | Rôle | Licence |
|---|---|---|---|
| **Duke-Breast-Cancer-MRI** | IRM dynamiques DICOM de patientes avec cancer, une boîte de lésion par patient | localisation ; corpus de la démo et corpus nnU-Net | CC BY-NC 4.0 |

Sur cette machine : 189 patients téléchargés, **834 dossiers** de séries (829 séries dynamiques et
5 autres ; 12 restent en bronze après la purge). Le corpus de la démo compte **186 volumes**, un par patient : la soustraction de deux
séries, pas un volume par série. Les trois patients restants (`Breast_MRI_106`, `_120`, `_203`) n'ont
pas à la fois la phase pré-contraste et la 2ᵉ post-contraste. La collection ne publie ni site
(`InstitutionName` est vide) ni date exploitable (anonymisée au 1990-01-01) : le fabricant, le modèle
et le champ magnétique du scanner sont la seule variable d'acquisition disponible.

### Stockage en médaillon : bronze → silver → gold

```
data/                                    41,7 Gio  ignoré par git, sauf les trois cas de démo
├── bronze/tcia/                          0,6 Gio  zone de transit : la source telle que publiée, jamais
│   │                                              réécrite, supprimée une fois la série en silver
│   ├── duke_mri/                                  12 dossiers restants : 7 séries de 3 patients sans
│   │                                              copie en silver et 5 séries non dynamiques
│   └── Annotation_Boxes.xlsx                      les boîtes de lésion (jamais purgées : minuscules)
├── silver/
│   ├── dce_mri_p2/                       5,0 Gio  corpus de la démo : un .npz par patient, un canal
│   └── dce_mri_nnunet/                  30,4 Gio  corpus nnU-Net (mri_nnunet/)
│       ├── native/                      26,1 Gio  NIfTI sans perte, RAS, toutes les phases : 186
│       │                                          patients, ~144 Mio chacun
│       ├── cases/                        4,3 Gio  185 cas traités : canaux normalisés, pseudo-masque,
│       │                                          masque de l'organe, case.json
│       ├── qc/                                    rapport de QC : 10 cas tirés (graine 42) + agrégat
│       ├── registry.csv · exclusions.csv          ce qu'il contient, ce qu'il a écarté et pourquoi
│       └── log/cases.jsonl                        étapes, paramètres, durées et anomalies de chaque cas
└── gold/                                 5,8 Gio  dérivé de silver et reconstructible
    ├── slice_bank_p2/                    5,7 Gio  banque de coupes (memmap)
    ├── nnunet_raw/                      (4,3 Gio) Dataset501_DukeDCEBreast : 185 cas, en liens physiques
    │                                              vers cases/, donc sans espace disque en plus
    └── demo_cases/                      13,7 Mio  les trois cas de démo, seuls fichiers versionnés
models/                                  70,9 Mio  checkpoints + les métriques qui les justifient
reports/                                  6,2 Kio  rapports JSON
docs/img/                                 0,6 Mio  images de la documentation
```

Tailles mesurées le 2026-09-26 avec `du -sb`, en unités binaires (1 Gio = 2³⁰ octets, comme `du -h`),
**après les purges et la construction du corpus nnU-Net**. `du` compte une seule fois un fichier en
liens physiques : le total ne compte pas l'export `nnunet_raw` en double.

**Le bronze est une zone de transit, pas une archive.** Dès qu'une série est en silver
**dans tous les corpus qui la lisent**, son dossier DICOM est supprimé : la donnée n'est plus
conservée qu'une fois. C'est l'étape `purge` du flow (`download → preprocess → purge`) ;
`--keep-bronze` la désactive. Une seule fonction supprime du brut,
`TransformData.purge_bronze_series`, et elle **refuse plutôt que de parier** : le dossier doit
être un enfant direct du bronze, chaque fichier silver doit s'ouvrir comme un volume (un fichier non
vide mais tronqué ne suffit pas), et il faut au moins un fichier silver.

Le corpus nnU-Net est un second lecteur des mêmes DICOM, avec une exigence plus stricte : une série
n'est supprimée que si sa copie native a été **écrite puis relue identique** au DICOM (même type de
pixel, mêmes valeurs, même géométrie physique). Cette copie est sans perte : l'espacement, la
normalisation et les canaux restent réglables après la purge, sans télécharger à nouveau. Une phase
non ingérée (patient sans boîte, série aux coupes manquantes) reste en bronze.

**Ce qui a été fait le 2026-09-20.** La purge a été exécutée sur le DBT, puis sur l'IRM (822 séries,
63,8 Go) une fois la copie native de chaque phase relue identique, et les données DBT ont été
retirées avec leur code (§4.18). Les deux corpus IRM ont été reconstruits depuis le bronze
et comparés à l'ancien silver : les 186 volumes du corpus de la démo sont **identiques valeur par
valeur**, et son manifeste (absent jusque-là, le corpus étant antérieur à `lineage.py`) est écrit.

**Le coût est assumé** : silver n'est plus « reconstructible » à volonté, c'est la source. C'est
pourquoi la copie native du corpus nnU-Net garde toutes les phases.

Chaque chemin est défini **une seule fois** dans `config.py`. `data/` est ignoré par git, sauf les
trois cas de démo ; `models/` l'est aussi, sauf les checkpoints de la démo et les artefacts de
mesure (§4.4 à §4.15).

---

## Pipeline : fonctionnement et commandes

### Vue d'ensemble

| Étape | Où | Ce qu'elle garantit |
|---|---|---|
| **Extraction** | `ExtractData.py`, `http_timeouts.py` | Téléchargements qui reprennent là où ils s'arrêtent ; plafond exprimé en volume ajouté par l'appel ; délai maximal sur chaque requête ; une série n'est comptée que si son dossier existe. |
| **Orchestration** | `pipelines/dce_mri.py` | Un flow Prefect : `download → preprocess → purge → train → evaluate`, chaque étape saute ce qui est fait. |
| **Stockage** | `config.py` | Médaillon `bronze` → `silver` → `gold`, chemins définis une fois ; le bronze est purgé une fois la série en silver. |
| **Transformation** | `TransformData.py`, `mri_nnunet/` | Deux chemins qui ne se mélangent pas : la soustraction de la démo, et le corpus nnU-Net (N4, masque, recalage, rééchantillonnage, normalisation dans le masque, pseudo-masques). |
| **Validation** | `validation.py` | Dimensions, type, valeurs finies et masque binaire vérifiés au point unique d'écriture. |
| **Lignage** | `lineage.py` | Un `manifest.json` par dossier : révision git (suffixe `-dirty`), source, paramètres, statistiques par cas. Écrit en dernier : son absence signale une passe interrompue. |
| **Curation** | `imaging/slicebank.py` | Banque memmap qui paie la décompression une seule fois (×7,1 sur le temps d'époque). |
| **Entraînement / évaluation** | `imaging/` | Découpage par patient, IC bootstrap par patient. |
| **Service** | `app/`, `Dockerfile`, `run_demo.py` | Flask HTML + API JSON ; image sans JVM ni ITK, lecture seule, non-root, boucle locale uniquement ; préflight qui analyse un cas réel et vérifie le port avant de servir. |

### Prérequis et installation

- Python ≥ 3.12 ; `torch` adapté à la machine (CPU : `--index-url https://download.pytorch.org/whl/cpu`).
- Accès TCIA via `tcia_utils` / `nbia` : les collections publiques ne demandent pas de clé.

```bash
pip install -e .                  # la démo seule : torch, Flask, numpy, Pillow
pip install -e ".[all]"           # tous les cas ci-dessous à la fois
pip install -e ".[data]"          # + pandas, pydicom, openpyxl : tables et DICOM
pip install -e ".[collect]"       # + client TCIA, python-gdcm : télécharger et décoder du vrai DICOM
pip install -e ".[nnunet]"        # + SimpleITK, PyYAML, nibabel, nnunetv2 : le corpus nnU-Net
pip install -e ".[orchestration]" # + Prefect, pour pipelines/
pip install -e ".[dev]"           # + pytest, ruff
```

**Un seul fichier de dépendances : `pyproject.toml`.** Il n'y a plus de `requirements*.txt`, donc
plus de seconde liste qui dérive. Le socle (`dependencies`) est exactement la démo : elle n'importe
que quatre paquets, et l'écart avec le reste est mesuré au §4.17 (68 paquets / 355 Mo contre 17 /
153 Mo sur Windows). Chaque autre usage est un extra, et `all` les réunit : un test échoue si un extra
n'y figure pas, si le socle grossit, ou si le code importe un paquet qu'aucun extra ne déclare. Le
`Dockerfile` lit le socle avec `tomllib` plutôt que de le recopier. La CI installe `.[dev,data]`.

### Pipeline DCE-MRI orchestré (Prefect)

Les cinq étapes — téléchargement, prétraitement, purge du bronze, entraînement,
évaluation — forment un flow Prefect dans `pipelines/dce_mri.py` :

```bash
python -m pipelines.dce_mri --dry-run          # affiche le plan, n'exécute rien
python -m pipelines.dce_mri                    # exécute
python -m pipelines.dce_mri --from preprocess  # reprend en sautant le téléchargement de 60 Go
python -m pipelines.dce_mri --keep-bronze      # garde les DICOM après le prétraitement
```

La purge supprime **toutes les phases** DCE d'un patient dont le volume est en silver (pas
seulement les deux que la soustraction lit : 457 des 829 séries DCE ne sont lues par aucune
soustraction, et resteraient) et laisse les patients sans annotation. Le téléchargement se saute
aussi quand le bronze est vide et le silver plein : c'est l'état final normal, pas « rien de
téléchargé ».

Chaque étape vérifie sa propre sortie et saute ce qui est déjà fait (`--force` pour
refaire). **Seul le téléchargement réessaie** : un échec TCIA est transitoire, un
entraînement qui plante replanterait au même endroit une heure plus tard. Les tâches
appellent les mêmes fonctions que les commandes manuelles, qui ne peuvent donc pas
diverger.

**La purge attend le corpus nnU-Net** quand il existe : une série n'est supprimée que si sa copie
native a été vérifiée (voir « Stockage en médaillon »).

### Préparer une IRM jamais annotée

Le flow ci-dessus prépare le corpus **annoté** : `preprocess_dce_mri_with_boxes`
ignore tout patient absent de la table d'annotations. Un examen nouveau est exactement
ce patient-là — il n'y avait donc aucun chemin du DICOM jusqu'à un volume que l'app
accepte. `preprocess_dce_mri_exams` le fournit :

```python
from TransformData import preprocess_dce_mri_exams

# root_dir contient les dossiers de séries du patient (un par SeriesInstanceUID).
preprocess_dce_mri_exams("data/bronze/nouveaux_examens",
                         output_dir="data/silver/nouveaux_examens")
```

Il ne demande que ce qu'exige une soustraction : la série pré-contraste et la passe
post-contraste choisie (la 2ᵉ par défaut, §4.1). Le masque est vide et **la trame
n'est jamais recadrée** — les deux découlent de l'absence d'annotation : il n'y a
aucune lésion sur laquelle recadrer, et la pleine trame est la géométrie sur laquelle
le checkpoint servi a été entraîné.

**Ce qui garantit que le volume est le bon.** Les deux chemins appellent la même
fonction `dce_subtraction` — une seule définition de « post − pré, seuillé à 0, puis
z-normalisé ». L'égalité est donc structurelle, pas seulement testée ; un test la
vérifie quand même, et une mutation prouve qu'il sait échouer (§4.16).

Un examen incomplet (pré ou post manquante, passe demandée absente, formes
discordantes entre phases) est compté et sauté, sans interrompre les autres.

### Le corpus nnU-Net (`mri_nnunet/`)

Un module et un corpus à part, pour ne rien changer à ce qui sert la démo. Il part du DICOM brut et
produit ce que nnU-Net v2 attend : plusieurs canaux en géométrie native, des pseudo-masques, un
export `nnUNet_raw`. Tous les paramètres sont dans `mri_nnunet/config.yaml`, aucun en dur.

```bash
pip install -e ".[data,nnunet]"
python -m mri_nnunet build --ingest-only     # DICOM -> NIfTI natif sans perte (médiane 15 s par patient)
python -m mri_nnunet build                   # + traitement complet (médiane 55 s par patient, §4.19)
python -m mri_nnunet spacing                 # distribution des boîtes et espacement choisi
python -m mri_nnunet export                  # data/gold/nnunet_raw/Dataset501_DukeDCEBreast
python -m mri_nnunet qc --n 10               # histogrammes avant/après, coupes avec overlay, statistiques
```

**Deux étapes, qui changent à des vitesses différentes.** L'*ingestion* lit le bronze : DICOM →
NIfTI en géométrie native, réorienté RAS (vérifié avec `nibabel`), **toutes les phases**, relu et
comparé pixel à pixel. Le *traitement* n'a plus besoin du bronze et s'exécute dans l'ordre imposé :

1. **N4** (correction de biais), avant que quoi que ce soit ne regarde les intensités ;
2. **masque de l'organe** : Otsu, morphologie, plus grande composante, trous comblés ;
3. **recalage rigide** de chaque canal sur la référence (voir ci-dessous) ;
4. **rééchantillonnage** : B-spline d'ordre 3 pour les images, le recalage repris dans la même
   interpolation ; un axe plus de deux fois plus épais que le plan garde son espacement natif ;
5. **clip aux percentiles [0,5 ; 99,5]**, calculés par cas et par canal sur les voxels du masque ;
6. **z-score** par cas et par canal avec la moyenne et l'écart-type du masque, **0 hors masque** ;
7. **crop** sur la boîte englobante du masque de l'organe.

**Les pseudo-masques** sont des ellipsoïdes inscrits dans les boîtes, calculés dans l'espace
physique (ils suivent l'anatomie à travers la réorientation, le recalage et le rééchantillonnage) et
rastérisés sur la grille finale, jamais interpolés. Le rapport volume de la boîte / volume du
pseudo-masque est rapporté par cas : 1,66 à 2,19 sur les 185 cas (médiane 1,91), pour 6/π = 1,91 attendu. Un
**contraste de rehaussement** (le pseudo-masque contre le reste de l'organe, en écarts-types)
signale une boîte lâche ou mal placée ; son seuil est une hypothèse, jamais utilisée pour écarter un cas.
À l'échelle, cet indice **ne discrimine pas** : il alerte sur 118 cas sur 185 (§4.19).

**L'espacement suit les petites lésions**, pas la médiane du jeu de données : il est choisi pour que
le plus petit axe de 90 % des lésions garde au moins 8 voxels (`spacing.py`), borné entre 0,5 et 1,5 mm.

**Deux mesures ont changé la conception** (détail au §4.18) : les numéros de coupe des boîtes suivent
`InstanceNumber`, qui décroît le long de z ; et le recalage par information mutuelle dégradait
l'alignement pré/post. Il est remplacé par la corrélation, avec une garde : une transformation n'est
acceptée que si elle **améliore** la corrélation dans l'organe, sinon l'identité est conservée et
l'écart est journalisé.

**Traçabilité.** Chaque cas laisse une ligne dans `log/cases.jsonl` (étapes, paramètres, durées,
anomalies) ; un cas écarté laisse une ligne dans `exclusions.csv` avec sa raison (phase manquante,
série incomplète ou illisible, boîte incohérente, masque vide, lésion hors de l'organe…), jamais un
échec silencieux. Chaque cas est **idempotent** : le hash des paramètres dont il dépend est stocké à
côté, et un cas inchangé est sauté.

**Ce qui n'est pas fait** : raffinement MedSAM des pseudo-masques, augmentations TorchIO, contrôle de
biais scanner/site, post-traitement en composantes connexes et FROC, 5 plis et 3 graines. Ils relèvent
de l'entraînement, pas du prétraitement (voir « Prochaines pistes »).

### Entraînement et évaluation

| Commande | Rôle |
|---|---|
| `python -m imaging.train --data-dir data/silver/dce_mri_p2 --epochs 25` | U-Net 2D de localisation (BCE + soft-Dice), découpage par patient, meilleur checkpoint + `segmentation_metrics.csv` dans `models/dce_mri/`. `--smoke-test` valide la boucle sans données. |
| `python -m imaging.evaluate --data-dir data/silver/dce_mri_p2 --checkpoint models/dce_mri_p2_negfix/unet_best.pt` | IC bootstrap par patient, sensibilité lésion, faux positifs par volume, temps d'inférence → `eval_report.json` + `eval_per_patient.csv`. |
| `python -m imaging.sliceclf --slice-bank data/gold/slice_bank_p2 --epochs 25` | Classifieur de coupe, sélectionné sur le top-1 (§4.3). |

---

## Démo et application web

### Lancer la démo

```bash
git clone https://github.com/Elias-Ouafi/breastcancer && cd breastcancer
pip install -e .
python run_demo.py --open       # préflight, puis le navigateur sur http://127.0.0.1:5000
docker compose up --build       # alternative avec Docker seul (image jamais construite ici)
```

Rien d'autre à télécharger : le modèle (`models/dce_mri_p2_negfix/unet_best.pt`) et les
trois cas de démo sont versionnés. Le serveur écoute uniquement sur `127.0.0.1`.

**Le préflight analyse un vrai cas** (~0,5 s ici, GPU chaud) avant d'ouvrir le port. Il
ne se contente pas de vérifier que les fichiers existent : un checkpoint tronqué par un
clone partiel, un PyTorch qui ne démarre pas, un `.npz` dont les clés ont bougé passent
tous un contrôle d'existence et meurent au premier clic — le seul moment où ils ne
doivent pas.

| Commande | Ce qu'elle fait |
|---|---|
| `python run_demo.py` | Préflight complet (fichiers + une analyse réelle + port libre), puis sert |
| `python run_demo.py --open` | Idem, et ouvre le navigateur une seconde après |
| `python run_demo.py --check` | Même contrôle, sans occuper de port — **à faire la veille** |
| `python run_demo.py --fast-check` | Inventaire des fichiers seulement, sans charger le modèle |
| `python run_demo.py --port 5001` | Autre port |

**Le port est vérifié avant de démarrer.** Sous Windows, une deuxième instance se liait
sans erreur à un port que werkzeug tenait déjà (SO_REUSEADDR) : elle affichait
« Running on http://127.0.0.1:5000 » pendant que le système continuait de router vers le
premier processus. Le lanceur interroge maintenant le port en s'y **connectant**, et
refuse de démarrer en nommant le port et l'alternative (§4.17).

### Déroulé conseillé

1. Cliquer sur **Cas 1** (pas de sélecteur de fichier à manipuler en direct).
2. Dérouler le résultat : verdict et temps de calcul (~70 ms à chaud, ~0,6 s au premier
   appel), coupe annotée avec le cadre sur la zone de rehaussement, **curseur** entre les
   coupes (la lésion apparaît, culmine, disparaît), **Vue MIP** (projection d'intensité
   maximale), *Détail technique* dépliable.
3. Enchaîner sur **Comment ça marche** si l'interlocuteur veut le pipeline.

Le cadre n'est tracé que sur la coupe réellement évaluée ; les voisines sont affichées
telles quelles.

### Les trois cas

| Fichier | Patient | Coupe | IoU vérifié | Cadre |
|---|---|:---:|:---:|---|
| `demo_1_Breast_MRI_135.npz` | Breast_MRI_135 | 52 sur 176 | 0,830 | 39×42 px |
| `demo_2_Breast_MRI_105.npz` | Breast_MRI_105 | 62 sur 156 | 0,738 | 42×44 px |
| `demo_3_Breast_MRI_079.npz` | Breast_MRI_079 | 104 sur 154 | 0,728 | 66×92 px |

Chaque fichier contient un pavé de **25 coupes** centré sur la coupe indiquée (~4,5 Mo au
lieu de 30 Mo) : le modèle n'en évalue qu'une, les autres servent au curseur et au MIP.
Régénération : `python scripts/make_demo_cases.py` (volumes complets requis). Le GIF du
README : `python scripts/make_demo_gif.py` (Playwright requis, hors dépendances du
projet).

### L'application web (`app/`)

Une petite app Flask : on envoie une IRM DCE prétraitée, elle renvoie le verdict, la
coupe et le cadre. Le `.npz` envoyé vient de `preprocess_dce_mri_exams` pour un
examen neuf, ou de `preprocess_dce_mri_with_boxes` pour un examen de la collection
annotée — les deux écrivent le même volume, et un test l'épingle (§4.16). Elle parle uniquement à un `Predictor` (`app/predictor.py`) :

| Backend | Sélection | Rôle |
|---|---|---|
| `mock` (défaut) | — | Fabrique un résultat plausible, ignore les pixels |
| `dce_mri` | `MRI_APP_BACKEND=dce_mri` (posé par `run_demo.py`) | Localisation DCE-MRI via `inference.predict_dce_mri`, sur le volume de soustraction post − pré |

Un troisième backend `unet` (DBT) a été **supprimé le 2026-09-15** : son checkpoint avait
été écrasé par un smoke test (§4.1) et jamais reconstruit.

**Formats acceptés, par backend.** Le formulaire n'offre que ce que le moteur servi
sait lire : `.npz` sous `dce_mri`, la liste large (`.dcm`, `.nii`, images) sous `mock`,
qui ne regarde aucun pixel. Le texte sous la zone de dépôt, l'attribut `accept` et le
contrôle serveur sont **rendus depuis le même ensemble**, et un test l'épingle. Avant
le 2026-09-20 la zone annonçait « .npz, DICOM, NIfTI ou image » sous `dce_mri` : un
DICOM déposé renvoyait un 500 avec un message en anglais **et le chemin temporaire du
serveur affiché sur la page** (§4.17).

**Points d'entrée** : `GET /` (formulaire + boutons des cas de démo),
`GET /comment-ca-marche`, `POST /predict` (page HTML, champ `mri`), `POST /demo/<n>`,
`POST /api/predict` (JSON). Contrat de résultat : `lesion_detected`,
`slice_preselected`, `confidence`, `best_slice`, `box_xywh`, `n_slices`.

**Local par conception** : liaison sur `127.0.0.1` uniquement (seul le port est
configurable, `MRI_APP_PORT`), fichiers envoyés supprimés juste après le calcul, toutes
les coupes embarquées en data URI (curseur et MIP sans aller-retour serveur).

**Limite connue** : la sélection automatique de coupe sur un volume complet ne trouve pas
la lésion de façon fiable (0/186 par la confiance de segmentation, 43 % avec le
classifieur de coupe). Les cas de démo imposent une coupe vérifiée via la clé
`forced_slice`, et le résultat le dit (`slice_preselected: true`). `confidence` est un
maximum de probabilité **par pixel**, saturé à 1,0 avec ce checkpoint : l'app l'affiche
sous ce nom, pas comme une confiance d'examen. Le bandeau *Research Use Only* n'est
jamais repliable.

---

## Partie 4 — Journal des mesures

Journal daté, réduit à ce qu'il faut retenir : la **conclusion** de chaque mesure, échecs
compris, et le **prochain test** qu'elle appelle quand il y en a un. Les tableaux détaillés,
les incidents et les versions successives de la prose sont dans l'historique git. Les pistes
sont dans « Prochaines pistes ».

### 4.1 Optimisation de l'entraînement et du prétraitement (2026-07-26)

**Conclusion.** Le seul gain franc est d'ingénierie : la banque de coupes memmap ramène
l'époque de 1 020 s à 143 s (**×7,1**, 30 époques : 8,5 h → 1,2 h). Côté précision, le Dice
test reste vers **0,55** dans toutes les configurations viables (retenue : phase 2 +
GroupNorm, 0,552) ; l'encodeur ImageNet s'effondre (0,414) et la 2ᵉ passe post-contraste
n'est pas répliquée (+0,001, du bruit). Le plafond vient de la cible : les masques sont des
boîtes englobantes. Un smoke test avait écrasé le checkpoint DBT servi ; il écrit désormais
dans `smoke_test/`.

**Prochain test.** Cesser d'optimiser le Dice et mesurer la détection (centre dans la boîte,
sensibilité par faux positif). N4 et recadrage sur la région mammaire, non appliqués, ne se
testent que si la détection plafonne à son tour.

### 4.2 Échec de la localisation automatique sur volume complet, et contournement (2026-07-26)

**Conclusion.** La coupe de plus haute confiance contient la lésion dans **0/186** cas : la
confiance est saturée à ~1,0 sur toutes les coupes (aire prédite 1 228,6 px avec lésion,
1 228,3 sans). Plus de négatifs (ratio 2 → 8) améliore le Dice (0,552 → 0,580) sans rien
changer à la localisation. La perte de Tversky focale divergeait en fp16 (NaN) ; calculée en
fp32, la divergence passe de l'époque 11 à 15 sans disparaître. **C'est un problème de
sélection, pas de segmentation**, contourné par une coupe figée (trois cas de démo).

**Prochain test.** Entraîner en fp32 de bout en bout pour savoir si la divergence résiduelle
vient du forward en précision mixte.

### 4.3 Évaluation chiffrée et intervalles de confiance (2026-08-02)

**Conclusion.** Sur 28 patients de test (bootstrap sur les patients) : Dice **0,533**
[0,473 – 0,593], sensibilité (IoU ≥ 0,1) 88,0 %, 222 faux positifs par volume, **99,97 %** des
coupes saines en alarme — la confiance ne discrimine pas. Un classifieur de coupe dédié
choisit la bonne coupe dans **42,9 %** des cas (12/28, contre 0 %), 50,0 % en top-3, 16/28
patients en top-5, AUC intra-volume 0,803. Tractable, pas livrable : un envoi libre se
tromperait plus d'une fois sur deux, et l'IC est large. Le rang est bimodal (médian 3,5,
moyen 19,7).

**Prochain test.** Mesurer une sortie « 5 coupes candidates à revoir » (57 % en top-5) et
resserrer l'IC en évaluant sur davantage de patients de test.

### 4.4 à 4.15 Archive : la piste DBT (retirée le 2026-09-20)

Le projet a d'abord cherché à répondre à « y a-t-il un cancer ? » sur la tomosynthèse (collection
BCS-DBT, 5 060 patients, 89 cancers). Les mesures ci-dessous sont **négatives** et la piste est
abandonnée ; le code et les données ont été retirés (§4.18). Le code complet est dans le commit
`1a364d4` ; les artefacts de mesure (`models/examclf/`, `reports/examclf_operating_point.json`)
restent versionnés comme preuves.

| § | Ce qui a été mesuré | Conclusion |
|---|---|---|
| 4.4 | Latéralité des séries | Le tag DICOM lit `L` sur 262 séries sur 262 (pixels : R 134 / L 128) ; avant correction 23 masques sur 147 tombaient sur du fond. Latéralité par les pixels, 14 séries en miroir retournées |
| 4.5 | Bénin ou malin sur recadrage (130 patients) | ROC-AUC 0,591 [0,491 – 0,693] puis 0,513 [0,411 – 0,615] : les deux IC contiennent 0,5 ; mémorisation (perte 0,83 → 0,27, AUC au hasard) |
| 4.6 | Appariement boîte ↔ série | La collection publie l'inventaire : jointure sur (patient, étude, vue) au lieu d'une inférence ; 260 séries annotées au lieu de 253, 4 masques déplacés |
| 4.7 | Tête de décision au niveau examen (multi-instance, 56 cancers) | ROC-AUC patient **0,457** [0,369 – 0,544], sensibilité 0 % à 0,5, exactitude 79,4 % (= « toujours négatif ») ; sac de 32 : 0,414 |
| 4.8 | Warm start | AUC patient 0,426 ; AUC par coupe hors pli 0,502 : l'encodeur n'apprend rien même avec une étiquette 10 fois plus dense, le goulot n'est pas l'agrégation |
| 4.9 | Signal des pixels | Le pouvoir discriminant est **relatif à l'examen** : 99ᵉ percentile AUC 0,532 mis en commun, 0,736 normalisé par examen |
| 4.10 | Score relatif, mesures sans modèle, résolution native | 0,465 ; une ligne de numpy bat le CNN en intra-examen (0,732 contre 0,592) ; rien ne bat le hasard (0,366 à 0,442) ; résolution native 0,451 : trouver la lésion annotée n'est pas détecter un cancer |
| 4.11 | Point de fonctionnement (seuil hors pli) | Sensibilité 78,6 %, spécificité 20,4 % (cible 91,4 %), VPP 20,4 % = prévalence 20,6 % : le modèle est sous le hasard |
| 4.12 – 4.15 | Catalogue DuckDB, tests dbt, chaîne DBT orchestrée, stockage S3 | Fonctionnels (36 tests dbt, parité vérifiée ligne à ligne, synchronisation idempotente) et retirés avec les données. Trois leçons reprises ailleurs : un `socket.setdefaulttimeout` ne suffit pas, il faut un `timeout=` explicite ; un téléchargement raté ne doit pas être compté comme réussi ; un manifeste décrit le corpus, pas la dernière passe |

**Ce qui reste utile de cette piste** : la démonstration qu'un score absolu ne se compare pas d'un
examen à l'autre (§4.9), qui vaut aussi pour l'IRM (le z-score dans le masque est fait par cas), et
que trouver la lésion annotée n'est pas détecter un cancer.

### 4.16 Le chemin « nouvelle IRM » : trois défauts, et ce que la mesure a corrigé (2026-09-19)

**Conclusion.** Aucune fonction ne pouvait préparer une IRM jamais annotée ;
`preprocess_dce_mri_exams` le fait, et `dce_subtraction` est la définition unique de la
soustraction. Sur DICOM brut réel (`Breast_MRI_037`, 144 coupes), le volume est **identique
bit à bit** à celui du corpus, un test le rejoue ; `POST /api/predict` répond en 3,86 s. La
coupe choisie tombe dans la lésion : une cohérence, pas une mesure, le chiffre citable reste
43 % de top-1 (§4.3). Autres corrections : défaut `crop=True` qui ne reproduisait pas le
corpus servi, code hérité supprimé (−105 lignes, 3 dépendances en moins). La normalisation
écrête au 99ᵉ percentile : c'est une raison de plus pour que les statistiques du §4.9 ne
valent que par examen. Leçon : **60 Go et 840 séries IRM étaient présents** ; le balayage qui
les disait absents ne regardait pas un niveau plus bas. **288 tests**.

**Prochain test.** Mesurer le top-1 du choix de coupe sur des IRM neuves non annotées : le
seul chiffre disponible est 43 % sur 28 patients.

### 4.17 La démo mesurée comme un parcours, pas comme un fichier (2026-09-20)

**Conclusion.** Le parcours entier, chronométré, a révélé quatre défauts : (1) l'installation
pèse **68 paquets / 355 Mo** (Windows) alors que la démo n'en importe que **17 / 153 Mo** →
socle de `pyproject.toml` réduit à ces 4 paquets ; (2) le préflight ne prouvait rien du modèle → `--check` le charge et
analyse un cas (2,6 s) ; (3) un port occupé ne produisait aucune erreur sous Windows → le
lanceur s'y connecte avant de démarrer ; (4) l'interface invitait des formats que le modèle ne
lit pas (HTTP 500 et chemin serveur affiché) → un seul ensemble d'extensions partout. La CI
n'installait pas Pillow : la démo n'y exerçait que son repli texte, jamais la coupe annotée.
**300 tests**.

**Prochain test.** Construire l'image (`docker compose up --build`) : Docker n'est pas installé
ici, l'image n'a été construite ni ici ni en CI.

### 4.18 Médaillon, purge, retrait du DBT et corpus nnU-Net (2026-09-20)

**Conclusion.**
- **Reproductibilité prouvée avant de supprimer.** Le silver reconstruit depuis le bronze est
  identique valeur par valeur à l'ancien : 271 fichiers DBT sur 271, 186 volumes IRM sur 186. C'est
  ce qui a autorisé la purge, et le manifeste du corpus de la démo manquait jusque-là.
- **Purge et retrait du DBT.** 885 séries DBT purgées (69,7 Go), puis 177 séries qu'aucun corpus
  ne lisait, supprimées à la demande (14,1 Go ; c'était tout le split test officiel : 60 cancers,
  61 bénins, 56 normaux), puis les corpus, le catalogue et les tables (11,2 Go) : environ 95 Go au total.
  L'app ne dépend d'aucun fichier ni module DBT : elle ne charge que `app`, `imaging` (dataset,
  metrics, unet), `inference` et `config`, et son préflight passe sans ces données.
- **Purge IRM.** 186 patients copiés en NIfTI natif (26,1 Gio, contre 59,4 Gio de DICOM), chaque phase
  relue identique ; 822 séries supprimées (63,8 Go). Restent 12 dossiers sans copie en silver, donc
  non supprimés : 7 séries de `Breast_MRI_106`, `_120` et `_203` (un canal manque, ils sont hors des
  deux corpus) et 5 séries qui ne sont pas dynamiques.
- **L'ordre des coupes des boîtes est celui d'`InstanceNumber`**, qui décroît le long de z (144 → 1
  pour z de −77,7 à +79,6 mm). Le rehaussement tombe dans la boîte pour 8 patients sur 8 avec cet
  ordre, contre nul ou négatif pour 6 sur 8 avec l'ordre spatial.
- **Le recalage par information mutuelle dégradait l'alignement** : la corrélation pré/post baissait
  dans 7 cas sur 8 (0,61 → 0,32 ; jusqu'à 7 mm inventés), alors que les deux séries partagent la même
  grille. Avec la corrélation et la garde d'acceptation, sur 6 cas réels : accepté 6 fois, décalages
  de 0,13 à 0,99 mm, corrélation en hausse à chaque fois.
- **Un indice mal conçu, corrigé par la mesure.** Le premier indice de « boîte lâche » comparait les
  coins de la boîte à son centre : ces coins sont vides par construction autour d'une lésion
  ellipsoïdale, il valait ≈ 0 même sur une boîte parfaite et alertait 5 cas sur 6. Remplacé par un
  contraste de rehaussement.
- **Défauts des tests eux-mêmes.** Un test de « pas de nouveau téléchargement » sans faux
  téléchargeur passait même quand la garde était retirée (le téléchargement réel échouait sans bruit
  hors ligne) ; un extracteur nommé `extract_dicom_mri_images` interrogeait en réalité la collection
  DBT.
- **Tests** : 210, 46 pour le module nnU-Net. Sept mutations sur huit attrapées pour la purge ; la
  huitième est équivalente (la primitive refuse d'elle-même un patient sans volume silver).

**Non vérifié.** Le recalage corrigé n'a été validé que sur 6 cas réels ; le seuil de contraste (1,0 σ)
et le choix des canaux (pré + post2) sont des hypothèses marquées comme telles dans `config.yaml` ;
la construction complète des 186 patients et le rapport de QC sur 10 cas n'ont pas encore été
exécutés à l'échelle ; ni Docker ni la CI n'ont tourné.

**Prochain test.** Construire les 186 cas, lire le rapport de QC, puis entraîner nnU-Net en 5 plis et
mesurer la sensibilité lésionnelle en FROC (« Prochaines pistes »). *Fait pour la construction et le
QC : voir §4.19.*

### 4.19 Construction complète du corpus nnU-Net (2026-09-21 → 09-22)

**Conclusion.**
- **185 cas construits, 4 écartés avec leur raison** dans `exclusions.csv` : `Breast_MRI_106`, `_120`
  et `_203` (phase manquante, à l'ingestion) et `Breast_MRI_118` (lésion hors du masque de l'organe,
  au traitement). Export `Dataset501_DukeDCEBreast` écrit : 185 cas, 2 canaux (pré, post2), 185
  pseudo-masques, en liens physiques vers silver. Rapport de QC sur 10 cas tirés avec la graine 42,
  plus un agrégat sur les 185 (`data/silver/dce_mri_nnunet/qc/`).
- **Coût réel : médiane 55 s par cas** (quartiles 41 et 122 s), pas les ~30 s annoncées sur 6 cas ;
  le recalage domine. Trois durées aberrantes (`_039`, `_161`, `_202` : 3,4 à 10,9 h) tombent dans des
  étapes qui prennent quelques secondes ailleurs (N4, masque, recalage) : c'est la mise en veille de la
  machine, pas du calcul. L'ingestion : médiane 15 s par patient.
- **Le recalage gardé fonctionne à l'échelle** : accepté 175 fois, refusé 10 fois (aucun gain de
  corrélation, identité conservée et journalisée). Translation médiane 0,39 mm, maximum 5,6 mm.
- **Le masque de l'organe a deux défauts**, que rien n'arrête aujourd'hui :
  - `Breast_MRI_017` : le masque couvre **1,6 % du champ** (médiane du corpus : 31 %). La lésion y
    tombe par chance, donc le cas passe **sans aucune alerte** ; ses canaux sont normalisés sur une
    région fausse.
  - `Breast_MRI_118` : le masque (« plus grande composante ») ne contient **aucune partie** de la
    lésion. Le cas est écarté, ce qui est correct, mais c'est le masque qui est en cause, pas la boîte.
- **L'indice de contraste ne discrimine pas** : médiane 0,79 σ, et son seuil de 1,0 σ alerte sur
  **118 cas sur 185**. Avec les 3 cas dont la seule anomalie est un recalage refusé, 121 cas sont
  signalés. Une alerte sur deux tiers du corpus n'est plus une alerte.
- **Répartition des scanners** (seule variable d'acquisition disponible) : 8 modèles, de 9 à 55 cas ;
  Siemens Avanto 1,5 T est le plus fréquent (55).
- **Pseudo-masques** : volume médian 8,1 cm³ (de 0,17 à 477 cm³) ; le plus petit axe garde au moins
  7 voxels pour 95 % des lésions (minimum 4,8), cohérent avec l'espacement de 1,25 mm choisi pour que
  90 % en gardent 8.

**Non vérifié.** Une vérification ponctuelle du 2026-09-22 (script non versionné) confirmait l'ordre
`InstanceNumber` sur 156 cas sur 161 (rehaussement post − pré brut dans la boîte : médiane 1,67 σ,
contre 0,11 σ avec l'ordre inversé) et trouvait l'indice du pipeline peu corrélé à cette mesure brute
(0,60) : ces deux chiffres ne sont reproductibles par aucun code du dépôt. Le masque n'a été inspecté
visuellement que sur les 10 cas du QC.

**Prochain test.** Corriger le masque de l'organe (garde sur la fraction du champ, et plus d'une
composante quand les deux seins sont séparés), remplacer l'indice de contraste par le rehaussement
post − pré brut, reconstruire (~6 h), puis entraîner nnU-Net. `nnunetv2` n'est pas installé : dans
l'environnement principal il ferait changer de version `torch` et `numpy`, d'où un environnement
séparé à décider.

---

## Prochaines pistes

État de chaque piste, issu du journal :

| # | Piste | État |
|---|---|---|
| 1 | Construire les 186 cas nnU-Net, lire le QC, exporter `nnUNet_raw` | **Fait** : 185 cas, 4 écartés, export et QC écrits (§4.19) |
| 1 bis | Corriger le masque de l'organe (`_017`, `_118`) et l'indice de contraste, puis reconstruire | À faire avant l'entraînement (§4.19) |
| 2 | Métrique principale : composantes connexes 3D, FROC à 0,5 / 1 / 2 / 4 faux positifs par examen, sensibilité par taille de lésion | À faire : aujourd'hui sensibilité et faux positifs à un seul seuil, par coupe (§4.3) |
| 3 | Entraîner nnU-Net v2 en 5 plis, 3 graines, intervalles de confiance | À faire (`nnunetv2` est dans l'extra, non installé ici) |
| 4 | Valider les canaux : pré + post2 est une hypothèse ; comparer à la soustraction seule et aux quatre phases | À faire |
| 5 | Augmentations IRM (TorchIO), mirroring gauche-droite à valider comme anatomiquement licite | À faire |
| 6 | Stratification du découpage : le fabricant du scanner est la seule variable disponible ; contrôle de biais scanner | À faire |
| 7 | Sortie « 5 coupes candidates » du classifieur de coupe (16/28 patients en top-5) | À mesurer (§4.3) |
| 8 | Entraînement en fp32 de bout en bout pour la divergence NaN résiduelle | À faire (§4.2) |
| 9 | Mesurer le top-1 du choix de coupe sur des IRM neuves non annotées | À faire (§4.16) |

**Protocole fixé avant tout essai** : une piste à la fois, hyper-paramètres annoncés d'avance et
non ajustés ensuite, un résultat négatif publié comme tel.

---

## État d'avancement et feuille de route

### Fait

| Date | Livré |
|---|---|
| 2026-07-26 → 08-02 | U-Net DCE-MRI, banque memmap (×7,1), évaluation avec IC par patient, classifieur de coupe (0 % → 43 %) |
| 2026-08-18 | Docker, validation de schéma, manifestes de lignage, flow Prefect DCE-MRI |
| 2026-09-12 → 09-17 | Piste DBT : appariement, corpus, classifieur d'examen, catalogue, dbt, stockage S3 — **mesures négatives, retirée le 2026-09-20** (§4.4 à §4.15) |
| 2026-09-19 | Chemin « nouvelle IRM » : `preprocess_dce_mri_exams`, `dce_subtraction` en définition unique, défaut `crop` aligné sur le corpus, code hérité IRM retiré (§4.16) |
| 2026-09-20 | Parcours de démo mesuré de bout en bout : installation dédiée (68 paquets → 17), préflight qui analyse un vrai cas, port occupé détecté, formats de l'interface alignés (§4.17) |
| 2026-09-20 | Médaillon bronze → silver → gold, bronze purgé une fois en silver, `pyproject.toml` seul fichier de dépendances, corpus nnU-Net avec son module, code et données DBT retirés (§4.18) |
| 2026-09-21 → 09-22 | Corpus nnU-Net construit : 185 cas, export `Dataset501_DukeDCEBreast`, rapport de QC ; deux défauts du masque de l'organe relevés (§4.19) |
| 2026-09-26 | Audit de la démo : les trois cas, l'API et les erreurs vérifiés dans l'app ; deux chiffres faux et l'encadré DBT retirés des pages (« Écarts doc ↔ code ») |
| 2026-09-16 | **P0 portfolio** : README réorienté data engineering, licence MIT, citations TCIA, GIF de démo, documentation unique en français |

### Feuille de route

| Priorité | Tâche | État |
|---|---|---|
| P0 | Présentation : README, schéma, licence, GIF | **Fait** |
| P1 | Délai maximal sur les requêtes TCIA, échecs de téléchargement détectés | **Fait** |
| P1 | Tests d'orchestration exécutés en CI | **Fait** (job `orchestration`) |
| P1 | Médaillon et purge du bronze | **Fait**, exécuté (DBT puis IRM) |
| P1 | Corpus nnU-Net : ingestion, traitement, export, QC | **Fait**, construit (185 cas) ; masque de l'organe à corriger (§4.19) |
| P2 | Structure `src/`, découpage de `TransformData.py`, build Docker en CI, registre de modèles | À faire |
| — | Entraînement nnU-Net et FROC | À faire (« Prochaines pistes ») |

### Points ouverts

| Point | Détail |
|---|---|
| Bronze IRM résiduel | 12 dossiers (0,6 Gio) sans copie en silver : 7 séries des patients 106, 120 et 203, et 5 séries non dynamiques ; à supprimer sur demande |
| Canaux nnU-Net | pré + post2 est une hypothèse, à comparer (piste 4) |
| Seuil de contraste | 1,0 σ alerte sur 118 cas sur 185 : l'indice lui-même est à remplacer (§4.19) |
| Masque de l'organe | 1,6 % du champ sur `Breast_MRI_017` sans aucune alerte ; lésion hors masque sur `_118` (§4.19) |
| Environnement nnU-Net | `nnunetv2` non installé ; l'installer dans l'environnement principal changerait `torch` et `numpy` : environnement séparé à décider |
| Test bit à bit sur DICOM réel | Toujours ignoré depuis la purge IRM : aucun patient n'a encore ses deux phases en bronze. La garantie du §4.16 n'est plus exercée par la suite |
| Site des acquisitions | Inconnu dans Duke : impossible de stratifier par site |
| Trois patients sans phase pré ou post2 | `Breast_MRI_106`, `_120`, `_203` : sautés par les deux corpus ; leurs séries restent en bronze |
| 840 dossiers annoncés, 834 mesurés | Cause de l'écart non établie |
| Bug NaN fp16 | Divergence repoussée à l'époque 15, non résolue ; checkpoint servi antérieur (§4.2) |
| IC du top-1 à 43 % | Aucun code ni artefact versionné ne produit l'intervalle cité |
| Erreur `cudaErrorIllegalAddress` | Observée une fois sur `/demo/1`, non reproduite |
| Registre de traitement RGPD | Une page à écrire : base légale, nature des données, finalité, conservation, sécurité |
| Coupe choisie sur une IRM neuve | 43 % de top-1 (§4.3) : l'examen est préparé et servi correctement, la coupe reste le maillon faible |
| Nom de produit et logo | À choisir |

---

## Écarts doc ↔ code

**Principe** : la documentation est relue **contre le dépôt qui tourne**, pas contre
elle-même, et chaque écart corrigé reste noté ici — ce sont les chiffres qu'un relecteur
vérifie en premier.

**Le nombre de tests a dérivé quatre fois** : « 202 » et « 209 » ont été écrits sans
mesure, puis « 176 », mesuré le 2026-09-15, est devenu faux dès les tests suivants ;
le badge du README est resté à « 254 » quand le texte disait déjà 267. Un nombre de
tests se recompte, il ne s'estime pas — et le badge se recompte avec le texte.

| Date | Écart | Correction |
|---|---|---|
| 2026-08-18 | 68 tests annoncés, 87 réels ; chemin d'un rapport tabulaire faux | Corrigés |
| 2026-09-12 | Pastille « Confiance 100 % » (maximum par pixel constant) ; mesure « hors échantillon » annoncée sur un modèle ajusté sur 569/569 ; chiffres d'imagerie affichés sur `/biopsie` | Retirés de l'écran (P0), 9 tests de rendu |
| 2026-09-15 | 202 tests annoncés, 176 réels ; backend `unet` documenté mais impossible ; checkpoint mal documenté ; « aucun manifeste » alors que deux existaient ; « ~80 Go » pour 138 Go | Corrigés |
| 2026-09-16 | 176 tests annoncés, 198 réels ; « 0,76 s par volume » pour 0,825 s ; « ~110 ms » pour ~70 ms mesurés ; « ~200 Mo par patient » pour ~310 Mo ; taille d'image Docker jamais mesurée | Corrigés |
| 2026-09-16 | « 222 tests » en local contre 211 en CI, noté « non expliqué » ; « 152 normaux, non investigué » | Expliqués (§4.14) : 12 tests d'orchestration sans Prefect + 1 test Windows ; 2 normaux hors tirage |
| 2026-09-19 | « La couche brute DCE-MRI n'est pas sur cette machine », écrit dans §4.16, l'ADR 0013 et la PR #31 | **Faux** : 60 Go et 840 séries étaient dans `tcia/duke_mri/`, que le balayage sautait faute de `.dcm` à sa racine. Corrigé, et le chemin DICOM → `.npz` est désormais vérifié bit à bit sur données réelles (§4.16) |
| 2026-09-19 | Badge README « 254 tests » contre 267 dans le texte ; `crop=True` par défaut alors que le corpus servi est en pleine trame ; message d'erreur d'`imaging/dataset.py` renvoyant à une fonction cassée ; `SimpleITK`/`itk`/`itkwidgets` déclarés mais importés nulle part | Corrigés (§4.16), badge recompté à 287 |
| 2026-09-20 | « Port déjà utilisé → `--port 5001` » laissait croire qu'une erreur s'affichait : sous Windows le second lanceur affichait son bandeau de succès ; zone de dépôt annonçant DICOM/NIfTI sous un backend qui ne lit que `.npz` ; préflight qui ne chargeait jamais le modèle ; badge « 288 tests » | Corrigés (§4.17), badge recompté à 300 |
| 2026-09-20 | « 138 Go » (binaire) et « 82,6 Go de DBT » (décimal) additionnés dans le même paragraphe ; « 1 047 séries » (15 téléchargées depuis) ; « écart jamais expliqué » entre 186 volumes et 840 séries | Mesurés à nouveau, unité précisée, écart expliqué ; 840 dossiers IRM annoncés, 834 mesurés, cause non établie |
| — | `models/dce_mri_p2_negfix/` nomme une expérience | Ouvert (le renommer casserait la démo) ; son `eval_report.json` cite encore `data/preprocessed_data/dce_mri_p2` (le chemin d'avant le médaillon), mesure historique laissée telle quelle |
| 2026-09-20 | Une fonction nommée `extract_dicom_mri_images` téléchargeait en réalité la collection BCS-DBT, et non de l'IRM | Supprimée avec le code DBT |
| 2026-09-26 | Page « Comment ça marche » : « Entraînement : 186 patients » pour 130 (186 est le corpus, découpé 130 / 28 / 28) ; page de résultat : « 0/186 sur les patients de test » alors qu'il n'y en a que 28 (le 0/186 porte sur tout le corpus, §4.2) ; encadré « Et y a-t-il un cancer ? » présentant le classifieur DBT comme « servi par un autre modèle » six jours après son retrait ; README et doc annonçant le corpus nnU-Net « à construire » alors que 185 cas l'étaient ; « ~30 s par patient » pour 55 s mesurées ; ligne ci-dessus citant `data/silver/…` pour `data/preprocessed_data/…` | Corrigés ; encadré retiré et son absence épinglée par un test ; badge recompté à 209 |

---

## Partie 3 — Charte graphique

Appliquée à l'application (`app/templates/base.html`, cadre de lésion dans `inference.py`).

**Positionnement** : *instrument de diagnostic* — rigueur, lisibilité radiologique. Inspiré
du vocabulaire de la perfusion DCE (cinétique de rehaussement) : fonds sombres de station
de lecture, une couleur froide « signal », un accent chaud « rehaussement ». **Interdit** :
ruban rose, dégradés « féminins », cœurs, imagerie compassionnelle. Ton sobre, factuel,
jamais alarmiste, toujours accompagné de *Research Use Only — Not for diagnostic use*.

| Rôle | Token | Hex |
|---|---|---|
| Fond principal | `--bg` | `#0B0F14` |
| Surface | `--surface` | `#141A22` |
| Surface haute | `--surface-2` | `#1E2733` |
| Bordure | `--border` | `#2A3644` |
| Texte / secondaire | `--text` / `--text-muted` | `#E8EDF2` / `#93A1B0` |
| **Primaire (signal froid)** | `--primary` / `--primary-700` | `#2FB6C9` / `#1B7F8E` |
| **Accent (rehaussement)** | `--accent` | `#FF7A59` |
| Accent secondaire (perfusion haute) | `--accent-2` | `#F2C14E` |
| Succès / alerte / danger | `--success` / `--warning` / `--danger` | `#3FB98A` / `#E4B34A` / `#E5544B` |

Superposition de lésion : rampe `#1B7F8E → #2FB6C9 → #F2C14E → #FF7A59`, opacité 45–60 %.
Pendant clair (documents) : fond `#F7F9FB`, surface `#FFFFFF`, texte `#0B0F14`, bordure
`#DCE3EA`.

Typographie (licences OFL) : **Space Grotesk** (titres), **Inter** (corps), **IBM Plex
Mono** (mesures). Échelle 12 · 14 · 16 · 20 · 24 · 32 · 40 px ; interlignage 1,5 (corps),
1,15 (titres).

**À faire** : accent chaud réservé à la lésion et aux actions principales, police mono pour
tout chiffre, contraste AA minimum. **À éviter** : plus d'un accent chaud par écran,
superposition opaque qui masque l'anatomie, chiffres de performance présentés comme
cliniques.

---

## Développement

```bash
pip install -e ".[all]"   # la CI, elle, n'installe que .[dev,data]
ruff check .
pytest                    # 209 tests, sans GPU ni jeu de données
```

La [CI](.github/workflows/ci.yml) a deux jobs à chaque push et pull request : `check` (ruff + pytest,
installation volontairement étroite : PyTorch CPU, numpy, pandas, pydicom, flask) et `orchestration`
(Prefect, tests du flow DCE-MRI). Ajouter un test qui importe un nouveau module impose de l'ajouter à
l'extra qui convient. Les tests qui demandent SimpleITK (`tests/test_mri_nnunet_pipeline.py`) sont
ignorés en CI, où l'extra `nnunet` n'est pas installé ; les fonctions pures du corpus nnU-Net
(conversion boîte → pseudo-masque, normalisation dans le masque, choix de l'espacement) y tournent.
La suite couvre les définitions de métriques, le contrat de stockage, la logique d'orchestration et de
purge, le corpus nnU-Net, le rendu des pages, et vérifie que git **suit** bien les artefacts de la démo.

Les pipelines journalisent via `logging` (`logging_setup.py`), avec horodatage et copie
sous `logs/` ; `BREASTCANCER_LOG_LEVEL=DEBUG` augmente le volume.

**Scripts** : `scripts/make_demo_cases.py` (cas de démo), `scripts/make_demo_gif.py` (GIF du
README) — le second demande Playwright, hors dépendances du projet.

---

## Licence et données

Le **code** est sous [licence MIT](LICENSE).

Les **données d'imagerie** n'en relèvent pas. Duke-Breast-Cancer-MRI est distribuée sous
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), avec citation obligatoire. Cela vaut
pour les trois cas de démo versionnés et pour le GIF, qui en sont dérivés : réutilisables avec
attribution, à des fins non commerciales uniquement. La collection BCS-DBT, sous la même licence, a
servi aux mesures archivées du §4.4 à §4.15.

- Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E., Ghate, S. V.,
  Walsh, R., & Mazurowski, M. A. (2021). *Dynamic contrast-enhanced magnetic resonance
  images of breast cancer patients with tumor locations* [Data set]. The Cancer Imaging
  Archive. <https://doi.org/10.7937/TCIA.e3sv-re93>
- Buda, M., Saha, A., Walsh, R., Ghate, S., Li, N., Swiecicki, A., Lo, J. Y., Yang, J., &
  Mazurowski, M. (2020). *Breast Cancer Screening – Digital Breast Tomosynthesis
  (BCS-DBT)* (Version 5) [Data set]. The Cancer Imaging Archive.
  <https://doi.org/10.7937/E4WT-CD02>
