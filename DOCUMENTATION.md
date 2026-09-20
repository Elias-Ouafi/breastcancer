# Documentation du projet

> Outil de recherche, pas un dispositif
> médical. Aucune décision clinique ne doit en dépendre.

## Sommaire
1. [Contexte et objectif](#contexte-et-objectif)
2. [Cible chiffrée et voie retenue](#cible-chiffrée-et-voie-retenue-2026-09-12)
3. [Données](#données)
4. [Pipeline : fonctionnement et commandes](#pipeline--fonctionnement-et-commandes)
5. [Catalogue de métadonnées (DuckDB + dbt)](#catalogue-de-métadonnées-duckdb--dbt)
6. [Stockage objet (S3 / MinIO)](#stockage-objet-s3--minio)
7. [Démo et application web](#démo-et-application-web)
8. [Décisions d'architecture (ADR)](#décisions-darchitecture-adr)
9. [Partie 4 — Journal des mesures](#partie-4--journal-des-mesures)
10. [Prochaines pistes pour l'étape 1](#prochaines-pistes-pour-létape-1)
11. [État d'avancement et feuille de route](#état-davancement-et-feuille-de-route)
12. [Écarts doc ↔ code](#écarts-doc--code)
13. [Partie 3 — Charte graphique](#partie-3--charte-graphique)
14. [Développement](#développement)
15. [Licence et données](#licence-et-données)

---

## Contexte et objectif

### La question

Un examen de dépistage mammaire (tomosynthèse DBT ou IRM) en entrée, une réponse en
sortie : **y a-t-il un cancer ?** 
Le projet utilise uniquement des données publiques de [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) (TCIA).

### Deux objectifs

1. **Contribuer, à mon niveau, au secteur médical** en construisant l'outil de bout en
   bout, honnêtement mesuré.
2. **Un projet de portfolio de data engineering** :
   ingestion de 138 Go de DICOM, stockage en couches, validation, lignage, orchestration,
   catalogue de métadonnées en SQL/dbt, CI, Docker. Depuis cette date, les livrables de
   data engineering passent avant les nouvelles expériences de modélisation.

### Deux modalités, deux rôles
- **DBT / mammographie** (`Breast-Cancer-Screening-DBT`) porte la **détection** (étape 1).
- **IRM dynamique DCE-MRI** (`Duke-Breast-Cancer-MRI`) porte la **localisation** d'une
  lésion une fois l'examen jugé suspect. C'est le modèle servi dans la démo.

---

## Cible chiffrée et voie retenue

### La cible
Point de fonctionnement visé, **au niveau patient**, provenant de Santé publique France, dépistage organisé 50-74 ans :

| Mesure | Cible | Actuel |
|---|---:|---|
| Sensibilité | **82,8 %** | 100 % |
| Spécificité | **91,4 %** | 0 % |
| Spécificité par coupe | — | 0,03 % (99,97 % des coupes saines alarment) |

Sous hypothèse binormale, ce couple correspond à une **ROC-AUC patient d'environ 0,95**.
Repères de contexte (pas des cibles) : faux négatifs > 15 % (MSD), 85 à 90 % des
anomalies détectées ne sont pas des cancers (MSD), VPP 11,3 % en 2020 contre 7,8 % en
2008 (SpF).

### La voie retenue : bascule de l'étape 1 sur DBT

Trois raisons, indépendantes, rendaient la cible inatteignable sur le corpus IRM :

1. **Mauvaise modalité** : les chiffres visés sont ceux de la mammographie de dépistage.
2. **Mauvaise population** : Duke est une cohorte diagnostique, prévalence 100 %, et le
   prétraitement écarte en plus tout patient sans boîte. Une spécificité ne se mesure pas
   sans négatifs.
3. **Le piège de la solution évidente** : cancers de Duke + négatifs d'ailleurs donnent
   un modèle qui apprend le scanner et une AUC de 0,99 qui ne vaut rien. Les deux classes
   doivent venir de la **même collection**.

`Breast-Cancer-Screening-DBT` lève les trois : collection de dépistage, majoritairement
normale. La bascule n'annule pas le travail IRM : le U-Net garde sa fonction de
localisation, il quitte seulement le chemin critique de l'étape 1.

### Ce que la cible coûte en données

| Objectif de mesure (jeu de test seul) | ± 5 points | ± 3 points |
|---|---:|---:|
| Sensibilité 82,8 % → patients avec cancer | ~220 | ~610 |
| Spécificité 91,4 % → patients sans cancer | ~121 | ~335 |

**La collection entière ne contient que 89 patients cancer.** La spécificité est donc
mesurable finement (4 581 normaux), la sensibilité non : un test à 20 % laisse ~18
cancers (IC ±17 points) ; même en consacrant les 89 cancers au test, on resterait vers
±8 points. BCS-DBT ne peut pas départager 82,8 % de 76 % ou 94 %. Tout point obtenu doit
être publié avec son IC et **jamais présenté comme une comparaison au programme
national**.

---

## Données

### Collections

| Collection | Contenu | Rôle | Licence |
|---|---|---|---|
| **Breast-Cancer-Screening-DBT** (BCS-DBT) | 5 060 patients, tomosynthèses DICOM + 9 tables d'annotations | détection (étape 1), catalogue | CC BY-NC 4.0 |
| **Duke-Breast-Cancer-MRI** | IRM dynamiques DICOM de patientes avec cancer, boîtes de lésion | localisation (démo) | CC BY-NC 4.0 |

Espace disque mesuré le 2026-09-20 : **138 Go** sous `data/bronze/tcia/`, dont 78 Go de
séries DBT (1 062 séries) et 60 Go d'IRM (834 dossiers : 829 séries DCE de 189 patients,
plus 5 autres séries). Unités et détail dans l'arbre ci-dessous.

### Les tables BCS-DBT

Neuf tables, trois par split (train, validation, test), téléchargées par
`ExtractData.download_dbt_tables()` :

| Table | Contenu | Lignes (3 splits) |
|---|---|---:|
| `BCS-DBT-labels-*.csv` | statut par vue : Normal / Actionable / Benign / Cancer — **seule source du mot « normal »** | 22 032 |
| `BCS-DBT-file-paths-*.csv` | inventaire : (PatientID, StudyUID, View) et chemin de chaque série | 22 032 |
| `BCS-DBT-boxes-*.csv` | une boîte par lésion annotée : coupe, x/y/largeur/hauteur, `Class` (benign/cancer), `AD` | 435 (224 + 75 + 136) |

Statut patient, lu **à sa pire vue** (cancer > benign > actionable > normal) :

| Collection entière | Patients |
|---|---:|
| normal | **4 581** |
| actionable | 278 |
| benign | 112 |
| cancer | **89** |
| **Total** | **5 060** |

Ces chiffres sont reproduits indépendamment par le code Python
(`TransformData.dbt_patient_status`) et par le catalogue SQL.

Deux pièges propres à ces données :
- **le tag DICOM de latéralité est faux** : il lit `L` sur toutes les séries (§4.4) ;
- **les DICOM ne portent aucun tag d'espacement de pixel** : une fenêtre constante en
  pixels ne vaut fenêtre constante en millimètres que parce que la géométrie du détecteur
  est constante (2 457 lignes partout).

### Stockage en médaillon : bronze → silver → gold

```
data/                                   159 Go   ignoré par git, sauf les trois cas de démo
├── bronze/                             138 Go   zone de transit : la source telle que publiée,
│   │                                            jamais réécrite, supprimée une fois la série
│   │                                            en silver
│   └── tcia/                           138 Go   tables d'annotations BCS-DBT + séries DICOM
│       ├── <SeriesInstanceUID>/         78 Go   séries DBT, à plat (1 062 séries)
│       └── duke_mri/                    60 Go   séries DCE-MRI, un niveau plus bas (834 dossiers)
│                                                — cet écart de profondeur a déjà fait conclure à
│                                                tort que la couche IRM était absente (§4.16)
├── silver/                             9,7 Go   volumes z-normalisés + masques validés, un .npz
│   │                                            par série : la seule copie une fois le bronze purgé
│   ├── dbt/                            1,0 Go   preprocess_dbt_with_boxes : examens annotés,
│   │                                            recadrés sur la lésion
│   ├── dbt_exams/                      3,7 Go   preprocess_dbt_exams : tous les examens en
│   │                                            384×384, cancer ou non
│   └── dce_mri_p2/                     5,0 Go   preprocess_dce_mri_with_boxes (le modèle de la démo)
│                                                preprocess_dce_mri_exams pour un examen jamais annoté
└── gold/                              11,3 Go   dérivé de silver et reconstructible
    ├── slice_bank_p2/                  5,7 Go   banque de coupes (memmap)
    ├── exam_bank/                      5,6 Go   banque d'examens (memmap)
    ├── catalog/                       13,7 Mo   catalog.duckdb + Parquet (python -m catalog build)
    └── demo_cases/                    13,7 Mo   les trois cas de démo, seuls fichiers versionnés
models/                                70,9 Mo   checkpoints + les métriques qui les justifient
reports/                                6,2 Ko   rapports JSON
docs/img/                              0,8 Mo   images de la documentation
```

Tailles mesurées le 2026-09-20 avec `du -sb`, en unités binaires (1 Go = 2³⁰ octets, comme
`du -h`), **avant toute purge** : le bronze pèse encore 138 Go parce que la règle
ci-dessous n'avait pas encore été appliquée. L'ancien « 82,6 Go de DBT » était en unités
décimales (le catalogue divise par 10⁹) : c'est 83,7 Go décimaux, soit 78 Go binaires, et
les 1 047 séries du 2026-09-16 sont 1 062 depuis le téléchargement des 15 séries manquantes.

**Le bronze est une zone de transit, pas une archive.** Dès qu'une série est en silver
**dans tous les corpus qui la lisent**, son dossier DICOM est supprimé : la donnée n'est plus
conservée qu'une fois. C'est l'étape `purge` des deux flows (`download → preprocess → purge`) ;
`--keep-bronze` la désactive. Une seule fonction supprime du brut,
`TransformData.purge_bronze_series`, et elle **refuse plutôt que de parier** : le dossier doit
être un enfant direct du bronze, chaque fichier silver doit s'ouvrir comme un volume (un
fichier non vide mais tronqué ne suffit pas), et il faut au moins un fichier silver.

Le corpus « lésions » et le corpus « examens » lisent les mêmes séries DBT : purger après le
premier aurait privé le second de sa source, d'où « dans **tous** les corpus qui la lisent ».
« Détenu » veut dire manifeste **et** volume qui s'ouvre, pas seulement fichier présent : au
2026-09-20, 11 fichiers `.npz` de `silver/dbt/` existaient sans figurer au manifeste (écrits
depuis le 2026-09-17, corpus en cours de reconstruction), et leurs séries sont restées en bronze.

Ce qui **reste** en bronze : les tables d'annotations (petites, et tous les plans en partent),
les séries qu'aucun corpus ne lit (patients IRM sans annotation, séries DBT hors des splits
demandés) et celles qui ne sont pas encore en silver. Avant purge, le plan (`--dry-run`)
désignait **870 séries DBT sur 1 062 (63,8 Go)** ; côté IRM, **aucune** : le corpus
`dce_mri_p2/` n'avait pas de manifeste (antérieur à `lineage.py`), donc rien ne prouvait d'où
venaient ses volumes, et supprimer 60 Go de brut sur cette foi aurait été un pari. Régénérer le
corpus (`--from preprocess --force`) écrit le manifeste et débloque la purge.

**Le coût est assumé** : silver n'est plus « reconstructible » à volonté, c'est la source. Refaire
un corpus avec d'autres paramètres (marge de coupes, phase post-contraste, un split de plus)
demande de télécharger à nouveau.

Chaque chemin est défini **une seule fois** dans `config.py`. `data/` est ignoré par git,
sauf les trois cas de démo ; `models/` l'est aussi, sauf les deux checkpoints de la démo
et les rapports de validation croisée.

---

## Pipeline : fonctionnement et commandes

### Vue d'ensemble

| Étape | Où | Ce qu'elle garantit |
|---|---|---|
| **Extraction** | `ExtractData.py`, `http_timeouts.py` | Téléchargements qui reprennent là où ils s'arrêtent ; plafond exprimé en volume ajouté par l'appel ; examens normaux tirés avec une graine (les ID suivent le site et la date) ; délai maximal sur chaque requête ; une série n'est comptée que si son dossier existe. |
| **Orchestration** | `pipelines/dbt.py`, `pipelines/dce_mri.py` | Deux flows Prefect ; chaque étape de la chaîne DBT décide depuis un plan calculé hors ligne (ADR 0011) ; une étape `purge` vide le bronze une fois les séries en silver. |
| **Stockage** | `config.py` | Médaillon `bronze` → `silver` → `gold`, chemins définis une fois ; le bronze est purgé une fois la série en silver. |
| **Transformation** | `TransformData.py` | Annotations rattachées par **jointure**, jamais inférées ; étiquette tirée de la seule table qui dit « normal » ; même géométrie pour les deux classes. |
| **Validation** | `validation.py` | Dimensions, type, valeurs finies et masque binaire vérifiés au point unique d'écriture. |
| **Lignage** | `lineage.py` | Un `manifest.json` par dossier : révision git (suffixe `-dirty`), source, paramètres, statistiques par cas. Écrit en dernier : son absence signale une passe interrompue. |
| **Curation** | `imaging/exambank.py`, `imaging/slicebank.py` | Banques memmap qui paient la décompression une seule fois (×7,1 sur le temps d'époque). |
| **Catalogue** | `catalog/` | Tables, bronze restant, manifestes et scores joints dans DuckDB ; couches et tests dbt. |
| **Publication** | `objectstore/` | Tables, manifestes et tables mart en Parquet synchronisés vers un bucket S3 (MinIO en local) ; idempotent, sans suppression. |
| **Entraînement / évaluation** | `imaging/` | Découpages par patient, validation croisée, IC bootstrap par patient, seuil hors pli. |
| **Service** | `app/`, `Dockerfile`, `run_demo.py` | Flask HTML + API JSON ; image sans JVM ni ITK, lecture seule, non-root, boucle locale uniquement ; préflight qui analyse un cas réel et vérifie le port avant de servir. |

### Prérequis et installation

- Python ≥ 3.12 ; `torch` adapté à la machine (CPU : `--index-url https://download.pytorch.org/whl/cpu`).
- Accès TCIA via `tcia_utils` / `nbia` : les collections publiques ne demandent pas de clé.

```bash
pip install -e .                  # la démo seule : torch, Flask, numpy, Pillow
pip install -e ".[all]"           # tous les cas ci-dessous à la fois
pip install -e ".[data]"          # + pandas, pydicom, openpyxl : tables et DICOM
pip install -e ".[collect]"       # + client TCIA, python-gdcm : télécharger et décoder du vrai DICOM
pip install -e ".[catalog]"       # + duckdb, dbt-duckdb : construire et interroger le catalogue
pip install -e ".[orchestration]" # + Prefect, pour pipelines/
pip install -e ".[storage]"       # + boto3, pour publier vers un stockage objet S3
pip install -e ".[dev]"           # + pytest, ruff, moto
```

**Un seul fichier de dépendances : `pyproject.toml`.** Il n'y a plus de `requirements*.txt`,
donc plus de seconde liste qui dérive. Le socle (`dependencies`) est exactement la démo — la
démo n'importe que quatre paquets, et l'écart avec le reste est mesuré au §4.17 : 68 paquets /
355 Mo contre 17 / 153 Mo sur Windows. Chaque autre usage est un extra, et `all` les réunit :
un test échoue si un extra n'y figure pas, si le socle grossit, ou si le code importe un paquet
qu'aucun extra ne déclare. Le `Dockerfile` lit le socle avec `tomllib` plutôt que de le
recopier. La CI installe `.[dev,data,catalog,storage]` — pas `collect`, dont le client TCIA
n'est exercé par aucun test et transformerait un contrôle d'une minute en plusieurs.

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

### Chaîne DBT orchestrée (Prefect)

`pipelines/dbt.py` enchaîne `tables → download → preprocess → purge → catalog → publish` :

```bash
python -m pipelines.dbt --dry-run            # le plan, calculé hors ligne, rien n'est exécuté
python -m pipelines.dbt                      # exécute ce qui manque
python -m pipelines.dbt --from preprocess    # sans les étapes réseau
```

**Chaque étape décide depuis un plan, pas depuis « le dossier existe »** (ADR 0011). La
le bronze grossit dans le temps, et un corpus construit avant un téléchargement est
un dossier qui existe et qui est périmé. Chaque étape calcule donc hors ligne, à partir
des tables de la collection, ce qui devrait exister, le compare à ce qui existe et ne
s'exécute que s'il manque quelque chose :

| Étape | Plan | Exécution |
|---|---|---|
| `tables` | les 9 tables présentes ? | téléchargement, 3 reprises |
| `download` | séries des patients annotés (splits choisis) + échantillon de normaux (graine), lues dans l'inventaire, contre les dossiers du bronze **et** les séries déjà en silver | seuls les patients incomplets, 2 reprises ; une série ratée lève `DownloadIncomplete` |
| `preprocess` | séries que chaque corpus devrait contenir d'après le disque, contre les cas de son manifeste | corpus d'examens repris (seules les séries manquantes sont décodées) ; corpus de lésions reconstruit |
| `purge` | séries du bronze que **chaque** corpus qui les lit détient déjà en silver (manifeste et volume lisible) | suppression du dossier DICOM ; `--keep-bronze` pour l'éviter ; pas de reprise (elle refuse plutôt qu'elle n'échoue) |
| `catalog` | toujours | build DuckDB + dbt ; un test dbt `error` en échec fait échouer le flow |
| `publish` | `BREASTCANCER_S3_ENDPOINT` défini ? | synchronisation idempotente vers le bucket, 2 reprises ; sautée sinon |

Options : `--annotated-splits` (défaut : les trois), `--corpus-splits` (défaut :
`train,validation`, les corpus de toutes les mesures publiées), `--max-normal-patients`
(150), `--max-gb-added` (50), `--seed` (0), `--force`, `--keep-bronze`.

**Plan réel au 2026-09-16** (`--dry-run`, 10 s, aucun accès réseau) :

```
  tables      9/9 present -> skip
  download    201 annotated (train,validation,test) + 150 normal patients -> 1060 series planned, 1047 series on disk
              -> 15 missing across 9 patient(s): DBT-P03621, DBT-P03628, DBT-P03689, DBT-P03728, DBT-P04097, DBT-P04346...
  preprocess  lesion corpus (train,validation): 260 expected, 260 in manifest -> skip
  preprocess  exam corpus (train,validation): 870 expected, 870 in manifest -> skip
  catalog     always rebuilt -> data\gold\catalog\catalog.duckdb
```

**Plan réel au 2026-09-20** (même commande, après le passage en médaillon, avant purge) :

```
  download    201 annotated (train,validation,test) + 150 normal patients -> 1060 series planned, 1062 in bronze, 870 in silver
              -> 0 missing, skip
  preprocess  lesion corpus (train,validation): 275 expected, 260 in manifest -> 15 to build
  preprocess  exam corpus (train,validation): 885 expected, 870 in manifest -> 15 to build
  purge       870 of 1062 bronze series held in silver by every corpus that reads them -> 870 to delete
```

Les 15 séries à construire sont celles téléchargées depuis, les mêmes pour les deux corpus.
Aucune n'est supprimable : la purge ne part que de ce qui est prouvé en silver.

Le plan du 2026-09-16 recoupe le catalogue sans le lire : les 15 séries manquantes sont celles des 9
patients annotés jamais téléchargés, et le nombre de séries attendues par corpus,
recalculé depuis les tables et le disque, retombe exactement sur le contenu des
manifestes. Exécution réelle `--from preprocess` : deux corpus jugés à jour, catalogue
reconstruit, 36/36 tests dbt, flow `Completed` en 60 s (démarrage du serveur Prefect
temporaire compris).

**Robustesse des téléchargements.** `tcia_utils` 3.3.4 envoie ses requêtes sans délai
maximal et avale toutes les exceptions. Deux corrections :
- `http_timeouts.install(nbia)` ajoute un délai de 30 s à la connexion et de 600 s
  d'inactivité en lecture à chaque `get`/`post` du client. Un `socket.setdefaulttimeout`
  ne suffit pas : mesuré, une requête vers un serveur muet restait bloquée après 8 s,
  alors qu'un `timeout=` explicite lève `ReadTimeout` en 2 s. Le délai de lecture est
  calibré sur le journal réel (122 séries : médiane 13 s, p90 49 s, pire cas 565 s) ;
- une série n'est comptée téléchargée que si son dossier existe après l'appel ; avec
  `raise_on_failure`, le téléchargement termine ce qu'il peut puis lève
  `DownloadIncomplete`, ce qui déclenche la reprise Prefect.

### Chaîne DBT (commandes manuelles)

```python
import config
from ExtractData import download_annotated_dbt_series, download_dbt_tables, download_normal_dbt_series
from TransformData import preprocess_dbt_exams, preprocess_dbt_with_boxes

download_dbt_tables()                                   # les 9 tables, une fois

BOXES = [config.DBT_BOXES_TRAIN, config.DBT_BOXES_VALIDATION]
download_annotated_dbt_series(BOXES, max_patients=None, max_gb=25)
download_normal_dbt_series(max_patients=150, max_gb_added=50)  # ~310 Mo par patient

preprocess_dbt_with_boxes(boxes_csv=BOXES, file_paths_csv=config.DBT_FILE_PATHS,
                          output_dir="data/silver/dbt")   # corpus « lésions »
preprocess_dbt_exams(boxes_csv=BOXES)                                 # corpus « examens »
```

**Corpus « lésions » (`preprocess_dbt_with_boxes`).** Chaque série est rattachée à ses
boîtes par une jointure sur `(PatientID, StudyUID, View)` via l'inventaire `file-paths`
(ADR 0004). Un dossier absent de l'inventaire est ignoré et compté. Les pixels ne servent
qu'à détecter une étude stockée en miroir, qui est retournée ; le manifeste le trace. La
colonne `Class` est **lue et exigée** : chaque `.npz` porte `lesion_class`
(`benign`/`cancer`) et `label` 0/1, car le masque ne peut pas porter la classe — une
lésion bénigne peint les mêmes pixels qu'un cancer. `mask_classes=("cancer",)` ne peint
que les cancers ; par défaut les deux sont peints. Résultat : **260 séries, 132 patients
(76 bénins, 56 cancers), 0 masque vide, 0 avertissement, 1,00 Go, 56 min**.

**Corpus « examens » (`preprocess_dbt_exams`).** Le corpus sur lequel une décision
d'examen se mesure. L'étiquette vient des tables `labels`, lue à la pire vue ; une série
sans ligne de labels est ignorée et comptée, jamais supposée normale. `label` vaut 1
uniquement pour `cancer` (`actionable` et `benign` valent 0) et le mot est gardé dans
`exam_status`. **Une seule géométrie pour les deux classes** (ADR 0005) : trame entière
ramenée à 384×384 en gardant les proportions, complétée par des zéros, toutes les coupes
conservées, aucun recadrage ; le sous-échantillonnage moyenne au lieu d'échantillonner ;
le même retournement s'applique aux négatifs. Mesuré : 4,9 Mo par série compressée,
~14 s de décodage chacune, reprise possible (`skip_existing=True`) : une passe reprise
conserve dans le manifeste les cas qu'elle saute, et un volume sans entrée de manifeste
(passe interrompue) est reconstruit (§4.14). Résultat : **870 examens, 272 patients, 56
cancers**.

### Entraînement et évaluation

| Commande | Rôle |
|---|---|
| `python -m imaging.train --data-dir <dir> --epochs 25` | U-Net 2D de localisation (BCE + soft-Dice), découpage par patient, meilleur checkpoint + `segmentation_metrics.csv`. `--smoke-test` valide la boucle sans données. |
| `python -m imaging.evaluate --data-dir data/silver/dce_mri_p2 --checkpoint models/dce_mri_p2_negfix/unet_best.pt` | IC bootstrap par patient, sensibilité lésion, faux positifs par volume, temps d'inférence → `eval_report.json` + `eval_per_patient.csv`. |
| `python -m imaging.sliceclf --slice-bank data/gold/slice_bank_p2 --epochs 25` | Classifieur de coupe, sélectionné sur le top-1 (§4.3). |
| `python -m imaging.examclf --data-dir data/silver/dbt_exams --folds 5` | Tête de décision au niveau examen (MIL), validation croisée 5 plis par patient → `cv_report.json` + `cv_predictions.csv` (§4.7). |
| `python -m imaging.oppoint` | Point de fonctionnement depuis les prédictions hors pli, sans réentraîner → `reports/examclf_operating_point.json`, et affiche la section reprise au §4.11. |

---

## Catalogue de métadonnées (DuckDB + dbt)

Un fichier DuckDB unique qui rassemble tout ce qui décrit les données DBT : les 9 tables
d'annotations, les séries encore en bronze, le contenu de chaque corpus et le score
de chaque patient. Il s'interroge en SQL ; ses transformations sont un **projet dbt** et
chaque build exécute **36 tests dbt**.

```bash
pip install -e ".[catalog]"
python -m catalog build            # ~10 s sur la collection complète
python -m catalog checks --all     # rapport qualité du dernier build
python -m catalog tables
python -m catalog query "SELECT status, count(*) FROM mart.dim_patient GROUP BY ALL"
python -m catalog query --file catalog/queries/01_data_funnel.sql
python -m catalog docs             # site de documentation dbt, graphe de lignage compris
```

Sortie : `data/gold/catalog/catalog.duckdb` (7,6 Mo), les tables `mart` en
Parquet sous `parquet/`, les artefacts dbt sous `dbt_target/`. Tout est dérivé : le
supprimer ne perd rien.

**Pourquoi il existe.** Avant lui, « combien de cancers du split validation sont
téléchargés, et sont-ils tous dans le corpus d'examens ? » demandait de charger trois CSV,
deux manifestes et un listing de dossier dans pandas, et de réécrire la jointure. Chaque
réponse était un script jetable, jamais testé.

### Déroulé d'un build

| Étape | Outil | Ce qui se passe |
|---|---|---|
| 1. Charger `raw` | Python (`catalog/build.py`) | Ce que dbt ne sait pas faire : réunir des CSV aux schémas différents (`union_by_name`), lister le disque par `stat`, aplatir les manifestes JSON |
| 2. `dbt run` | dbt (`catalog/dbt/`) | Construit les vues `stg` et les tables `mart` |
| 3. `dbt test` | dbt | 25 tests génériques + 11 tests singuliers ; les lignes fautives de chaque test restent dans `qa` |
| 4. Enregistrer et exporter | Python | `qa.check_results`, `main.build_info`, export Parquet, remplacement atomique du fichier |

`dbt run` et `dbt test` sont séparés à dessein : `dbt build` sauterait les modèles
en aval d'un test en échec et cacherait les tables nécessaires à l'enquête. Un test en
échec de sévérité `error` rend un code de sortie non nul, mais le catalogue est quand
même écrit (`SELECT * FROM qa.<nom du test>`).

Le build écrit dans un dossier temporaire, **sous le nom de fichier définitif**, puis
remplace le fichier seulement quand tout a réussi : un build raté laisse l'ancien
catalogue intact. Le nom compte : dbt-duckdb inscrit le nom de la base (tiré du nom de
fichier) dans chaque vue, et un build dans `tmpXXXX.duckdb` renommé ensuite cassait
toutes les vues `stg` (§4.13).

### Couches

![Graphe de lignage dbt : les sources raw alimentent les vues stg, qui alimentent fct_series, dim_patient et les tests singuliers](docs/img/dbt-lineage.png)

*Graphe de lignage dbt (`python -m catalog docs`, capturé par
`scripts/make_dbt_lineage_png.py`). En vert, les sources `raw` ; en bleu, les modèles et
les tests singuliers. Les 25 tests génériques, déclarés en YAML, ne sont pas dessinés.*

| Couche | Contenu | Défini dans |
|---|---|---|
| `raw` | Sources telles que publiées, plus le fichier d'origine de chaque ligne (7 tables) | `catalog/build.py`, déclarées comme sources dans `_sources.yml` |
| `stg` | 7 vues, une par source : clés nettoyées, vues en minuscules, split tiré du nom de fichier, statut de pire drapeau | `catalog/dbt/models/staging/` |
| `mart` | 4 tables à granularité déclarée et testée | `catalog/dbt/models/marts/` |
| `qa` | Lignes fautives de chaque test + `check_results` | `catalog/dbt/tests/` et YAML des modèles |

Une macro garde les noms de schéma tels quels (`stg`, `mart`, `qa` au lieu de `main_stg`)
et les modèles de staging sont aliasés (`stg_dbt_labels` exposé en `stg.dbt_labels`).

### Tables `mart`

| Table | Granularité | Colonnes principales |
|---|---|---|
| `mart.fct_series` | une série de la collection (22 032) | `series_uid`, `patient_id`, `split`, `view_status`, `n_boxes`, `n_cancer_boxes`, `on_disk` (présente dans une zone), `in_bronze`, `bronze_bytes`, `in_lesion_corpus`, `in_exam_corpus`, `exam_label`, `n_slices`, `mirrored` |
| `mart.dim_patient` | un patient (5 060) | `status` (pire vue), `split`, `n_series`, `n_boxes`, `n_series_on_disk`, `n_series_in_bronze`, `fully_on_disk`, `bronze_gb`, `n_series_in_exam_corpus`, `examclf_score`, `examclf_fold` |
| `mart.collection_coverage` | split × statut | patients publiés, téléchargés, dans le corpus d'examens, notés, et Go encore en bronze |
| `mart.corpus_summary` | corpus silver | séries, patients, patients positifs, profondeur moyenne, miroirs, révision git de la passe |

### Tests de qualité

**25 tests génériques** (tous `error`) : `unique` + `not_null` sur les granularités
(`fct_series.series_uid`, `dim_patient.patient_id`, `stg_dbt_file_paths.series_uid`,
clé patient/étude/vue des labels, corpus/série des cas…), `accepted_values` (split,
statut, latéralité, incidence, classe de lésion, corpus, étiquette d'examen),
`relationships` (`fct_series.patient_id` → `dim_patient`).

**11 tests singuliers** (règles métier) :

| Test | Sévérité | Protège contre |
|---|---|---|
| `boxes_match_inventory` | error | une boîte qui n'appartient à aucune série (ADR 0004) |
| `patient_single_split` | error | un patient dans deux splits, donc une fuite train/test |
| `corpus_cases_match_inventory` | error | un volume rattaché au mauvais patient, étude ou vue |
| `exam_corpus_label_matches_labels` | error | une étiquette d'examen en désaccord avec la table labels |
| `lesion_corpus_label_matches_boxes` | error | une étiquette de lésion en désaccord avec la classe de la boîte |
| `predictions_patients_in_exam_corpus` | error | un score pour un patient que le classifieur n'a jamais vu |
| `predictions_label_matches_corpus` | error | une étiquette de score différente de celle du corpus |
| `labels_exactly_one_flag` | warn | une ligne de labels avec zéro ou plusieurs drapeaux |
| `file_paths_have_labels` | warn | une série sans ligne de labels (ignorée au prétraitement) |
| `bronze_series_already_in_silver` | warn | une série encore en bronze alors qu'un corpus silver la détient : purge en attente, ou run avec `--keep-bronze` |
| `disk_series_in_inventory` | warn | un dossier du bronze que l'inventaire ne liste pas |

**Sur les données réelles au 2026-09-20 : 35 sur 36, aucune erreur.** Le seul avertissement est
`bronze_series_already_in_silver`, 1 130 lignes (260 + 870), c'est-à-dire la purge qui n'avait pas
tourné : il se résorbe en l'exécutant. Le 2026-09-16, les 36 passaient. Des tests qui passent tous pourraient ne
jamais savoir échouer : `tests/test_catalog.py` exécute le vrai projet dbt sur une
collection miniature et y injecte six défauts (étiquette d'examen fausse, étiquette de
lésion fausse, cas sur le mauvais patient, score hors corpus, étiquette de score fausse,
clé de labels dupliquée), en vérifiant à chaque fois que le bon test échoue ; une série
promue qui traîne encore en bronze ne doit lever qu'un avertissement.

## Stockage objet (S3 / MinIO)

La couche de données se publie vers n'importe quel stockage compatible S3 : **MinIO** en
local, **AWS S3** (ou équivalent) dans le cloud. Le code ne parle que l'API S3 standard
(boto3).

```bash
pip install -e ".[storage]"
docker compose --profile storage up -d minio        # MinIO local (Docker requis)
export BREASTCANCER_S3_ENDPOINT=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin

python -m objectstore plan                   # ce qu'une synchronisation enverrait, rien n'est envoyé
python -m objectstore sync                   # tables, manifestes, catalogue en Parquet
python -m objectstore sync --dicom-sample 5  # + les DICOM bruts de 5 séries, pour la démonstration
python -m objectstore ls --prefix gold/
python -m pipelines.dbt --from publish       # la même chose, comme étape du flow
```

Le bucket (`BREASTCANCER_S3_BUCKET`, défaut `breastcancer`) et le point d'accès viennent de
l'environnement ; les identifiants viennent des variables AWS standard, jamais du code.
Les identifiants par défaut de `docker-compose.yml` ne valent que pour une instance locale,
publiée sur la boucle locale uniquement.

### Ce qui est publié, et où

Les clés reprennent les couches locales : un chemin dans le bucket dit à quelle couche il
appartient.

| Clé | Contenu | Taille |
|---|---|---:|
| `bronze/tcia/tables/BCS-DBT-*.csv` | les 9 tables d'annotations | 8,1 Mo |
| `silver/<corpus>/manifest.json` | le manifeste de lignage de chaque corpus | 0,5 Mo |
| `gold/catalog/<table>.parquet` | les 4 tables `mart` du catalogue | 0,9 Mo |
| `bronze/tcia/series/<SeriesInstanceUID>/…` | DICOM bruts, **uniquement sur demande** (`--dicom-sample`) | ~70 Mo par série |
| `_meta/last_sync.json` | trace de la dernière synchronisation : date, révision git, décompte, clés | — |

**Par défaut, les DICOM ne sont pas publiés** (78 Go de séries DBT dans le bronze) : on publie
ce qui est léger et utile à partager. La synchronisation sait envoyer les DICOM (envoi en plusieurs parties au-delà
de 64 Mo) et le fait pour un échantillon quand on le demande.

### Interroger le bucket directement

Les tables `mart` publiées en Parquet se lisent **sans rien télécharger au préalable**,
par exemple avec DuckDB et son extension `httpfs` :

```sql
INSTALL httpfs; LOAD httpfs;
CREATE SECRET (TYPE S3, KEY_ID '…', SECRET '…', ENDPOINT '127.0.0.1:9000',
               URL_STYLE 'path', USE_SSL false);
SELECT status, count(*) FROM 's3://breastcancer/gold/catalog/dim_patient.parquet' GROUP BY ALL;
SELECT count(*) FROM read_csv('s3://breastcancer/bronze/tcia/tables/BCS-DBT-labels-*.csv');
```

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

## Partie 4 — Journal des mesures

Journal daté, réduit à ce qu'il faut retenir : la **conclusion** de chaque mesure, échecs
compris, et le **prochain test** qu'elle appelle quand il y en a un. Les tableaux détaillés,
les incidents et les versions successives de la prose sont dans l'historique git. Les pistes
classées sont dans « Prochaines pistes pour l'étape 1 ».

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

### 4.4 Appariement boîte ↔ série DBT : le tag DICOM de latéralité est faux (2026-09-12)

**Conclusion.** Le tag DICOM lit `L` sur les 262 séries ; les pixels donnent R 134 / L 128.
Avant correction, **23 masques sur 147** tombaient sur du fond. La latéralité se déduit des
pixels, l'incidence de l'en-tête, et les 14 séries stockées en miroir (7 patients) sont
retournées. **Contrainte** : tout futur chemin d'inférence DBT doit appliquer
`TransformData.image_laterality`, sinon la moitié des examens arrive dans le mauvais repère.

**Prochain test.** Un test qui échoue si le chemin d'inférence du futur détecteur n'applique
pas cette règle.

### 4.5 Étape 2 en image : mesurée, et elle ne marche pas (2026-09-12, code retiré le 2026-09-14)

**Conclusion.** Bénin ou malin sur recadrage (130 patients, 55 cancers, validation croisée à
5 plis) : ROC-AUC patient **0,591** [0,491 – 0,693] avec recadrage étiré, **0,513**
[0,411 – 0,615] à échelle constante. Les deux IC contiennent 0,5 ; la perte descend
(0,83 → 0,27) pendant que l'AUC reste au hasard : mémorisation. Abandonnée, code retiré.

**Prochain test.** Aucun avant que l'étape 1 dispose d'un détecteur.

### 4.6 Appariement boîte ↔ série : la collection le dit, il suffisait de le lire (2026-09-13)

**Conclusion.** L'inventaire `file-paths` couvre les 262 dossiers : séries annotées 253 →
**260**, 4 masques déplacés (acquisitions répétées), 14 miroirs confirmés, 0 masque vide, en-tête
et jointure d'accord sur 262/262. La colonne `Class` donne 82 bénins / 59 cancers. Leçon : un
blocage réseau se re-teste avant d'être écrit au présent.

### 4.7 Tête de décision au niveau examen : mesurée, et elle n'apprend rien (2026-09-13)

**Conclusion.** Multi-instance (max sur 16 coupes ; 870 examens, 272 patients, **56 cancers**,
5 plis par patient) : ROC-AUC patient **0,457** [0,369 – 0,544], sensibilité 0 % à 0,5,
exactitude 79,4 % (= « toujours pas de cancer »). La perte ne descend dans aucun pli ; le
signal utile est < 1 % des coupes (5,4 peintes sur 71). Un sac de 32 ne change rien
(0,414 [0,334 – 0,497]).

### 4.8 Warm start : ne marche pas, et son diagnostic déplace le problème (2026-09-14)

**Conclusion.** Pré-entraîner l'encodeur sur l'étiquette par coupe : AUC patient **0,426**
[0,338 – 0,513]. L'AUC **par coupe** hors pli est de 0,502 (0,510 après MIL) : l'encodeur
n'apprend rien même avec une étiquette 10 fois plus dense, donc **le goulot n'est pas
l'agrégation**. La résolution est disculpée (boîte médiane 19×18 px dans la trame de 224).

### 4.9 Le signal est relatif à l'examen — pourquoi rien ne marchait (2026-09-14)

**Conclusion.** La lésion est plus brillante que sa propre coupe dans 117 examens sur 117
(d de Cohen 1,15). Le 99ᵉ percentile a une AUC de **0,532** mis en commun entre examens et de
**0,736** normalisé par examen : **le pouvoir discriminant est relatif à l'examen**, alors que
`examclf` comparait des scores absolus entre patients. Normaliser sur le tissu seul n'apporte
rien (0,520).

### 4.10 Le score relatif ne marche pas non plus, et une ligne de numpy bat le CNN (2026-09-14)

**Conclusion.** Le score relatif ne change rien (AUC **0,465** [0,382 – 0,551]). En intra-examen,
le 99ᵉ percentile (une ligne de numpy) bat le CNN : 0,732 contre 0,592. Sur la vraie question,
sans aucun modèle (272 patients), aucune statistique ne bat le hasard (0,366 à 0,442), deux sont
significativement en dessous : **trouver la lésion annotée n'est pas détecter un cancer** (deux
tiers des coupes peintes sont bénignes). La géométrie est réfutée : en résolution native
(blocs de 96 px, 60 cancers contre 60 normaux) l'AUC est de 0,451 [0,352 – 0,556] (384 px :
0,431 ; 224 px : 0,433). Ce qui distingue un cancer est la **forme** (spiculation, distorsion
architecturale), qui demande des features apprises sous la supervision la plus riche : les
**boîtes**. Les données ont suivi : 86 patients cancer sur disque. Leçon d'un téléchargement
figé 2 h 30 : compter des fichiers ne mesure pas un débit, lire l'horodatage du journal.

**Prochain test (retenu).** Un détecteur supervisé par boîtes en résolution native (Buda et
al. : 65 % de sensibilité à 2 faux positifs par sein), perte focale réutilisée, score d'examen =
maximum des détections, évalué par patient avec les hyper-paramètres annoncés avant l'essai.

### 4.11 Le point de fonctionnement, publié — et la VPP vaut la prévalence (2026-09-15)

**Conclusion.** Seuil visé à 82,8 % de sensibilité, calé hors du pli noté : sensibilité
**78,6 %** [67,2 – 88,9], spécificité **20,4 %** [15,3 – 25,9] (cible 91,4 %), VPP 20,4 % pour
une prévalence de 20,6 %. À cette sensibilité, le hasard donnerait 21,4 % de spécificité : le
modèle est en dessous. Le seuil calé sur quatre plis ne se transporte pas au cinquième, ce qui
mesure une échelle de scores qui ne veut rien dire d'un groupe de patients à l'autre (§4.9).
L'écran nomme désormais chaque modèle, son corpus et son échec, sans mélanger leurs chiffres.

**Prochain test.** Rejouer `python -m imaging.oppoint` sur les scores du détecteur du §4.10 :
la cible est 91,4 % de spécificité à 82,8 % de sensibilité, et une VPP au-dessus de la
prévalence.

### 4.12 Un catalogue de métadonnées DuckDB — et ce qu'il a vu dès le premier build (2026-09-16)

**Conclusion.** Build en 3,9 s, 14 contrôles sur 14, statuts recalculés en SQL (4 581 / 278 /
112 / 89) identiques à ceux du code. Il a vu d'emblée : 3 cancers jamais téléchargés, le split
test hors corpus, 152 normaux au lieu de 150, et un score moyen des normaux supérieur à celui
des cancers.

### 4.13 Le catalogue passe sous dbt — parité vérifiée ligne à ligne, et deux pièges (2026-09-16)

**Conclusion.** `stg` et `mart` deviennent un projet dbt (25 tests génériques, 11 singuliers,
**36/36** passent). Parité avec la version sans dbt : 0 ligne de différence dans les deux sens
(`EXCEPT ALL`) sur les 4 tables `mart`. Deux pièges : dbt-duckdb inscrit le nom de la base dans
chaque vue (le build se fait sous le nom définitif) et garde sa connexion ouverte (verrou sous
Windows, environnement fermé explicitement). Coût : 9,7 s contre 3,9 s.

### 4.14 La chaîne DBT orchestrée — et trois défauts trouvés en l'écrivant (2026-09-16)

**Conclusion.** Écrire le plan hors ligne a fait apparaître trois défauts, corrigés :
(1) les requêtes TCIA n'avaient aucun délai, et `socket.setdefaulttimeout` ne suffit pas (seul
`timeout=` explicite marche) → `http_timeouts.py`, `timeout=(30, 600)`, calibré sur 122 séries
(p90 49 s, pire réussite 565 s, blocage 8 537 s) ; (2) un téléchargement raté était compté
comme réussi → `DownloadIncomplete` ; (3) une passe reprise du corpus d'examens réécrivait un
manifeste d'un seul cas → les entrées des volumes sautés sont reportées. Plan hors ligne en
10 s, exécution réelle en 60 s, 36/36 tests dbt ; un job CI dédié fait tourner les tests des
deux flows. **254 tests**.

### 4.15 Stockage objet S3, et premier vrai run du flow DBT avec téléchargement (2026-09-17)

**Conclusion.** Premier run réseau du flow : 15 séries téléchargées, 0 échec, 1,1 Go en 2 min 44
(estimation de 0,5 Go fausse : elle ne comptait que les 3 patients cancer) ; le corpus de
lésions, jugé périmé (275 attendues, 15 absentes), est reconstruit. La synchronisation S3 est
vérifiée contre un serveur S3 réel (moto) sur les vraies données : la relance envoie 0 fichier
(19 inchangés), DuckDB relit le Parquet dans le bucket en 0,06 s. **267 tests**.

**Prochain test.** L'exécuter contre un MinIO réel : Docker n'est pas installé ici, il n'a
jamais tourné.

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

---

## Prochaines pistes pour l'étape 1

Revue de littérature du 2026-09-14, **largement réfutée l'après-midi même** (§4.8 à
§4.10). État de chaque piste :

| # | Piste | État |
|---|---|---|
| 1 | Pré-entraîner l'encodeur sur l'étiquette par coupe (warm start) | **Mesurée, négative** : 0,426 (§4.8) |
| 2 | Réunir les trois splits annotés (56 → 89 cancers) | **Données faites** : 86 cancers sur disque ; ne débloque rien seule |
| 3 | Top-k au lieu du max | **Annulée** (§4.9) : rien à agréger |
| 4 | Échantillonnage de sac orienté | Tombe avec la 3 |
| 5 | Pertes auxiliaires multi-tâches (vue, latéralité) | Tombe avec la 3 |
| 6 | Pré-entraînement auto-supervisé sur les 4 581 normaux | Non testée, non prioritaire (preuve mitigée dans la littérature) |
| 7 | Architecture globale + locale type GMIC / 3D-GMIC | Direction à moyen terme (validée sur DBT, mais entraînée sur 85 526 patients) |
| 8 | **Détecteur supervisé par boîtes en résolution native** (Buda et al.) | **Prochaine étape retenue** : 65 % de sensibilité à 2 faux positifs par sein dans la publication d'origine ; perte focale réutilisable |

**Protocole fixé avant tout essai** : une piste à la fois, chiffres d'hyper-paramètres
annoncés d'avance et non ajustés ensuite, un résultat négatif publié comme tel.

Sources : Buda et al., *A Data Set and Deep Learning Algorithm for the Detection of Masses
and Architectural Distortions in Digital Breast Tomosynthesis Images*, JAMA Netw Open
2021 · Shen et al., *GMIC*, MLMI 2019 · *3D-GMIC*, IEEE TMI 2023 · 1ʳᵉ place du concours
RSNA Screening Mammography Breast Cancer Detection 2023 · *Avg-TopK*, Expert Systems with
Applications 2023.

---

## État d'avancement et feuille de route

### Fait

| Date | Livré |
|---|---|
| 2026-07-26 → 08-02 | U-Net DCE-MRI, banque memmap (×7,1), évaluation avec IC par patient, classifieur de coupe (0 % → 43 %) |
| 2026-08-18 | Docker, validation de schéma, manifestes de lignage, flow Prefect DCE-MRI |
| 2026-09-12 | Colonne `Class` lue, latéralité par les pixels, affirmations fausses retirées de l'écran |
| 2026-09-13 | Tables BCS-DBT téléchargeables, statut par vue, examens normaux téléchargés, appariement par jointure, corpus d'examens à deux classes, `examclf` mesuré (négatif) |
| 2026-09-14 | Étape 2 retirée ; warm start, score relatif, mesures sans modèle (négatives) ; split test téléchargé |
| 2026-09-15 | Point de fonctionnement publié ; code mort retiré ; paquet réparé ; écarts doc ↔ code corrigés |
| 2026-09-19 | Chemin « nouvelle IRM » : `preprocess_dce_mri_exams`, `dce_subtraction` en définition unique, défaut `crop` aligné sur le corpus, code hérité IRM retiré (−105 lignes, 3 dépendances lourdes en moins), 20 tests (§4.16) |
| 2026-09-20 | Parcours de démo mesuré de bout en bout : installation dédiée (68 paquets → 17, 355 Mo → 153 Mo sur Windows), préflight qui analyse un vrai cas, port occupé détecté, formats de l'interface alignés sur le backend servi, 12 tests dont le premier clic de démo (§4.17) |
| 2026-09-20 | Médaillon `bronze → silver → gold` : étape `purge` dans les deux flows, primitive qui refuse plutôt que de parier, manifeste IRM cumulatif, catalogue adapté ; 341 tests |
| 2026-09-16 | **P0 portfolio** : README réorienté data engineering, décisions d'architecture, licence MIT, citations TCIA, GIF de démo, documentation unique en français. **P1** : catalogue DuckDB, migration dbt, flow Prefect DBT, délai sur les requêtes TCIA, trois défauts corrigés (§4.14) |

### Feuille de route « portfolio data engineering »

| Priorité | Tâche | État |
|---|---|---|
| P0 | Présentation : README, schéma, décisions, licence, GIF | **Fait** |
| P1 | Catalogue de métadonnées DuckDB/Parquet | **Fait** (§4.12) |
| P1 | Modèles et tests dbt sur le catalogue | **Fait** (§4.13) |
| P1 | Flow Prefect pour la chaîne DBT (tables → téléchargement → prétraitement → catalogue) | **Fait** (§4.14) |
| P1 | Délai maximal sur les requêtes TCIA, échecs de téléchargement détectés | **Fait** (§4.14) |
| P1 | Tests d'orchestration exécutés en CI | **Fait** (job `orchestration`) |
| P1 | Stockage objet S3 / MinIO : publication des tables, manifestes et Parquet | **Fait** (§4.15) |
| P2 | Structure `src/`, découpage de `TransformData.py`, build Docker en CI, registre de modèles | À faire |
| — | Détecteur supervisé par boîtes (piste 8) | Reporté : relève du ML, pas du portfolio data engineering |

### Points ouverts

| Point | Détail |
|---|---|
| Corpus après le téléchargement du 2026-09-17 | Les 15 séries des 9 patients manquants (dont 3 cancers) sont téléchargées ; reconstruction des corpus en cours au moment de ce commit, chiffres à publier (§4.15) |
| Bug NaN fp16 | Divergence repoussée à l'époque 15, non résolue ; checkpoint servi antérieur (P3) |
| Manifeste DCE-MRI | `dce_mri_p2/` n'en a pas (antérieur à `lineage.py`) |
| `output_dir` du manifeste DBT | Indique `dbt_join` alors que le dossier a été renommé en `dbt` |
| Latéralité à l'inférence DBT | Tout futur chemin d'inférence doit appliquer `image_laterality` (§4.4) |
| IC du top-1 à 43 % | Aucun code ni artefact versionné ne produit l'intervalle cité |
| Erreur `cudaErrorIllegalAddress` | Observée une fois sur `/demo/1`, non reproduite |
| Registre de traitement RGPD | Une page à écrire : base légale, nature des données, finalité, conservation, sécurité |
| Coupe choisie sur une IRM neuve | 43 % de top-1 (§4.3) : l'examen est préparé et servi correctement, la coupe reste le maillon faible |
| Purge du bronze | 870 séries DBT (63,8 Go) supprimables ; 0 côté IRM tant que `dce_mri_p2/` n'a pas de manifeste. L'écart « 186 volumes / 840 séries » est expliqué : 834 dossiers = 829 séries DCE de 189 patients + 5 autres, et le corpus est un volume par patient (186 = 189 − 3 patients sans phase pré ou post-2) |
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
| — | `models/dce_mri_p2_negfix/` nomme une expérience | Ouvert (le renommer casserait la démo) ; son `eval_report.json` cite encore `data/silver/dce_mri_p2`, mesure historique laissée telle quelle |

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
pip install -e ".[all]"   # la CI, elle, n'installe que .[dev,data,catalog,storage]
ruff check .
pytest                 # 341 tests, sans GPU ni jeu de données
                       # (260 collectés en retirant les 4 fichiers qui exigent
                       #  Prefect, dbt-duckdb ou boto3 ; mesuré le 2026-09-20)
```

La [CI](.github/workflows/ci.yml) a deux jobs à chaque push et pull request : `check`
(ruff + pytest, installation volontairement étroite : PyTorch CPU, numpy, pandas, pydicom,
flask, duckdb, dbt-duckdb, boto3, moto) et `orchestration` (Prefect + pandas, tests des deux flows).
Ajouter un test qui importe un nouveau module impose de l'ajouter à la CI. Les tests qui
demandent le client TCIA (`tests/test_extract_download.py`) ne tournent qu'en local :
`tcia_utils` tire `idc-index`, `ipython` et `plotly`. La suite couvre les définitions de métriques, le contrat de stockage, la logique
d'orchestration, le catalogue dbt, le rendu des pages, et vérifie que git **suit** bien
les artefacts de la démo.

Les pipelines journalisent via `logging` (`logging_setup.py`), avec horodatage et copie
sous `logs/` ; `BREASTCANCER_LOG_LEVEL=DEBUG` augmente le volume.

**Stockage objet** : `tests/test_objectstore.py` simule S3 en mémoire avec moto ; aucun
service externe n'est nécessaire, ni en local ni en CI.

**Scripts** : `scripts/make_demo_cases.py` (cas de démo), `scripts/make_demo_gif.py` (GIF du
README), `scripts/make_dbt_lineage_png.py` (graphe de lignage) — les deux derniers
demandent Playwright, hors dépendances du projet.

---

## Licence et données

Le **code** est sous [licence MIT](LICENSE).

Les **données d'imagerie** n'en relèvent pas. Les deux collections TCIA sont distribuées
sous [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), avec citation
obligatoire. Cela vaut pour les trois cas de démo versionnés et pour le GIF, dérivés de
Duke-Breast-Cancer-MRI : réutilisables avec attribution, à des fins non commerciales
uniquement.

- Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E., Ghate, S. V.,
  Walsh, R., & Mazurowski, M. A. (2021). *Dynamic contrast-enhanced magnetic resonance
  images of breast cancer patients with tumor locations* [Data set]. The Cancer Imaging
  Archive. <https://doi.org/10.7937/TCIA.e3sv-re93>
- Buda, M., Saha, A., Walsh, R., Ghate, S., Li, N., Swiecicki, A., Lo, J. Y., Yang, J., &
  Mazurowski, M. (2020). *Breast Cancer Screening – Digital Breast Tomosynthesis
  (BCS-DBT)* (Version 5) [Data set]. The Cancer Imaging Archive.
  <https://doi.org/10.7937/E4WT-CD02>
