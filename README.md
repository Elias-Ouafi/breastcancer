# Dépistage du cancer du sein — pipeline de données d'imagerie médicale

[![CI](https://github.com/Elias-Ouafi/breastcancer/actions/workflows/ci.yml/badge.svg)](https://github.com/Elias-Ouafi/breastcancer/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Tests : 341](https://img.shields.io/badge/tests-341-brightgreen)
[![Licence : MIT](https://img.shields.io/badge/licence-MIT-lightgrey)](LICENSE)

> **Research Use Only — Not for diagnostic use.** Outil de recherche, pas un dispositif
> médical, non validé cliniquement.

## Le projet en quelques lignes

**La question** : à partir d'un examen de dépistage mammaire — une tomosynthèse (DBT,
mammographie 3D) ou une IRM —, dire **s'il y a un cancer**.

**Ce que le projet construit pour y répondre** : une chaîne de données complète, de
l'archive publique jusqu'à une application web.

1. **Collecter** 138 Go d'examens DICOM publics (The Cancer Imaging Archive) et leurs
   tables d'annotations, en téléchargements reprenables.
2. **Transformer** ces examens en jeux de données exploitables : volumes normalisés,
   masques de lésion, étiquettes fiables — rattachées par jointure, jamais devinées.
3. **Contrôler et tracer** chaque fichier produit : validation de schéma au moment de
   l'écriture, manifeste de lignage (quelle source, quels paramètres, quel commit).
4. **Cataloguer** l'ensemble dans une base DuckDB construite avec dbt, interrogeable en
   SQL et testée à chaque build, puis **publier** tables, manifestes et Parquet vers un
   stockage objet S3 (MinIO en local).
5. **Entraîner et évaluer** des modèles avec une méthodologie stricte : découpage par
   patient, intervalles de confiance, seuil fixé hors échantillon.
6. **Servir** le résultat dans une application web qui se lance depuis un simple clone.

**L'objectif personnel** : contribuer, à mon niveau, au secteur médical, et en faire un
projet de portfolio de **data engineering**. Chaque chiffre publié ici peut être relié à
un fichier, un commit et un découpage par patient — y compris les chiffres qui disent
qu'un modèle ne marche pas.

![Démo : ouverture d'un cas IRM, zone repérée, balayage des coupes, vue MIP](docs/img/demo.gif)

*La démo : un cas en un clic, la zone de rehaussement repérée, le balayage des coupes,
la projection d'intensité maximale.*

## Où en est le projet

| Brique | État |
|---|---|
| **Pipeline de données** (collecte, transformation, validation, lignage) | Opérationnel : 5 060 patients catalogués, deux corpus construits (260 séries annotées, 870 examens à deux classes), 0 avertissement de validation |
| **Orchestration** (Prefect) | Opérationnelle : deux flows ; la chaîne DBT calcule hors ligne ce qui manque et n'exécute que ça ; téléchargements protégés par un délai maximal et repris en cas d'échec |
| **Stockage objet** (S3 / MinIO) | Opérationnel : synchronisation idempotente des tables, manifestes et tables en Parquet, lisibles directement depuis le bucket ; vérifié contre un point d'accès S3 local (MinIO fourni dans docker-compose, non exécuté ici faute de Docker) |
| **Catalogue de métadonnées** (DuckDB + dbt) | Opérationnel : 22 032 séries, 36 tests de qualité qui passent, reconstruit en ~10 s |
| **Localisation de lésion** (IRM, modèle de la démo) | Fonctionne **quand on lui montre la bonne coupe** : lésion trouvée dans 88 % des cas [IC 82–93 %] sur 28 patients de test |
| **Nouvelle IRM** (examen jamais annoté) | Opérationnel : `preprocess_dce_mri_exams` prépare un examen sans annotation. Vérifié sur **DICOM brut réel** — le volume produit est **identique bit à bit** à celui du corpus d'entraînement, puis servi par l'app en 3,9 s |
| **Détection du cancer au niveau de l'examen** (DBT) | **Ne fonctionne pas, et c'est publié** : ROC-AUC 0,457 [0,369–0,544] sur 272 patients, soit le hasard |
| **Démo** | Se lance depuis un clone, sans téléchargement de données : 4 dépendances (153 Mo sur Windows) au lieu de 68 paquets, et le lanceur **analyse un cas réel avant d'ouvrir le port** |

**Pourquoi la détection échoue.** Mesure après mesure (warm start, score relatif,
statistiques sans modèle jusqu'en résolution native), le diagnostic converge : ce qui
distingue un cancer est la **forme** de la lésion, pas sa luminosité, et une étiquette
« cancer / pas cancer » par examen ne suffit pas à l'apprendre sur 56 patients cancer.
Au seuil visé, la valeur prédictive positive (20,4 %) égale la prévalence (20,6 %) :
la réponse du modèle n'apporte aucune information. La piste suivante est un détecteur
supervisé par les boîtes de lésion, en résolution native.

**Priorité actuelle : le data engineering** — structure du code (`src/`), découpage de
`TransformData.py` et build de l'image Docker en CI.

## Architecture

```mermaid
flowchart TB
    subgraph SRC["Source — The Cancer Imaging Archive"]
        DBT[("Breast-Cancer-Screening-DBT<br/>DICOM + tables boîtes, labels, inventaire")]
        MRI[("Duke-Breast-Cancer-MRI<br/>IRM DCE en DICOM")]
    end

    subgraph EXTRACT["Collecte — ExtractData.py"]
        TABLES["download_dbt_tables<br/>9 tables d'annotations"]
        SERIES["download_*_series<br/>reprenable · plafonné · tirage avec graine"]
    end

    RAW[("bronze/tcia<br/>tel que publié · supprimé une fois en silver")]

    subgraph TRANSFORM["Transformation — TransformData.py"]
        JOIN["Jointure boîte ↔ série<br/>sur PatientID, StudyUID, View"]
        LABEL["Étiquette d'examen<br/>pire vue de la table labels"]
        GEOM["Rééchantillonnage + normalisation<br/>même géométrie pour toutes les classes"]
    end

    VALID{{"validation.py<br/>contrôle de schéma à l'écriture"}}
    LINEAGE[/"lineage.py<br/>manifest.json : commit, paramètres, stats"/]
    PRE[("silver<br/>un .npz par série · seule copie")]
    CUR[("gold<br/>banques de coupes, cas de démo")]

    subgraph ML["Entraînement et évaluation — imaging/"]
        TRAIN["U-Net · classifieur de coupe · classifieur d'examen<br/>découpage par patient, validation croisée"]
        EVAL["evaluate · oppoint<br/>IC bootstrap par patient, seuil hors pli"]
    end

    ART[("models/ · reports/<br/>checkpoints + métriques")]
    APP["App Flask + API JSON<br/>Docker, lecture seule, local uniquement"]
    CAT[("catalog.duckdb · dbt<br/>raw → stg → mart · 36 tests · Parquet")]
    S3[("Bucket S3 / MinIO<br/>bronze/ · silver/ · gold/")]

    DBT --> TABLES
    DBT --> SERIES
    MRI --> SERIES
    TABLES --> RAW
    SERIES --> RAW
    RAW --> JOIN --> LABEL --> GEOM
    GEOM --> VALID --> PRE
    PRE -.->|"purge du bronze"| RAW
    PRE -.-> LINEAGE
    PRE --> CUR --> TRAIN --> EVAL --> ART
    ART --> APP
    RAW -.-> CAT
    LINEAGE -.-> CAT
    ART -.-> CAT
    CAT --> S3
    RAW -.->|tables| S3
```

Les flèches en pointillés alimentent le catalogue : tables d'annotations, disque,
manifestes et scores des modèles, joints et testés. Le catalogue et les tables sont
ensuite publiés vers un stockage objet S3.

## Démarrage rapide

La démo ne demande aucun téléchargement de données : le modèle et trois cas sont
versionnés. Elle n'a besoin que de quatre paquets — `torch`, `Flask`, `numpy`,
`Pillow` — et pas du reste du pipeline :

```bash
pip install -e .              # le socle de pyproject.toml : ces quatre paquets seulement
python run_demo.py --open     # préflight, puis http://127.0.0.1:5000
```

`run_demo.py` ne se contente pas de vérifier que les fichiers sont là : il **charge le
checkpoint et analyse un cas** avant d'ouvrir le port, et refuse de démarrer si le port
est occupé. `--check` fait le même contrôle sans lancer le serveur, `--fast-check` s'en
tient à l'inventaire des fichiers.

Ou avec Docker seul — l'image n'a **jamais été construite** ici ni en CI, mais son
contenu est vérifié sans daemon : les `COPY` suffisent à faire tourner la démo, et ses
quatre paquets aussi (§4.17 de [DOCUMENTATION.md](DOCUMENTATION.md)) :

```bash
docker compose up --build
```

Pour tout le reste — collecte, transformation, catalogue —, l'installation complète. Un seul
fichier déclare les dépendances, `pyproject.toml` : le socle est la démo, chaque autre usage
est un extra (`data`, `collect`, `catalog`, `orchestration`, `storage`, `dev`) et `all` les
réunit :

```bash
pip install -e ".[all]"
```

Chaîne de données DBT et catalogue (nécessitent les tables et les données téléchargées) :

```bash
pip install -e ".[data,collect,catalog,orchestration,storage]"
python -m pipelines.dbt --dry-run     # ce qui manque, calculé hors ligne
python -m pipelines.dbt               # télécharge, prétraite, catalogue, publie ce qui manque
python -m catalog query --file catalog/queries/01_data_funnel.sql
```

## Choix techniques marquants

- **Joindre plutôt que deviner.** Le tag DICOM de latéralité est faux sur 262 séries sur
  262 ; l'inventaire publié par la collection transforme l'appariement en jointure :
  260 séries, 0 masque vide.
- **Une seule source et une seule géométrie pour les deux classes**, sinon un modèle
  apprend le scanner ou la taille du tableau au lieu de la lésion.
- **Un seuil qui ne voit jamais les patients qu'il juge** : fixé à la sensibilité du
  programme national de dépistage (82,8 %), calé sur les autres plis.
- **Des tests qui savent échouer** : chaque contrôle de qualité clé du catalogue est mis
  en défaut sur des données où l'erreur est injectée.
- **Un examen neuf préparé exactement comme le corpus d'entraînement** : les deux chemins
  de prétraitement partagent une seule définition de la soustraction, et sur du DICOM
  brut réel le volume obtenu est identique **bit à bit** à celui du corpus.
- **Une migration vérifiée** : le passage du catalogue à dbt a été validé par comparaison
  ligne à ligne avec l'ancienne version.
- **Une orchestration qui décide depuis un plan** : chaque étape compare ce qui devrait
  exister (d'après les tables de la collection) à ce qui existe, au lieu de sauter une
  étape dès que son dossier existe — ce qui laisserait passer un corpus périmé.

## Stack

Python 3.12 · SQL · dbt · DuckDB · Parquet · S3 / MinIO (boto3) · pydicom · NumPy · pandas · PyTorch ·
Prefect · Flask · Docker · GitHub Actions

## Organisation du dépôt

```
pyproject.toml         dépendances : socle = les 4 paquets de la démo, plus des extras ; l'image Docker lit le socle
ExtractData.py      collecte TCIA : tables d'annotations, séries annotées et normales
TransformData.py    DICOM → volumes normalisés + masques, jointure boîte/série, étiquettes
validation.py       contrôles de schéma au point unique d'écriture
lineage.py          manifest.json par dossier prétraité
config.py           tous les chemins, définis une fois
pipelines/          flows Prefect : chaîne DBT (planifiée hors ligne) et chaîne IRM
http_timeouts.py    délai maximal sur les requêtes du client TCIA
catalog/            catalogue de métadonnées : chargement, projet dbt, requêtes d'exemple
objectstore/        publication idempotente vers un stockage objet S3 / MinIO
imaging/            jeux de données, banques, U-Net, classifieurs, métriques, évaluation
inference.py        chargement des modèles et prédiction pour l'app
app/                application Flask (HTML + API JSON)
tests/              tests sur données synthétiques, sans GPU ni jeu de données
models/, reports/   checkpoints et rapports versionnés
scripts/            régénération des cas de démo, du GIF et du graphe de lignage
DOCUMENTATION.md    toute la documentation détaillée
```

## Documentation

**[DOCUMENTATION.md](DOCUMENTATION.md)** rassemble tout le reste : contexte et
historique, cible chiffrée, données, commandes de chaque étape du pipeline, catalogue et
tests dbt, démo et application, décisions d'architecture, journal daté de toutes les
mesures (échecs compris), pistes, état d'avancement et points ouverts.

## Licence et données

Code sous [licence MIT](LICENSE). Les données d'imagerie restent sous licence
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) avec citation
obligatoire (voir [DOCUMENTATION.md](DOCUMENTATION.md#licence-et-données)).
