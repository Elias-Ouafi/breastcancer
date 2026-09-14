# plan.md — décisions de conception et journal des mesures

> **Outil visé** : deux étapes — (1) un examen en entrée, dire s'il y a un cancer ;
> (2) les retours de la biopsie en entrée, dire si c'est malin ou bénin.
> **Modalités** : étape 1 sur **DBT / mammographie** (`Breast-Cancer-Screening-DBT`),
> décision du 2026-09-12, voir la section suivante. L'IRM multiphase (DCE-MRI, DICOM)
> reste la modalité de la brique de **localisation**, qui s'active après l'étape 1.
> **Cible** : démo / portfolio. **Pas d'usage clinique, pas de certification.**
> **Mention obligatoire, partout** : *Research Use Only — Not for diagnostic use*.

Ce document garde ce qui ne se déduit pas du code : la charte graphique appliquée à
l'app (Partie 3) et le journal daté de ce qui a été mesuré, y compris les échecs
(§4.1 à §4.3). Le reste — comment lancer la démo, où vivent les données, comment
tourne le pipeline — est dans [README.md](README.md), au plus près du code.

## Cible chiffrée et voie retenue (2026-09-12)

Jusqu'ici l'étape 1 n'avait pas de cible chiffrée : « dire s'il y a un cancer » ne dit
pas à quel taux. Elle en a une maintenant, prise sur le programme national de
dépistage organisé, et elle change la modalité de l'étape 1.

### La cible

Point de fonctionnement visé, **au niveau patient**, un examen entrant, une décision
sortante :

| Mesure | Cible | Source |
|---|---:|---|
| Sensibilité | **82,8 %** | Santé publique France, dépistage organisé 50-74 ans |
| Spécificité | **91,4 %** | idem |
| Sensibilité selon la tranche d'âge | 76 – 88 % | idem |
| Sensibilité à 1 an (cancers d'intervalle inclus) | 94,2 % | idem |

Ce couple (Se, Sp) correspond, sous hypothèse binormale, à une **ROC-AUC patient de
l'ordre de 0,95** — c'est l'objectif à porter dans `eval_report.json`, avec son IC.

Repères de contexte, pas des cibles : faux négatifs > 15 % (MSD), 85 à 90 % des
anomalies détectées ne sont pas des cancers (MSD), VPP 11,3 % en 2020 contre 7,8 % en
2008 (SpF).

### Ce qu'on ne vise pas : la VPP

La sensibilité et la spécificité sont des propriétés du modèle à un seuil donné. **La
VPP n'en est pas une** : elle dépend de la prévalence. Au taux de détection du
programme (~6,7 cancers pour 1 000 dépistées), Se 82,8 % et Sp 91,4 % donnent

    0,0067 × 0,828 / (0,0067 × 0,828 + 0,9933 × 0,086) = 6,1 %

et non 11,3 %. Obtenir 11,3 % à cette prévalence et cette sensibilité demanderait une
spécificité de 95,6 % : les deux chiffres publiés ne décrivent pas le même point de
fonctionnement (années, définition du « test positif », dénominateurs). Aucun des deux
n'est faux ; ils ne s'additionnent simplement pas.

Conséquence opérationnelle : sur un jeu de test équilibré, le même modèle afficherait
une VPP de ~90 %, ce qui ne prouverait rien. **On fixe (Se, Sp), et on publie la
prévalence du jeu de test à côté de la VPP qui en découle.** Une VPP citée sans sa
prévalence est un chiffre sans unité.

### La voie retenue : bascule de l'étape 1 sur DBT / mammographie

Trois raisons, indépendantes l'une de l'autre, rendaient la cible inatteignable sur le
corpus DCE-MRI :

1. **Mauvaise modalité pour la comparaison.** Les chiffres visés sont ceux de la
   mammographie de dépistage. Un modèle IRM comparé à eux, même au bon niveau, compare
   deux choses différentes.
2. **Mauvaise population.** Duke-Breast-Cancer-MRI est une cohorte diagnostique :
   prévalence 100 %, et `preprocess_dce_mri_with_boxes` écarte en plus tout patient
   sans boîte. Une spécificité ne se mesure pas sur un corpus sans négatifs.
3. **Le piège de la solution évidente.** Positifs chez Duke + négatifs ailleurs donne
   un modèle qui apprend la source — scanner, protocole, centre — et une AUC de 0,99
   qui ne vaut rien. Les deux classes doivent venir de la **même collection**.

`Breast-Cancer-Screening-DBT` lève les trois d'un coup : c'est une collection de
dépistage, **majoritairement normale** — le docstring de
`ExtractData.download_annotated_dbt_series` le dit lui-même, juste avant d'exclure
délibérément les patients non annotés. Le code de téléchargement, de prétraitement et
d'appariement boîte ↔ série existe déjà et tourne.

**Ce que la bascule ne fait pas** : elle n'annule pas le travail DCE-MRI. Le U-Net
garde sa fonction — localiser une lésion une fois l'examen déclaré suspect — et ses
mesures restent valides pour cette fonction-là. Il quitte le chemin critique de
l'étape 1, il ne quitte pas le projet.

### Les chiffres actuels — ligne de base au 2026-09-12

Mesuré dans ce dépôt, pas repris de la documentation.

**Étape 1, ce que produit l'app aujourd'hui :**

| Mesure | Cible | Actuel |
|---|---:|---|
| Sensibilité patient | 82,8 % | **100 %**, trivialement : la sortie est constante |
| Spécificité patient | 91,4 % | **0 %**, et non mesurable — zéro patient sain au corpus |
| ROC-AUC patient | ≈ 0,95 | **non mesurable**, proxy le plus proche 0,50 |
| Spécificité par coupe | — | **0,03 %** (99,97 % des coupes saines alarment) |
| VPP | 11,3 % @ p ≈ 0,7 % | indéfinie (p = 100 %) |

`inference._localize_lesion` décide par `best_conf >= 0.5`, et `best_conf` vaut
**1,0000** partout : 28/28 patients du split test, et 160/160 coupes de
`Breast_MRI_001` dont les 136 sans lésion (moyenne 1,0000, minimum 1,0000). La sortie
binaire est une constante. L'app l'affichait en « Confiance 100 % », jauge remplie :
elle ne l'affiche plus depuis le 2026-09-12 (voir Livrés), et la valeur brute
n'apparaît plus que dans le détail technique, nommée pour ce qu'elle est. La branche
« aucune zone suspecte » des gabarits reste inatteignable avec ce checkpoint.

**Ce qui reste valide, et qui relève de la localisation** (28 patients test, IC 95 %
bootstrap par patient, `models/dce_mri_p2_negfix/eval_report.json`) : Dice 0,533
[0,473 – 0,593] ; sensibilité lésion IoU ≥ 0,1 88,0 % [81,9 – 93,4] ; 222,2 faux
positifs par volume [204,7 – 237,4] ; 0,825 s par volume. Classifieur de coupe :
top-1 42,9 % (12/28, reproduit à travers `predict_dce_mri`), AUC **intra-volume**
0,803 — un pouvoir de tri entre coupes d'un même patient, à ne pas lire comme une AUC
de détection.

**Pourquoi « durcir le seuil » ne marcherait pas.** Si la décision patient était « au
moins une coupe s'allume » sur ~160 coupes, il faudrait un taux de faux positifs par
coupe de 1 − 0,914^(1/160) ≈ **0,056 %**, soit une spécificité par coupe de 99,94 %
contre 0,03 % mesurée. L'agrégation multiplie l'exigence par ~160 : il faut une tête
de décision **au niveau volume**, pas un seuil mieux choisi.

**Ce que le corpus DBT contient déjà** — ignoré jusqu'au 2026-09-12 :

| | Lignes | Patients | Classes |
|---|---:|---:|---|
| `BCS-DBT-boxes-train.csv` | 224 | 101 | benign 137 / cancer 87 |
| `BCS-DBT-boxes-validation.csv` | 75 | 40 | benign 38 / cancer 37 |
| **Pool** | **299** | **141** | **82 patients bénins / 59 cancers** |

La colonne `Class` n'était **lue nulle part** : `preprocess_dbt_with_boxes` peignait
toute boîte dans le masque sans la regarder, donc « positif » voulait dire « une
lésion », pas « un cancer ». Sur les **72 patients prétraités** (147 `.npz`),
**48 sont bénins et 24 cancéreux** — deux tiers des positifs n'étaient pas des
cancers. C'était la mauvaise étiquette pour la cible, et c'est la bonne pour
l'étape 2.

Corrigé le 2026-09-12 (voir Livré) : chaque `.npz` porte maintenant son
`lesion_class` et un `label` 0/1, et le corpus a été reconstruit — deux fois, la
seconde après la correction d'appariement du §4.4. Mesuré sur les fichiers, pas
déclaré :

| | Étiquetage seul | Latéralité par les pixels (§4.4) | Jointure `file-paths` (§4.6) |
|---|---:|---:|---:|
| Séries | 147 | 253 | **260** |
| Séries bénignes / cancers | 100 / 47 | 151 / 102 | **153 / 107** |
| Patients | 72 | 130 | **132** |
| Patients bénins / cancéreux | 48 / 24 | 75 / 55 | **76 / 56** |
| Patients mélangeant les deux classes | 0 | 0 | **0** |
| Avertissements de validation | 5 | 0 | **0** |
| Taille | 0,55 Go | 1,06 Go | **1,00 Go** |

Le `manifest.json` couvre les 260 fichiers, un par série, porte la révision qui l'a
produit (`git_revision`), la vue et l'étude de chaque cas, et compte 14 séries lues
comme miroir. Durée d'une passe complète : 27 à 56 min pour 22,4 Go de DICOM à décoder
— la première passe, qui écartait 115 séries sur l'en-tête, était la plus rapide ; la
dernière, qui en décode 260, la plus lente.

**Ce que la reconstruction a fait apparaître** (2026-09-12, mesuré) :

- **5 séries sur 147 ont un volume constant après recadrage** (`validation` les
  signale : « nothing to learn from this series »), chez 3 patients — DBT-P00538
  (cancer, **ses deux séries**), DBT-P03677 (bénin, 2 séries), DBT-P02919 (bénin, 1).
  Le masque y couvre pourtant 38 000 à 800 000 voxels : la boîte désigne une zone
  uniforme. **Élucidé le même jour, et c'était plus large que ces 5 séries : voir
  §4.4.** Le masque était posé sur le mauvais sein.
- **Une trame complète DBT pèse ~745 Mo en float16** (2457×1890 à 1996, 47 à 100
  coupes). Le corpus recadré tient en 0,55 Go, médiane 37×252×253. Conséquence
  directe pour la tête de décision : un volume recadré sur la lésion **présuppose la
  réponse**, et `crop=False` demanderait ~100 Go. Il faudra sous-échantillonner le
  plan (p. ex. 512×512), pas seulement changer le drapeau.
- **115 des 262 séries sur disque n'ont aucune boîte** (62 patients), et cela ne veut
  **pas** dire « examen normal » : le statut par étude (normal / actionable / benign /
  cancer) vit dans `BCS-DBT-labels-*.csv`, qui **n'est pas téléchargé**. *(Téléchargé
  depuis le 2026-09-13, et la jointure du §4.6 ramène ces 115 à **2** séries sans
  boîte : les 113 autres en avaient une, que l'inférence ne trouvait pas.)* C'est le
  premier obstacle du P1 « corpus à deux classes », avant tout téléchargement.

### Ce que la cible coûte en données

Pour **mesurer** ces taux, dans le jeu de test seul :

| Objectif de mesure | ± 5 points | ± 3 points |
|---|---:|---:|
| Sensibilité 82,8 % → patients avec cancer | ~220 | ~610 |
| Spécificité 91,4 % → patients sans cancer | ~121 | ~335 |

Le split test actuel compte 28 patients, tous positifs : l'IC sur une sensibilité y
serait de **± 14 points**, incapable de distinguer 82,8 % de 76 % ou de 94 %. C'est le
même problème que celui déjà visible sur le classifieur de coupe (42,9 %, IC 25 – 61).
La contrainte principale du projet n'est pas le modèle, c'est le volume d'examens
annotés.

## Où en est le projet (2026-08-18)

> **Relu contre la cible produit**, énoncée ici pour la première fois : (1) une IRM en
> entrée, dire s'il y a un cancer ; (2) les retours de la biopsie en entrée, dire si
> c'est malin ou bénin. Les mesures ci-dessous étaient justes ; c'est leur lecture qui
> change. Aucune ne répond à l'étape 1. L'étape 2, elle, ne sortait pas du disque —
> c'est corrigé depuis le 2026-08-18.

**Ce qui est mesuré.** U-Net 2D entraîné sur 186 patients Duke-Breast-Cancer-MRI, servi
sur le volume de soustraction 2ᵉ phase post-injection. Sur 28 patients de test jamais
vus, avec intervalles de confiance bootstrap calculés **par patient** — les coupes d'un
même patient sont corrélées, rééchantillonner les coupes donnerait un intervalle
faussement étroit :

| | Valeur | IC 95 % |
|---|---|---|
| Dice | 0,533 | 0,473 – 0,593 |
| Sensibilité lésion (IoU ≥ 0,1) | 88,0 % | 81,9 – 93,4 |
| Faux positifs par volume | 222 | 205 – 237 |
| Coupes saines déclenchant une alarme | 99,97 % | 99,92 – 100 |
| Temps par volume | 0,82 s | — |

Détail par patient dans `models/dce_mri_p2_negfix/eval_report.json` et
`eval_per_patient.csv`, régénérables par `python -m pipelines.dce_mri --from evaluate`.

**Étape 1 — la détection n'est pas commencée.** Le modèle segmente bien une lésion *une
fois la bonne coupe montrée*, mais « y a-t-il un cancer ? » n'est posé nulle part dans
le code, et trois faits l'enchaînent :

1. *Aucun pouvoir de tri.* 99,97 % des coupes sans lésion déclenchent une alarme, et
   l'aire prédite est la même sur coupe avec lésion (1228,6 px) et sans (1228,3 px) —
   §4.2. Ce n'est pas un mauvais classement, c'est l'absence de signal discriminant.
2. *Aucun négatif dans le corpus.* Le préprocessing est piloté par les annotations :
   `download_annotated_dbt_series` ne récupère que les patients listés au CSV de boîtes
   (`ExtractData.py`), et le côté Duke ne retient que les patients avec boîte. **100 %
   des patients du corpus ont un cancer** : ni spécificité, ni ROC patient, ni même la
   possibilité d'entraîner la tâche.
3. *La démo contourne les deux.* Les trois cas portent une coupe figée choisie à
   l'avance par un humain (`forced_slice`). L'app le dit — c'est honnête, ce n'est pas
   une solution. Le classifieur de coupe dédié fait 0 % → 42,9 % en top-1 (§4.3) : le
   problème est tractable, il n'est pas résolu.

Ce qu'il faudrait : des examens **sans** cancer (les normaux de `Breast-Cancer-Screening-DBT`
sont la source la moins chère — la collection est majoritairement normale, c'est écrit
dans le docstring de la fonction qui les exclut), puis une tête de classification au
niveau **volume/patient**, jugée sur sensibilité/spécificité/ROC-AUC. Pas sur du Dice :
le Dice répond à « où », une fois que « si » est répondu.

**Étape 2 — branchée le 2026-08-18.** Wisconsin Diagnostic *est* l'étape 2 : 30
features morphologiques mesurées sur une cytoponction, 569 cas, label M/B, ROC-AUC
99,89 % (`Final_Report.md`). Elle est désormais servie à `/biopsie` : dix mesures ×
trois statistiques, deux exemples cliquables, verdict malin/bénin et probabilité.

Ce qui bloquait n'était pas le modèle mais le fait de le servir. `predict_tabular`
démarrait une session Spark et relisait un `PipelineModel` à chaque appel — des
secondes de JVM pour trente flottants, dans une image qui n'embarque volontairement
ni Spark ni JVM. Or ce pipeline est entièrement affine (impute → standardise → PCA →
standardise → logistique) : `tabular_export.py` en extrait les constantes dans
13 Ko de JSON qui reproduisent le modèle Spark **à 1,0 × 10⁻¹⁵ près, 0 désaccord de
label sur les 569 lignes**. Spark reste l'outil d'entraînement, il sort du chemin de
requête. L'artefact est versionné, comme les checkpoints, pour que `/biopsie` réponde
depuis un clone.

Deux choses corrigées au passage. `models/tabular/` ne contenait pas de
`metadata.json` : le modèle persisté était inutilisable, et personne ne s'en était
aperçu puisque rien ne l'ouvrait. Et `_get_spark` ne pointait pas vers le `winutils`
pourtant fourni dans le venv, donc le pipeline tabulaire ne démarrait que pour qui
avait posé `HADOOP_HOME` à la main dans son shell.

La probabilité affichée est bornée à « > 99,9 % » plutôt qu'arrondie : le modèle rend
0,9999999999991 sur l'exemple malin, et « 100,0 % » se lirait comme une certitude.

**BreakHis est une branche morte.** `ExtractBreakHis.py` télécharge et extrait ;
`BREAKHIS_DIR` n'apparaît nulle part ailleurs que dans `config.py`. Zéro modèle, zéro
test, zéro métrique. C'est pourtant le support image de l'étape 2 — les lames issues de
la biopsie. À assumer comme pendant image du Wisconsin, ou à retirer.

**Le dépôt, lui, tient.** Données en couches sous `data/` (raw → preprocessed →
curated), chemins centralisés dans `config.py`, checkpoints dans `models/`, pipeline
exécutable et reprenable via `python -m pipelines.dce_mri`, **116 tests** (~7 s hors
parité Spark, sans GPU ni dataset) et ruff en CI. Docker, validation de schéma et manifeste de lineage sont
écrits, testés et commités le 2026-08-18.

## Ce qui reste ouvert

Ordonné par ce qui rapproche de la cible en deux étapes, pas par facilité. Réordonné
le 2026-09-12 : la cible chiffrée et la bascule DBT déplacent ce qui bloque.

| Priorité | Tâche | Critère de « fait » |
|---|---|---|
| P1 | Corpus DBT à deux classes, d'une seule source | **Fait le 2026-09-13**, voir Livré : le filtre « patients annotés » est levé, **150 patients normaux sont téléchargés** (660 séries, 46,8 Go mesurés — 312 Mo par patient, pas les ~200 Mo qu'un premier patient laissait croire), l'étiquette vient du statut par vue, et `preprocess_dbt_exams` écrit les deux classes dans **une seule géométrie**. Reste à publier les chiffres du corpus construit — la passe sur 926 séries tourne |
| P1 | Tête de décision au niveau examen | Mesurée le 2026-09-13 et négative, voir Livré : ROC-AUC patient 0,457 [0,369 – 0,544] sur 870 examens / 272 patients / 56 cancers, IC contenant 0,5, 0 cancer détecté au seuil 0,5. Diagnostic, chiffres et pistes au §4.7 |
| P1 | Choisir et publier le point de fonctionnement | Seuil fixé sur la **validation** pour Se = 82,8 % ; spécificité, VPP et **prévalence du jeu de test** rapportées sur le **test**, avec IC. Le panneau « Limites connues » cite Se/Sp/IC/prévalence au lieu du Dice |
| P2 | Reprendre l'étape 2 en version image **si le corpus grossit** | Mesurée le 2026-09-12 et négative : ROC-AUC patient 0,513 [0,411 – 0,615] sur 130 patients, l'IC contient le hasard (§4.5). Le levier identifié est le nombre de patients, pas le modèle. Le réseau n'est plus l'obstacle (2026-09-13) : ce qui reste à décider est le volume disque. Le plus court chemin est désormais chiffré — les boîtes du **split test** de la collection existent (`BCS-DBT-boxes-test`, 136 lignes, 60 patients, 30 cancers) et n'ont jamais été utilisées ici : pooler les trois splits porte le corpus annoté de 141 à **201 patients, dont 89 cancers**, soit tous les cancers annotés de la collection |
| P2 | Donner à l'étape 2 tabulaire une mesure qui lui appartienne | `train_tabular_model.py` fait un split (ou une VC) et persiste les métriques du modèle **servi** ; `AnalyzeData` ajuste imputation/scaler/PCA **après** le split ; `reports/model_results.csv` recommité |
| P2 | Trancher le sort de BreakHis | Un modèle bénin/malin entraîné et mesuré, ou le script et `BREAKHIS_DIR` supprimés. Rétrogradé de fait : `Class` fournit un pendant image moins cher |
| P3 | Trancher le sort du pipeline tabulaire Spark | Assumé et documenté comme démo Spark, ou retiré. Rétrogradé de P2 : depuis l'export, la JVM n'est plus qu'une dépendance d'entraînement, plus une condition pour servir |
| P3 | Bug NaN fp16 non résolu | La divergence (§4.2, repoussée époque 11 → 15) est localisée dans le forward pass et corrigée, ou documentée comme acceptée. Rétrogradé de P2 : le U-Net DCE-MRI quitte le chemin critique de l'étape 1. Le checkpoint servi reste un instantané pré-divergence (époque ≤ 14 sur 30) |
| P3 | Retirer le code mort | `app/run_unet.py`, `DbtUNetPredictor` et `predict_dbt` pointent un checkpoint qui n'existe plus (`models/dbt/unet_best.pt`, écrasé par un smoke test, §4.1). À réécrire pour la nouvelle tête DBT ou à supprimer, pas à laisser documenté comme disponible dans `app/README.md` |
| P3 | Réparer le paquet | `pyproject.toml` omet `logging_setup`, `validation` et `lineage` de `py-modules` : hors du répertoire du dépôt, `import inference` échoue. Masqué parce qu'on lance toujours depuis la racine |
| P3 | Exécuter le lineage sur le corpus | Aucun `manifest.json` n'existe sous `data/preprocessed_data/` : le code est écrit et testé, jamais passé sur les données réelles |
| P3 | Registre de traitement RGPD | Une page : base légale, nature des données, finalité, conservation, sécurité |
| P3 | Nom de produit + logo | Choisi et intégré au header de l'app |

**Livré** (branche `ameliore-le-mvp`, le 2026-09-13) :

| Tâche | Où | Ce qui la rend faite |
|---|---|---|
| P1 — apparier les boîtes depuis `file-paths` au lieu de les inférer | `TransformData.py` (`series_uid_from_classic_path`, `view_position_of`, `_read_file_paths`, `BOX_JOIN_COLUMNS`, `preprocess_dbt_with_boxes`), `tests/test_dbt_preprocessing.py` | L'appariement est une **jointure** sur `(PatientID, StudyUID, View)` ; les trois helpers d'inférence sont supprimés et les pixels ne décident plus que du retournement. Corpus reconstruit et **mesuré sur les fichiers** : 260 séries (contre 253), 132 patients, 153 bénignes / 107 cancers, 76 / 56 patients, 0 masque vide, 0 avertissement, 1,00 Go, 56,1 min. Diff contre l'ancien corpus : +7 séries, 0 perdue, **exactement 4 masques déplacés** — les 4 acquisitions répétées que le §4.4 annonçait mal appariées — et 14 miroirs conservés. Les 11 séries à vue répétée (`lmlo1`, `rcc1`, `lcc2`…) sont appariées pour la première fois. Détail au §4.6 |
| P1 — rendre les tables BCS-DBT téléchargeables depuis un clone | `ExtractData.download_dbt_tables`, `config.py` (`DBT_FILE_PATHS*`, `DBT_LABELS_*`, `DBT_BOXES_TEST`) | Les **9 tables** (boîtes, labels par vue, inventaire `file-paths`, pour les trois splits) se téléchargent en une fonction ; deux d'entre elles sont **identiques octet pour octet** aux copies posées à la main le 2026-09-13. Jusqu'ici un clone ne pouvait pas prétraiter DBT du tout, puisque la jointure exige l'inventaire. Un nom publié ne suit pas le nom local (`BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv`) : la table le dit |
| P1 — lire le statut par vue, seule source du mot « normal » | `TransformData.read_dbt_labels`, `TransformData.dbt_patient_status`, 4 tests | Un patient est lu **à sa pire vue** (un cancer ⇒ examen cancer), une ligne sans aucun drapeau n'est pas comptée normale, et une table amputée d'une colonne est refusée. Reproduit depuis les fichiers les chiffres jusque-là repris de la documentation : 4 581 normaux / 278 actionable / 112 bénins / **89 cancers**, 5 060 patients |
| P1 — télécharger des examens sans cancer | `ExtractData.download_normal_dbt_series`, `ExtractData.download_dbt_series_for` | Le filtre « patients annotés » est levé : la sélection vient du statut par vue, un patient n'est pris que si **toutes** ses vues sont normales, et l'échantillon est tiré avec une graine plutôt que par ID croissant (les ID suivent le site et la date). Cap exprimé en **volume ajouté par l'appel** et non en taille totale du dossier — l'ancien cap se déclenchait immédiatement sur un dossier partagé avec Duke. Coût mesuré : ~200 Mo par patient normal (4 vues) |
| P1 — tête de décision au niveau examen, sur l'étape 1 enfin mesurable | `imaging/exambank.py`, `imaging/examclf.py`, `models/examclf/cv_report.json` | Entraînée et **mesurée** sur les deux classes réunies pour la première fois : ROC-AUC patient 0,457 [0,369 – 0,544] sur 870 examens, 272 patients, 56 cancers. L'IC contient 0,5 et le modèle ne détecte aucun des 56 cancers au seuil 0,5 : le résultat est **négatif**, et c'est ce qui est publié. Protocole, diagnostic et pistes au §4.7. Fait au sens du critère — un modèle mesuré — pas au sens d'un modèle utilisable |

**Livré** (branche `ameliore-le-mvp`, le 2026-09-12) :

| Tâche | Où | Ce qui la rend faite |
|---|---|---|
| P2 — étape 2 en version image, depuis `Class` | `imaging/lesionclf.py`, `imaging/metrics.py` (`roc_auc`, `bootstrap_auc`, `operating_point`), `imaging/dataset.py` (`kfold_by_patient`), `models/lesionclf/cv_report*.json` | Entraîné et **mesuré** : ROC-AUC patient 0,591 [0,491 – 0,693] puis 0,513 [0,411 – 0,615] après correction d'un défaut d'échelle, sur 130 patients dont 55 cancers. Les deux IC contiennent 0,5 et aucune exactitude ne bat « toujours bénin » : le résultat est **négatif**, et c'est ce qui est publié. Protocole, diagnostic et pistes au §4.5. Fait au sens du critère — un modèle mesuré — pas au sens d'un modèle utilisable |
| P1 — apparier les boîtes par la latéralité des pixels, pas par le tag DICOM | `TransformData.py` (`image_laterality`, `dbt_view_position`, `dbt_series_view`, `_candidate_boxes`, `_select_boxes`, `preprocess_dbt_with_boxes`), `tests/test_dbt_preprocessing.py` | Le tag lit `L` sur les 262 séries ; les pixels donnent 134 R / 128 L. L'appariement trouvait 147 séries annotées sur 253 et posait 23 masques sur du fond. Corrigé selon la sémantique du lecteur officiel du jeu de données, avec les deux cas distingués par la mesure et non par choix (9 mal appariées, 14 études stockées en miroir). Vérifié sur 11 séries réelles couvrant chaque cas avant relance, 0 avertissement de validation contre 2. Détail et chiffres : §4.4 |
| P1 — lire la colonne `Class` des CSV de boîtes | `validation.py` (`LESION_CLASSES`, `lesion_class_label`), `TransformData.py` (`_read_boxes`, `save_preprocessed`, `preprocess_dbt_with_boxes`), `README.md` | Chaque `.npz` porte `lesion_class` (`benign`/`cancer`) et un `label` 0/1 ; le masque reste binaire, parce que la classe n'est pas peignable — une lésion bénigne peint les mêmes pixels qu'un cancer. `Class` est **exigée** (un CSV sans elle est refusé, pas traité comme une classe unique) et une valeur inconnue est refusée. `mask_classes` choisit ce qui est peint sans changer ce que dit l'étiquette ; par défaut les deux, les bénins étant deux tiers du corpus annoté. Étiquette d'examen : toute boîte cancer ⇒ examen cancer, mélange journalisé (jamais rencontré : 82 patients bénins purs, 59 cancers purs). Corpus reconstruit et **mesuré sur les fichiers** : 147 séries, 0 sans étiquette, 100 bénignes / 47 cancers, 72 patients (48 / 24), aucun patient mélangé, 0,55 Go, 27 à 53 min par passe. Manifeste vérifié : 147 cas, un par fichier, révision `cdde11e`, 5 avertissements. 16 tests sur DICOM synthétiques, dont les cas que la vraie donnée ne fournit pas (série mélangée, classe inconnue, colonne absente) |
| P0 — corriger les affirmations que le code contredit | `app/templates/result.html`, `app/templates/biopsy.html`, `app/templates/base.html`, `app/predictor.py`, `inference.py`, `app/README.md` | Trois affirmations retirées de l'écran. (1) La pastille « Confiance 100 % » et sa jauge remplie : `best_conf` est un maximum de probabilité **par pixel**, constant à 1,0000, pas un score d'examen — la valeur reste lisible dans le détail technique sous le nom « Probabilité max. par pixel (non calibrée) », et le panneau de limites dit que le verdict lui-même est constant (28/28 patients, 160/160 coupes). (2) La carte de `/biopsie` annonçait une mesure « sur 20 % des 569 cas tenus à l'écart de l'entraînement » : le modèle servi est ajusté sur 569/569 sans découpage, et les 97,7 % / 99,8 % décrivaient un autre modèle — aucun chiffre n'est plus affiché, l'absence de mesure hors échantillon est écrite. (3) `/biopsie` empruntait les pastilles Dice / sensibilité / faux positifs du modèle d'imagerie, sous un texte disant que ce modèle ne lit pas d'image : elles sont passées dans un bloc `limits_numbers` que la page neutralise. Vérifié dans l'app réelle (backend `dce_mri`, cas de démo 1 : coupe 52/176, probabilité par pixel 1,0000) et gardé par 9 tests de rendu (`tests/test_result_page_claims.py`) — 125 tests au total, ruff propre |

**Livrés** (branche `amelioration-docker-preprocess-env`, commités le 2026-08-18) :

| Tâche | Où | Ce qui la rend faite |
|---|---|---|
| Packaging Docker | `Dockerfile`, `docker-compose.yml` | `docker compose up --build` rejoue la démo ; image étroite (ni Spark, ni JVM, ni ITK), torch CPU, `read_only`, port publié sur `127.0.0.1` seulement, healthcheck sur `run_demo.py --check` |
| Validation de schéma | `validation.py`, branchée dans `save_preprocessed` | Dimensions, dtype, finitude et binarité du masque vérifiés au point unique d'écriture ; le seuil `MIN_IN_PLANE = 128` est calibré sur l'incident de crop du §4.2, pas deviné. 13 tests |
| Manifeste de lineage | `lineage.py` | `manifest.json` par dossier prétraité : commit (suffixé `-dirty`), source, paramètres, stats par cas. Écrit en dernier — son absence signale une run interrompue. 8 tests |
| Étape 2 branchée (branche `ameliore-le-mvp`) | `tabular_export.py`, `/biopsie`, `app/templates/biopsy.html` | Formulaire 30 champs groupé en 10 mesures × 3 statistiques, deux exemples cliquables, API JSON jumelle. Export JVM-free vérifié contre Spark sur les 569 lignes (0 désaccord, écart max 1,0 × 10⁻¹⁵). 29 tests |

Rien de tout cela ne bloque la démonstration actuelle : elle tourne depuis un clone.
Tout, en revanche, sépare cette démonstration de l'outil décrit en tête de document.

## Écarts doc ↔ code relevés le 2026-08-18

Gardés ici une fois corrigés : ce sont les chiffres qu'un relecteur vérifie en premier,
et savoir qu'ils ont dérivé une fois dit où regarder la prochaine fois.

| Constat | Où | État |
|---|---|---|
| `README.md` annonçait « 68 tests, ~6 s » ; il y en a 87, en ~7 s | `README.md` §Development | Corrigé le 2026-08-18 |
| `Final_Report.md` pointait `data/model_results.csv` ; le code écrit `reports/model_results.csv` | `Final_Report.md` vs `config.py` (`TABULAR_RESULTS_CSV`) | Corrigé le 2026-08-18 |
| `models/dce_mri_p2_negfix/` nomme une expérience, pas une couche — contredit la règle « layers, not experiments » posée dans `config.py` | `config.py` | Ouvert — un renommage casse les chemins versionnés dont dépend la démo |

### Incident réseau du 2026-09-12, et ce que les labels ont appris (2026-09-13)

**L'obstacle était transitoire, et il a été rapporté comme permanent — c'est
l'erreur à retenir.** Entre ~18 h 30 et ~20 h le 2026-09-12, tous les hôtes de
`cancerimagingarchive.net` sortaient en timeout TCP, bac à sable désactivé compris,
comme `github.com` et `sites.duke.edu`, alors que `pypi.org`, `zenodo.org`,
`huggingface.co` et `raw.githubusercontent.com` répondaient. La conclusion tirée sur
le moment — « cette machine ne joint pas TCIA » — décrivait une fenêtre de temps, pas
une propriété de la machine : le 2026-09-13 tous ces hôtes répondent et les fichiers
sont téléchargés. Un blocage réseau se re-teste avant d'être écrit au présent.

**Ce que portent les labels.** `BCS-DBT-labels-*.csv` donne le statut par étude, et
c'est le seul moyen de savoir qu'un examen est normal — « absent du CSV de boîtes » ne
le dit pas. Téléchargés (train, validation PHASE-2, test PHASE-2) avec les
`BCS-DBT-file-paths-*.csv` :

| Collection entière | Patients |
|---|---:|
| **normal** | **4 581** |
| actionable | 278 |
| benign | 112 |
| **cancer** | **89** |
| **Total** | **5 060** |

Les 82 bénins / 59 cancers du pool train+validation recoupent exactement ce qui avait
été dérivé des boîtes seules : l'étiquetage du §« Livré » tient.

**Et ces chiffres plafonnent la cible.** Le § « Ce que la cible coûte en données »
demande ~220 patients avec cancer **dans le seul jeu de test** pour mesurer une
sensibilité à ±5 points. La collection en contient **89 en tout**, entraînement
compris. Conséquence à assumer plutôt qu'à découvrir plus tard :

- la **spécificité** est mesurable finement (4 581 normaux disponibles) ;
- la **sensibilité** ne le sera pas. Un test à 20 % laisserait ~18 cancers, soit un IC
  de l'ordre de ±17 points ; même en consacrant les 89 cancers au seul test — ce qui
  n'entraînerait plus rien — on resterait vers ±8 points.

BCS-DBT ne peut donc pas départager 82,8 % de 76 % ou de 94 %. Ce n'est pas une raison
de ne pas construire la tête de décision : c'est la raison d'annoncer son IC avant de
l'entraîner, et de ne jamais présenter le point obtenu comme une comparaison au
programme national.

**Ce que `file-paths` change pour le §4.4.** Ce CSV donne (PatientID, StudyUID, View)
par fichier, et le nom de dossier de série s'y lit dans `classic_path` : l'appariement
série ↔ boîte devient une jointure, plus une inférence. Confronté à nos 262 séries —
retrouvées **262/262**, PatientID concordant **262/262** :

| | |
|---|---:|
| Vue déduite des pixels = vue vraie | **237 / 262** |
| Écarts de latéralité | **14** — les 7 patients « miroir » × 2 vues, **confirmés** |
| Écarts de suffixe (`lmlo1`, `rcc1`, `lcc2`) | 11 — vues répétées, laissées de côté au §4.4 |
| Séries annotées selon la vérité | **260** (le code en apparie 253) |
| Séries appariées à une mauvaise boîte | **4** (une acquisition répétée prise pour l'autre) |

L'inférence du §4.4 était donc juste là où elle était risquée — les 14 miroirs — et
incomplète sur les vues répétées. **Fait le 2026-09-13** : `file-paths` est la source
d'appariement, la latéralité des pixels ne sert plus qu'à décider du retournement, comme
dans le lecteur officiel. Gain annoncé 253 → 260 séries et 4 masques corrigés ; gain
obtenu, mesuré sur les fichiers, 253 → 260 séries et **exactement** 4 masques déplacés
(§4.6).

### Relevés le 2026-09-12

Audit du dépôt contre la cible produit. Presque tous sont des corrections de
documentation, pas de code, et se décident une par une. Ce qu'un utilisateur voyait
à l'écran a été traité en premier (P0, voir Livrés) ; le reste est ouvert.

| Constat | Où | État |
|---|---|---|
| « mesurée sur 20 % des 569 cas **tenus à l'écart de l'entraînement** » — le modèle servi est ajusté sur 569/569, sans split. Les 97,67 % viennent d'un autre modèle, celui d'`AnalyzeData` | `app/templates/biopsy.html` vs `train_tabular_model.py` (`pipeline.fit(labelled)`) | Corrigé le 2026-09-12 |
| Parité annoncée « sur les 569 lignes, écart max 1 × 10⁻¹⁵ » ; le test compare **5 lignes** à 1e-9, et le dit dans son propre commentaire | `plan.md`, docstring `inference.predict_tabular` vs `tests/test_tabular_export.py` | Ouvert |
| Les métriques tabulaires renvoient à `reports/model_results.csv`, **absent du disque** — comme `pca_info.csv`, `feature_contributions.csv`, `scree_plot.png` | `Final_Report.md` | Ouvert |
| « 87 tests, ~7 s » ; il y en a **174**, tous passants (116 à l'audit, plus 9 pour le P0, 16 pour la colonne `Class`, 6 pour le split stratifié, 7 pour l'appariement et 20 pour l'étape 2 en image) | `README.md` §Development | Corrigé le 2026-09-13 : **198**, après les tests de la jointure et des labels |
| « 0,76 s par volume, 4,5 ms par coupe » ; l'artefact dit **0,825 s** et **4,83 ms** | `plan.md` §4.3 et `DEMO.md` vs `eval_report.json` | Ouvert |
| « temps de calcul ~110 ms » ; mesuré 69-73 ms à chaud, 585 ms au premier appel | `README.md`, `DEMO.md` | Ouvert |
| Checkpoint par défaut documenté `results_mri_p2/unet_best.pt` ; c'est `models/dce_mri_p2_negfix/unet_best.pt` | docstring `inference.predict_dce_mri` | Ouvert |
| Backend `unet` présenté comme disponible ; son checkpoint n'existe plus | `app/README.md`, `app/predictor.py` | Ouvert |
| Logs annonçant `data/transformed_data.csv`, `data/pca_info.csv`, `data/scree_plot.png` ; le code écrit dans `reports/` et `plots/` | `TransformData.transform_data` | Ouvert |
| IC 95 % top-1 `[25,0 – 60,7]` : aucun code ni artefact versionné ne la produit | `plan.md` §4.3 | Ouvert |
| Fuite de préprocessing : imputation, scaler et PCA ajustés sur les 569 lignes **avant** le `randomSplit`, donc les 97,67 % / 99,89 % sont optimistes | `TransformData.transform_data` → `AnalyzeData.prepare_data` | Ouvert |
| Étape 2 annoncée « livrée » ; la branche `ameliore-le-mvp` est locale, absente d'`origin` | `plan.md` §Livrés | Ouvert |
| `/biopsie` affichait les pastilles Dice 0,53 / sensibilité 88 % / faux positifs 99,97 % — les chiffres du modèle d'imagerie, sous un texte disant que ce modèle-ci ne lit pas d'image | `app/templates/base.html` (bloc de limites partagé) | Corrigé le 2026-09-12 |
| La pastille de moteur du bandeau est **vide** sur `/biopsie` : la route ne passe pas `backend` au gabarit, que `base.html` attend | `app/server.py` (`biopsy_form`, `biopsy_predict`) vs `base.html` | Ouvert — cosmétique, aucune affirmation fausse |
| Un `cudaErrorIllegalAddress` sur `/demo/1` dans le serveur Flask, **observé une fois, non reproduit** : le même appel passe en direct (coupe 52/176, 1,0000) et le GPU calcule normalement juste après. Noté parce qu'un plantage de service ne doit pas rester sans trace, pas parce qu'il est caractérisé | `app/predictor.py` (`DceMriUNetPredictor`) | Ouvert — à re-observer avant d'enquêter |

---

> **Note d'historique (2026-08-10).** Ce document a commencé comme un plan en six
> semaines, rédigé sur l'hypothèse « pas d'accès au code ». Ses Parties 1 et 2 — une
> grille d'audit, un script d'inventaire et une feuille de route par jalons — ont été
> retirées : elles décrivaient une arborescence qui n'existe plus, et un plan dont les
> jalons sont livrés. Les garder aurait fait de la première moitié du document le
> contraire de ce qu'un lecteur en attend. Elles restent dans l'historique git.
>
> Ce qui est conservé ci-dessous l'est parce que le code y renvoie explicitement :
> la Partie 3 pour les tokens de la charte, §4.1 à §4.3 pour les mesures.

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

---

### 4.1 Optimisation entraînement & préprocessing (2026-07-26)

Revue de littérature puis application de ce qui tient sur ces données. **Ce qui a été mesuré, pas supposé.**

#### Appliqué et validé

| Levier | Source / justification | Résultat mesuré |
|--------|------------------------|-----------------|
| **Banque de coupes memmap** (`imaging/slicebank.py`, `--slice-bank`) | Diagnostic local : lire une coupe ré-inflatait un `.npz` entier (0,20 s mesuré) avec un cache de 4 volumes sur 130 mélangés → GPU à 5-28 % d'utilisation. La banque paie la décompression une seule fois dans un memmap plat. | **143 s/époque contre 1020 s, soit 7,1×.** 30 époques : 8,5 h → 1,2 h. C'est ce qui rend les ablations abordables. |
| **2ᵉ passe post-contraste** au lieu de la 1ʳᵉ (`post_phase_rank=2`) | Zhou et al., [PMC10658935](https://pmc.ncbi.nlm.nih.gov/articles/PMC10658935/) : comparaison frontale sur cette tâche exacte, 2ᵉ soustraction > 1ʳᵉ (DSC p<0,05, 2D et 3D). Vérifié ici : les deux phases existent pour les mêmes 186 patients, donc zéro coût en échantillon. | ⚠️ **N'a pas répliqué ici** : 0,552 contre 0,550 en phase 1, soit +0,001 — du bruit. Voir l'encadré ci-dessous. Conservé comme défaut (aucun coût, et cohérent avec la littérature) mais **ne pas le présenter comme un gain**. |
| **Précision mixte AMP** (`torch.amp`, `--no-amp` pour désactiver) | Pratique standard CUDA. `unscale_` avant le clipping pour que le seuil porte sur les vraies normes de gradient. | Actif, aucune instabilité numérique observée sur 30 époques. |
| **U-Net 2D conservé** | Même source (PMC10658935) : leur 2D (DSC 0,806 sur masses) bat leur 3D (0,767). | Aucun changement nécessaire — l'architecture existante est le bon choix. |

#### Écarté sur preuve — encodeur ImageNet pré-entraîné

La littérature le recommande pour les petits jeux de données, et le code le supportait déjà
(`--architecture pretrained`). **Il s'effondre ici** : Dice validation à 0,000 de l'époque 10 à 30,
sans récupération. Cause vérifiée : le U-Net resnet34 de `smp` contient **46 couches BatchNorm**
(contre 0 dans le modèle from-scratch, qui utilise 18 GroupNorm). C'est exactement le mode d'échec
déjà documenté dans `imaging/unet.py` : avec une fraction de lésion minuscule et de petits batches,
les statistiques BatchNorm ne convergent jamais vers celles de l'inférence. Dice test 0,414 (issu
d'un checkpoint précoce sauvé avant l'effondrement) contre **0,550** pour le from-scratch.
*(Chiffre corrigé le 2026-08-10 : cette ligne annonçait 0,467, qui ne correspond à aucune
mesure de ce run — ni au Dice test 0,550225, ni au meilleur Dice validation 0,655. L'écart
réel est donc plus large que ce qui était écrit. Les deux CSV sont dans
`reports/experiments/results_ablation_*`.)*

> **Piste restante si besoin** : convertir les BatchNorm en GroupNorm tout en gardant les poids
> convolutifs pré-entraînés — récupère le prior ImageNet sans l'instabilité. Non testé.

> **Code supprimé le 2026-08-02.** L'option `--architecture pretrained`, ses arguments
> (`--encoder-name`, `--encoder-weights`) et la dépendance `segmentation-models-pytorch` ont été
> retirés du dépôt. Garder une option de ligne de commande qui produit silencieusement un modèle
> effondré coûte plus qu'elle ne rapporte. Le raisonnement, les chiffres et la piste ci-dessus
> restent ici ; le code est dans l'historique git si besoin de repartir de là.

#### Non appliqué (coût > bénéfice attendu à ce stade)

Correction de champ N4 et recadrage sur la **région mammaire** (à ne pas confondre avec le recadrage
sur la lésion, qui lui trichait) — [PMC9889463](https://pmc.ncbi.nlm.nih.gov/articles/PMC9889463/)
atteint DSC 0,781, comparable à l'accord inter-radiologue (0,778), avec ce préprocessing. Coûteux en
calcul ; à déclencher seulement si le Dice plafonne (voir P2).

#### Comparaison des configurations (186 patients, pleine trame, 30 époques, config identique)

| Configuration | Dice test | IoU test | Pic Dice val |
|---------------|:---------:|:--------:|:------------:|
| Phase 1 + encodeur pré-entraîné | 0,414 | 0,300 | effondré (0,000 dès l'ép. 10) |
| Phase 1 + GroupNorm (contrôle) | 0,550 | 0,417 | **0,655** |
| Phase 2 + GroupNorm (retenue) | 0,552 | 0,418 | 0,618 |

> **Le changement de phase n'a rien apporté de mesurable.** L'écart phase 2 − phase 1 est de
> **+0,001 sur le Dice test**, et le pic de validation est même *meilleur* en phase 1 (0,655 contre
> 0,618). Le résultat de Zhou et al. (p<0,05) **ne réplique pas sur nos données**. Explication la
> plus plausible : leurs masques sont des contours experts, où la conspicuité fine de la lésion
> compte ; les nôtres sont des boîtes englobantes, une cible bien plus grossière qui absorbe ce
> genre de différence. Un effet réel mais petit serait aussi invisible sur un split test de ~28
> patients. **À retenir : le seul gain franc de cette session est l'accélération 7,1×, qui est un
> gain d'ingénierie, pas de précision. La précision reste à ~0,55 dans toutes les configurations
> viables.**

> **Lecture honnête de ces chiffres.** Le Dice n'est pas comparable à la littérature (~0,80) : nos
> masques sont des **boîtes englobantes** TCIA, pas des contours fins, ce qui plafonne mécaniquement
> le Dice atteignable. Ces valeurs servent à comparer nos configurations entre elles, pas à se
> mesurer à l'état de l'art. Aucun de ces chiffres ne doit être présenté comme une performance
> clinique.

#### Incident à noter

Le smoke test (`--smoke-test`) écrivait par défaut dans `--output-dir results`, ce qui a **écrasé
`results/unet_best.pt`**, le checkpoint DBT servi par l'app en backend `unet`. Les données sources
(`preprocessed_data/`, 147 volumes) sont intactes, donc le modèle est ré-entraînable, mais il est
actuellement perdu. Correctif appliqué : un smoke run écrit désormais dans `results/smoke_test/`.
**À faire** : ré-entraîner le modèle DBT si le backend `unet` doit resservir.

---

### 4.2 Échec de la localisation automatique sur volume complet — et contournement (2026-07-26)

**Constat, vérifié sur les 186 patients, pas une hypothèse** : le pipeline choisi pour le MVP
(Phase 2 + GroupNorm) **ne trouve jamais tout seul la bonne coupe** dans un volume complet.
`predict_dce_mri` scanne les ~150-200 coupes d'un volume et garde celle où la confiance (probabilité
max d'un pixel) est la plus haute. Sur les 186 patients : **0/186** cette coupe tombe sur une coupe
contenant réellement la lésion.

**Cause identifiée** : la confiance du modèle est saturée à ~1,0 sur *chaque* coupe du volume, y
compris les coupes vides en bordure — pas de signal discriminant. Ce n'est pas propre à un seul
pixel isolé : l'aire moyenne de la région prédite au-dessus du seuil est quasi identique sur les
coupes avec lésion (1228,6 px) et sans (1228,3 px). Deux corrections tentées, aucune n'a résolu le
problème :

1. **Ratio négatif réaliste** (`neg_per_pos` 2 → 8, pour se rapprocher du ratio réel ~1:8 d'un volume
   complet) : Dice sur coupes positives amélioré (0,552 → 0,580), mais localisation réelle **toujours
   0/186**. La métrique d'entraînement s'est améliorée sans que le vrai problème bouge — piège à
   retenir : le Dice mesuré en `positive_only=True` ne dit rien de la capacité à *trouver* la coupe.
2. **Bug de NaN découvert en cours de route** : `FocalTverskyLoss` (`(1-tversky).clamp_min(eps) **
   gamma`, gamma=0,75) a un gradient qui diverge quand `(1-tversky) → 0`, ce qui devient fréquent avec
   plus de coupes négatives faciles. Sous AMP (fp16), ça produit un NaN qui finit par corrompre les
   poids (effondrement Dice → 0,000, irréversible). **Corrigé** : la perte force maintenant un calcul
   en fp32 (`imaging/metrics.py`, `FocalTverskyLoss.forward`), indépendamment du contexte autocast
   englobant. Le correctif a retardé la divergence (époque 15 au lieu de 11) mais ne l'a pas éliminée
   — la source résiduelle est probablement un débordement fp16 dans le forward pass du modèle
   lui-même, pas seulement dans la perte. Non résolu ; le checkpoint sauvegardé avant la divergence
   reste valide (la logique de sauvegarde ne retient que la meilleure époque lissée).

**Ce que ça signifie concrètement** : le modèle segmente correctement une lésion *quand on lui
montre la bonne coupe* (Dice 0,58 sur coupes positives, IoU jusqu'à 0,83 sur les meilleurs cas
vérifiés manuellement), mais ne sait pas la trouver seul dans un volume brut. C'est un problème de
sélection, pas de segmentation.

**Contournement retenu pour le MVP** : mode « coupe figée ». `TransformData.make_demo_case(source,
out, slice_index)` copie un `.npz` en y ajoutant une clé `forced_slice` ; `inference._localize_lesion`
détecte cette clé et évalue uniquement cette coupe au lieu de scanner tout le volume. Trois cas ont
été sélectionnés en évaluant le modèle sur la coupe la plus représentative de la lésion (aire de
masque maximale) pour chaque patient, puis en retenant les 3 meilleurs par IoU réel :

| Cas | Patient | Coupe | IoU (coupe forcée) | Cadre |
|-----|---------|:-----:|:-------------------:|-------|
| 1 | Breast_MRI_135 | 52 | 0,830 | 39×42 px |
| 2 | Breast_MRI_105 | 62 | 0,738 | 42×44 px |
| 3 | Breast_MRI_079 | 104 | 0,728 | 66×92 px |

Vérifiés via l'app réelle (`/api/predict`), 3 exécutions chacun : **9/9 résultats identiques**
(déterministe — pas d'aléatoire à l'inférence), overlays inspectés visuellement, cadres focaux et
distincts du réhaussement parenchymateux diffus environnant. Fichiers dans `demo_cases/`
(non versionnés, comme le reste des données patient — régénérables via le script ci-dessus).

**Limitation à afficher clairement dans toute démo** : ces 3 cas fonctionnent parce que la coupe a
été choisie à l'avance par un humain, pas par le modèle. Un upload libre d'un volume complet
n'aboutit pas encore à une localisation fiable. C'est une limitation connue et documentée, pas un
défaut caché.

**Prochaine étape si l'upload libre doit fonctionner** : la piste retenue est une tête de
classification de coupe entraînée séparément (« cette coupe contient-elle une lésion ? ») avec des
négatifs durs piochés dans tout le volume, plutôt que de réutiliser la confiance de segmentation
comme signal de tri. Non commencé — effort non trivial, à cadrer avant de s'y engager.

#### Test navigateur réel (tâche P0 « parcours complet »)

L'upload de fichier natif (sélecteur de fichier du navigateur) n'est pas automatisable dans cet
environnement : par sécurité, aucun navigateur ne permet de définir `input[type=file].value` par
script (vérifié : `InvalidStateError` levée). Vérifié à la place : la page se charge (200, formulaire
présent), les routes `/predict` (HTML) et `/api/predict` (JSON) répondent correctement en conditions
réelles avec les 3 cas de démo, et l'app ne contient **aucun JavaScript côté client** (HTML pur via
Jinja) — donc pas de logique JS susceptible de casser silencieusement hors de portée de ces tests.

---

### 4.3 Évaluation chiffrée et intervalles de confiance (2026-08-02)

`imaging/evaluate.py` remplace le chiffre unique de l'entraînement (une moyenne de Dice sur les
coupes positives, sans incertitude) par ce qu'un lecteur sceptique demandera. Sur les **28 patients
de test** (split par patient, seed 42 — le même qu'à l'entraînement), 4 782 coupes :

| Mesure | Valeur | IC95 |
|--------|:------:|:----:|
| Dice, coupes avec lésion | 0,533 | 0,473 – 0,593 |
| IoU, coupes avec lésion | 0,401 | 0,346 – 0,455 |
| Sensibilité (IoU ≥ 0,1) | 88,0 % | 81,9 – 93,4 % |
| Sensibilité (centre visé juste) | 81,1 % | 74,0 – 87,7 % |
| Faux positifs par volume | 222,2 | 204,7 – 237,4 |
| Coupes saines avec alarme | 99,97 % | 99,92 – 100 % |
| Temps par volume (RTX 5060) | 0,76 s | médiane 0,75, max 1,29 |
| Temps par coupe | 4,5 ms | — |

**Sur le bootstrap.** Le rééchantillonnage porte sur les **patients**, pas sur les coupes : les
coupes d'un même patient partagent l'anatomie, la lésion et l'acquisition, donc les traiter comme
indépendantes produirait un intervalle artificiellement étroit. 10 000 rééchantillonnages,
percentiles 2,5 / 97,5.

**Écart avec le 0,580 annoncé.** Ce dernier est une moyenne **par coupe** (chaque coupe pèse pareil,
donc un patient à 40 coupes lésionnelles pèse dix fois un patient à 4) ; 0,533 est une moyenne **par
patient**. Les deux sont défendables ; la moyenne par patient est celle qui a un IC interprétable et
c'est donc elle qui est citée désormais.

**Le chiffre nouveau et important : 99,97 %.** Le modèle lève une alarme sur pratiquement *toutes*
les coupes sans lésion, ~222 fausses zones par examen. C'est la formulation quantitative de l'échec
0/186 de §4.2, et elle montre que le problème n'est pas un mauvais classement mais une **absence
totale de signal discriminant** — la métrique d'entraînement (`positive_only=True`) ne pouvait
structurellement pas le voir, puisqu'elle ne regarde jamais une coupe saine.

**Le temps d'inférence est un non-sujet** : 0,76 s par volume complet contre les 10 s visées au
Jalon 2. Le goulot n'est pas le calcul. À noter : l'app rechargeait le checkpoint (31 Mo) à chaque
requête, ce qui dominait la réponse ; le prédicteur met désormais le modèle en cache
(352 ms à la première requête, ~40 ms ensuite).

#### Tête de classification de coupe — première mesure

La piste annoncée en §4.2 est implémentée (`imaging/sliceclf.py`) : un petit CNN GroupNorm entraîné
sur **toutes** les coupes (22 010 coupes d'entraînement, 15 % positives, `pos_weight` = 5,5) plutôt
que sur un sous-échantillon de négatifs, et sélectionné sur le top-1 — « la coupe la mieux notée du
volume contient-elle réellement une lésion ? », exactement la métrique qui valait 0/186.

**Résultat, 25 époques, 28 patients de test :**

| Mesure | Confiance de segmentation (avant) | Classifieur dédié |
|--------|:---------------------------------:|:-----------------:|
| Top-1 (la meilleure coupe contient la lésion) | **0,0 %** (0/186) | **42,9 %** [IC95 25,0 – 60,7] |
| Top-3 | — | 50,0 % [IC95 32,1 – 67,9] |
| Rang médian de la 1ʳᵉ coupe correcte | — | 3,5 |
| AUC (par patient) | ~0,50 par construction (aucun signal) | 0,803 |
| Lésion dans le top-5 | — | 16/28 patients |

**Lecture.** Le problème est **tractable** : là où la confiance de segmentation n'avait aucun pouvoir
discriminant (aire prédite identique sur coupes avec et sans lésion, §4.2), un modèle entraîné pour
la tâche de tri atteint 0,80 d'AUC. Le passage de 0 % à 43 % est franc et ne tient pas à la chance —
l'IC95 exclut largement zéro.

**Mais ce n'est pas livrable comme chemin principal.** 43 % veut dire qu'un upload libre se trompe
plus d'une fois sur deux, et l'IC est large (25–61 %) parce que 28 patients c'est peu. Écart val/test
notable aussi (57 % contre 43 %), cohérent avec des échantillons de cette taille. Les cas de démo
gardent donc leur coupe figée : une démo qui échoue une fois sur deux est pire qu'une coupe assumée
comme choisie à l'avance.

**Branché malgré tout pour l'upload libre** (`inference.load_slice_classifier`, consulté seulement
quand aucune coupe n'est imposée) : dans ce cas précis l'alternative est 0 %, donc 43 % est un gain
net. Le mécanisme utilisé remonte dans le résultat (`slice_selector` : `pinned` / `classifier` /
`segmentation_confidence`) et l'app affiche lequel a servi avec son taux de réussite, plutôt que de
laisser croire à une détection autonome.

**Rang médian 3,5 contre rang moyen 19,7** : la distribution est bimodale — soit le modèle vise
juste ou presque, soit il part complètement ailleurs. C'est ce qui suggère la piste suivante :
présenter les **5 meilleures coupes candidates** à revoir plutôt qu'une seule (16/28 patients, 57 %,
auraient leur lésion dans ce lot). Cadrage « candidats à examiner » plutôt que « voici la lésion » —
honnête et utile, contrairement à un top-1 à 43 % présenté comme une réponse.

**Pistes non explorées** (par ordre de rapport attendu) : entraîner sur plus de patients (186 reste
faible pour une tâche de tri), exploiter le contexte 3D (une lésion s'étend sur plusieurs coupes
consécutives — un modèle 2,5D avec ±2 coupes en entrée est peu coûteux), et calibrer sur la
validation plutôt que de prendre l'arg-max brut.

### 4.5 Étape 2 en version image : mesurée, et elle ne marche pas (2026-09-12)

`imaging.lesionclf` pose à l'image la question de l'étape 2 — cette lésion est-elle
bénigne ou maligne ? — sur les 253 recadrages étiquetés, 130 patients, 55 cancers.
**Ce n'est pas un modèle de détection** : le volume est recadré autour de la boîte
annotée, donc la position de la lésion est donnée. Aucun chiffre ci-dessous ne dit
quoi que ce soit sur le fait de trouver un cancer dans un examen de dépistage.

**Protocole.** Validation croisée à 5 plis stratifiés par patient plutôt qu'un test
unique : sur 130 patients, un test à 15 % laisse ~20 patients et ~8 cancers, et l'IC
couvrirait presque tout. Chaque patient est noté une fois par un modèle qui ne l'a pas
vu. **Aucune sélection dans un pli** — budget d'époques fixé d'avance, dernière époque
évaluée : choisir un checkpoint sur les patients tenus à l'écart est la façon
habituelle de rendre un chiffre de VC optimiste, et ce dépôt en a déjà publié un
(§`AnalyzeData`). Agrégation : probabilité moyenne sur les coupes de lésion du
patient. Encodeur réutilisé de `sliceclf`, entraîné de zéro — un encodeur ImageNet
avait été écarté sur preuve (§4.1) et ses poids sont de toute façon inaccessibles
depuis cette machine.

**Les deux mesures.** La première a été faite avec un défaut que j'ai introduit, la
seconde après l'avoir corrigé. Les deux sont versionnées
(`models/lesionclf/cv_report_stretched.json` et `cv_report.json`).

| | Recadrage étiré (1ʳᵉ) | Fenêtre à échelle constante (2ᵈᵉ) |
|---|---:|---:|
| **ROC-AUC patient** | **0,591 [0,491 – 0,693]** | **0,513 [0,411 – 0,615]** |
| Sensibilité à 0,5 | 54,5 % | 27,3 % |
| Spécificité à 0,5 | 61,3 % | 73,3 % |
| VPP (prévalence 42,3 %) | 50,8 % | 42,9 % |
| Exactitude | 58,5 % | 53,8 % |
| « Toujours bénin » | 57,7 % | 57,7 % |

**Les deux intervalles contiennent 0,5, et aucune des deux exactitudes ne bat la
constante triviale.** Ce modèle ne sépare pas le bénin du cancer sur ce corpus.

**Le défaut corrigé entre les deux.** Le premier passage redimensionnait à 224 des
recadrages allant de 97×149 à 710×505. Comme le recadrage est serré autour de la
boîte, sa taille suit celle de la lésion : le redimensionnement commun effaçait donc
la taille de la lésion — un des premiers indices de malignité — et floutait les
petites. Remplacé par une fenêtre de **640 px natifs** centrée sur la lésion, complétée
par des zéros (le volume est z-normalisé, zéro est donc sa moyenne) : elle contient
97,3 % des boîtes en entier, contre 91,0 % à 512 px et 60,5 % à 256. Deux tests
chiffrent l'écart : avec la fenêtre, une lésion de 24 px rend plus de 4 fois l'aire
d'une de 8 px ; avec l'étirement, le rapport tombe entre 0,8 et 1,25 — indistinguable.
La fenêtre constante en pixels ne vaut fenêtre constante en millimètres que parce que
la géométrie du détecteur est constante ici (2457 lignes partout) : **ces DICOM ne
portent aucun tag d'espacement de pixel**, et c'est le prix de l'hypothèse.

**Diagnostic.** La perte d'entraînement descend (0,83 → 0,27 sur le pli 2, 0,75 sur le
pli 3) pendant que l'AUC hors-pli reste au hasard : le modèle apprend ses ~1 100
coupes d'entraînement par cœur, sur 104 patients, et ne généralise rien. La variance
entre plis est forte, ce qui est ce à quoi ressemble un entraînement instable sur trop
peu d'exemples. **1 366 coupes de lésion au total** : c'est le chiffre à retenir.

**Arrêté à deux mesures.** Régler des hyper-paramètres contre cette même validation
croisée jusqu'à ce que le chiffre monte le viderait de son sens — c'est la version
lente de la fuite que le §« sélection » écarte. Ce qui suit sont des pistes, pas des
promesses, par ordre de rapport attendu :

- **Plus de patients.** La collection compte 5 060 patients ; 134 sont téléchargés,
  et le réseau bloque le reste (voir l'obstacle du 2026-09-12). C'est de loin le
  premier levier, et il n'est pas algorithmique.
- **Des features pré-entraînées**, inaccessibles depuis cette machine, et écartées sur
  preuve pour la segmentation (§4.1) — ce qui ne dit rien de leur valeur pour une
  classification.
- **Un modèle par patient plutôt que par coupe** : un patient a jusqu'à 4 vues de la
  même lésion, traitées aujourd'hui comme des exemples indépendants.
- **La colonne `AD`** (distorsion architecturale, 84 lignes sur 299) comme étiquette
  auxiliaire.

**Ce que ça ne bloque pas.** L'étape 2 du produit est servie par le modèle tabulaire
Wisconsin, qui répond depuis un clone. La version image en est un complément, pas un
prérequis — et son échec mesuré vaut mieux que son absence de mesure.

### 4.4 Appariement boîte ↔ série DBT : le tag DICOM de latéralité est faux (2026-09-12)

Parti d'un détail — 5 séries sur 147 au volume constant après recadrage (§ ci-dessus) —
et arrivé à un défaut qui touchait l'ensemble du corpus DBT.

**Ce que dit la source.** Le lecteur officiel du jeu de données
(`mazurowski-lab/duke-dbt-data`, `duke_dbt_data.py`, récupéré via
`raw.githubusercontent.com`) déduit la latéralité **des pixels** — quel bord porte du
signal — et documente son propre accès au tag DICOM par
« *Unreliable - DICOM laterality is incorrect for some cases* ». Il retourne ensuite
l'image de 180° (`np.flip(..., axis=(-1, -2))`) quand la latéralité de l'image ne
correspond pas à celle de la vue annotée, parce que les boîtes vivent dans ce
repère-là. Notre `dbt_series_view` appariait **par ce tag**, et ne retournait jamais rien.

**Ce que ça coûtait, mesuré sur nos 262 séries** (décodage complet, 23 min) :

| | |
|---|---:|
| Latéralité par le tag DICOM | `L` sur les **262** séries — constante, donc fausse |
| Latéralité par les pixels | R 134 / L 128 |
| Groupes boîte (patient + vue) jamais appariés, patient présent sur disque | **123 / 260** dont 58 `rmlo`, 54 `rcc` |
| Séries annotées trouvées | **147** au lieu de 253 |
| Masques posés sur du fond au lieu du tissu | **23 / 147** |

Le test décisif est l'intensité dans la boîte : sur les 124 séries concordantes, la
région annotée fait 416 de moyenne et 467 d'étendue (du tissu) et le retournement
l'enverrait à 90 / 155 (du fond) ; sur les 23 discordantes **c'est exactement
l'inverse** — 86 / 119 telles que peintes, 399 / 378 après correction. Les 5 volumes
constants sont tous dans ce lot.

**Les 23 se séparent en deux cas, et la distinction est tranchée par les données, pas
choisie.** 9 séries ont une boîte pour la vue déduite des pixels : ce sont des séries
droites appariées à la boîte gauche du même patient, et la correction est d'apparier
la bonne. Les 14 autres appartiennent à **7 patients dont toutes les boîtes sont
gauches** (`lcc`, `lmlo`), qui ont exactement 2 séries sur disque (cc et mlo) et dont
**les deux lisent « droite » en pixels** — zéro boîte droite. Ce sont donc des études
du sein gauche stockées en miroir, le cas que le lecteur officiel traite par le
retournement. L'hypothèse concurrente — séries droites non annotées, séries gauches
non téléchargées — demanderait que le téléchargement ait pris les 2 séries sans
annotation et laissé les 2 annotées, pour 7 patients indépendants, alors que la liste
de patients est justement construite depuis le CSV de boîtes.

**Décision.** La latéralité vient des pixels (`image_laterality`), l'incidence du
header (`dbt_view_position` — `ViewPosition` est fiable), et une étude stockée en
miroir est retournée plutôt que mal appariée. C'est la sémantique du lecteur officiel,
reconstruite sans `BCS-DBT-file-paths-*.csv` — qui donnerait l'appariement
série ↔ boîte de façon autoritative, et qui est sur l'hôte injoignable. Le retournement
est appliqué au **volume** et non aux coordonnées, pour que masque, `crop_offset` et
`.npz` soient tous dans un seul repère, celui de l'annotation. Le manifeste porte
`mirrored` par cas et `mirrored_series` au total : c'est une propriété de la
construction du cas, pas du volume.

**Vérifié sur les vraies données avant de relancer**, sur 11 séries choisies pour
couvrir chaque cas (2 miroirs, 2 séries droites jusque-là invisibles, les 4 séries de
DBT-P00538, 1 déjà correcte, 2 sans annotation) : 9 sauvées, 2 miroirs, 2 ignorées —
conforme à la prédiction, et **0 avertissement de validation** là où le même
échantillon en produisait 2. Les deux séries de DBT-P00538 sortent désormais avec
173 830 et 38 025 voxels de lésion sur du tissu, classées `cancer`.

**Ce qui reste ouvert ici.** 12 lignes de boîtes portent une vue suffixée (`lmlo1`,
`rmlo1`, `lcc1`, `lcc2`, `rcc1`) qu'aucune de nos séries ne peut réclamer sans savoir
laquelle des acquisitions répétées elle est ; elles restent non appariées. Et
`inference.load_dbt_dicom` ne normalise aucune latéralité : si un modèle DBT est
réentraîné sur ce corpus, l'inférence devra appliquer la même règle, sinon une moitié
des examens arrivera dans le mauvais repère.

### 4.6 Appariement boîte ↔ série : la collection le dit, il suffisait de le lire (2026-09-13)

Le §4.4 a remplacé un tag qui mentait par une **inférence** sur les pixels. Elle était
bonne — 23 masques remis sur du tissu — et elle restait une inférence. `BCS-DBT-file-paths-*.csv`,
téléchargé le 2026-09-13, donne `(PatientID, StudyUID, View)` pour **chaque dossier de
série** : l'appariement devient une jointure, et les pixels gardent un seul rôle, celui
que le lecteur officiel leur donne, décider du retournement.

**Ce que la jointure trouve, mesuré avant de recalculer quoi que ce soit.** Les
20 311 lignes de l'inventaire couvrent nos **262 dossiers sur 262**, avec un
`series_uid` unique par ligne (le nom de dossier se lit en avant-dernier segment de
`classic_path`) :

| | |
|---|---:|
| Séries annotées par la jointure | **260** (l'inférence en trouvait 253) |
| Lignes de boîtes appariées | 284 / 299 |
| Les 15 lignes restantes | 9 patients **absents du disque** — rien n'est perdu |
| Patients | **132** — 76 bénins, 56 cancers, 0 mélangé |

**Ce que la reconstruction a changé, mesuré fichier par fichier contre l'ancien corpus** :

| | Pixels (§4.4) | Jointure | |
|---|---:|---:|---|
| Séries | 253 | **260** | +7, aucune perdue |
| Séries bénignes / cancers | 151 / 102 | **153 / 107** | |
| Patients bénins / cancéreux | 75 / 55 | **76 / 56** | |
| Masques déplacés | — | **4** | exactement les 4 acquisitions répétées annoncées |
| Séries lues comme miroir | 14 | **14** | les 7 patients du §4.4, confirmés par la source |
| Séries sans boîte sur disque | 9 | **2** | |
| Masques vides | 0 | **0** | |
| Avertissements de validation | 0 | **0** | |
| Taille / durée | 1,06 Go | **1,00 Go** / 56,1 min | |

Les 4 masques déplacés sont la mesure du défaut que le §4.4 avait laissé ouvert :
DBT-P01347, DBT-P02750, DBT-P03423 et DBT-P02798, chacun avec deux acquisitions de la
même vue (`lmlo` et `lmlo1`, par exemple), dont le masque venait de l'autre acquisition.
Aucun pixel ne pouvait trancher ce cas : l'image des deux acquisitions est le même sein
sous la même incidence. Les **11 séries à vue répétée** sont d'ailleurs appariées pour la
première fois (`lcc1`, `lcc2`, `lmlo1`×3, `rcc1`×2, `rmlo1`×4).

**Ce que la jointure permet de vérifier, et que l'inférence ne permettait pas.** La vue
n'étant plus déduite de l'en-tête, l'en-tête devient un témoin : `PatientID` et
`ViewPosition` sont comparés à l'inventaire et un désaccord est journalisé. Sur les
262 séries, aucun. Le manifeste porte maintenant la vue et l'étude de chaque cas, plus
le chemin des tables qui l'ont produit — de quoi refaire la jointure sans relire un
volume.

**Ce qui est supprimé.** `dbt_series_view`, `_candidate_boxes` et `_select_boxes` n'ont
plus d'appelant : deux façons d'apparier, dont une mesurée fausse 25 fois sur 262,
c'est un piège qu'un corpus reconstruit par distraction paierait sans rien dire. La
latéralité par les pixels reste (`image_laterality`), pour le retournement.

**Ce qui reste ouvert ici.** `inference.load_dbt_dicom` ne normalise toujours aucune
latéralité : un modèle réentraîné sur ce corpus verra une moitié des examens dans le
mauvais repère si l'inférence n'applique pas la même règle. C'était déjà la dernière
ligne du §4.4 ; la jointure ne la traite pas.

### 4.7 Tête de décision au niveau examen : mesurée, et elle n'apprend rien (2026-09-13)

`imaging.examclf` pose enfin la vraie question de l'étape 1 — cancer ou pas, sur un
**examen entier**, pas un recadrage déjà centré sur une lésion comme `lesionclf`
(§4.5). C'est la première mesure jamais faite sur les deux classes ensemble : jusqu'ici
aucun corpus ne contenait de négatif, donc aucune spécificité, VPP ou ROC-AUC patient
n'était calculable du tout (voir « Cible chiffrée »). `TransformData.preprocess_dbt_exams`
a levé ce blocage (P1 du 2026-09-13, ci-dessus) ; ce paragraphe en publie la première
mesure.

**Protocole.** Agrégation par multi-instance learning : l'étiquette est au niveau
examen mais le signal ne l'est pas — la plupart des coupes d'un examen cancer ne
montrent rien — donc le score d'un sac est le **maximum** sur les coupes échantillonnées
à l'entraînement (`--bag-size 16`, tirées sans remise si l'examen en a assez) et sur
**toutes** les coupes de l'examen à l'évaluation, jamais la moyenne (qui diluerait la
minorité qui compte, la même faille documentée dans `sliceclf`). Score patient : le
maximum sur les examens du patient, une seule vue suspecte suffit. Encodeur
`sliceclf.SliceClassifier`, réentraîné de zéro. Même discipline de validation croisée
que `lesionclf` : 5 plis stratifiés par patient, budget de 25 époques fixé d'avance,
dernière époque évaluée, aucune sélection sur les patients tenus à l'écart.

**Le corpus.** 870 examens, 59 529 coupes, 272 patients, **56 cancers**. Mesuré dans
la banque de coupes : un examen cancer porte en moyenne **5,4 coupes peintes** (médiane
5) sur une profondeur moyenne de 71 — le signal utile est **moins de 1 %** des 59 529
coupes de tout le corpus.

**Le résultat.**

| Mesure | Valeur |
|---|---:|
| ROC-AUC patient | **0,457 [0,369 – 0,544]** |
| Sensibilité à 0,5 | **0,0 %** |
| Spécificité à 0,5 | 100 % |
| VPP à 0,5 | non définie (0 positif prédit) |
| Exactitude | 79,4 % |
| « Toujours pas de cancer » | 79,4 % |

**L'IC contient 0,5, et le modèle ne prédit jamais un cancer** : au seuil 0,5, les 272
patients — les 56 cancers compris — sont tous classés négatifs. L'exactitude est
exactement celle de la règle triviale. Rapport et prédictions versionnés
(`models/examclf/cv_report.json`, `cv_predictions.csv`) ; le checkpoint ne l'est pas.

**Diagnostic, différent de celui du §4.5.** Là où `lesionclf` mémorisait ses coupes
d'entraînement (perte tombée à 0,27 pendant que l'AUC restait au hasard), ici la
perte **ne descend dans aucun des 5 plis** : elle oscille entre 1,25 et 1,58 sur les
25 époques, aussi bruitée à l'époque 25 qu'à l'époque 5. Le modèle n'apprend pas même
à sur-ajuster ses propres sacs d'entraînement. Un facteur mesuré y contribue : un sac
aléatoire de 16 coupes tirées parmi les ~71 d'un examen cancer **manque toutes ses
coupes peintes une fois sur quatre** (25 % en moyenne, calculé sur les 107 examens
cancer) — un quart des pas de gradient sur un sac positif pousse donc une coupe qui ne
contient rien, un signal contradictoire pur. Ce facteur ne suffit pas à tout expliquer
(75 % des pas restent informatifs) : la tâche elle-même — une trame entière de 224 px,
sans a priori de position, un encodeur entraîné de zéro sur 56 patients cancer — est
plus dure que celle de `lesionclf`, qui échouait déjà en partant d'une lésion déjà
localisée.

**Arrêté à une mesure**, pour la même raison qu'au §4.5 : régler les hyper-paramètres
contre cette même validation croisée jusqu'à ce que le chiffre monte viderait le
protocole de son sens. Pistes, par ordre de rapport attendu, pas de promesses :

- **Plus de patients cancer.** 56 sur les 89 de toute la collection (§ »reste ouvert«,
  ligne P2 étape 2 image) — le même levier qu'au §4.5, pas encore tiré ici non plus.
- ~~**Un sac plus grand.**~~ Essayée et négative, voir ci-dessous.
- **Un sac plus intelligent** : échantillonner en excès les coupes voisines d'une
  coupe déjà suspecte (curriculum), non tentée pour ne pas ajouter un paramètre de
  plus à cette même validation croisée.
- **Des features pré-entraînées**, écartées pour les mêmes raisons d'accès qu'au §4.1
  et au §4.5.
- **Une agrégation top-k plutôt que max pur**, moins sensible à une seule coupe bruitée
  que le max, sans diluer la minorité comme le ferait une moyenne.

**Le sac plus grand, essayé (2026-09-14) : pas d'effet.** `--bag-size 32` (contre 16),
`--batch-size` divisé par deux en compensation pour garder le même nombre de coupes
par pas (128) et donc le même budget mémoire GPU — le premier essai à budget non
compensé a saturé les 8 Go de la carte (7,6 Go, 100 % d'utilisation, ~30× plus lent)
et a été arrêté avant d'écrire quoi que ce soit. Une fois corrigé : ROC-AUC patient
**0,414 [0,334 – 0,497]**, un intervalle qui ne contient presque plus 0,5 que par sa
borne haute, sensibilité toujours nulle au seuil 0,5. Doubler le sac ne corrige donc
pas le manque de signal — au mieux ne change rien, au pire l'aggrave légèrement — ce
qui pointe vers la tâche elle-même (trame entière, encodeur de zéro, 56 cancers)
plutôt que vers le taux de sacs sans coupe peinte. Rapport versionné
(`models/examclf/cv_report_bag32.json`, `cv_predictions_bag32.csv`), même règle que
pour le reste : pas le checkpoint.

**Ce que ça ne bloque pas.** La démo sert toujours le modèle DCE-MRI pour la
localisation et le tabulaire Wisconsin pour l'étape 2 ; aucun des deux ne dépend de
cette tête. Son échec mesuré est la première fois que l'étape 1 a un chiffre du tout —
et c'est ce chiffre qui est publié, pas un chiffre plus flatteur obtenu en cherchant.
