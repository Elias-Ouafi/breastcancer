# plan.md — décisions de conception et journal des mesures

> **Outil visé** : une seule étape — un examen de dépistage (DBT ou IRM) en entrée,
> dire s'il y a un cancer. **L'étape 2 (malin/bénin depuis une biopsie) a été retirée
> du dépôt le 2026-09-14** — voir "Retrait de l'étape 2" ci-dessous — pour concentrer
> l'effort sur l'étape 1, seule à ne pas avoir de modèle qui marche.
> **Modalités** : étape 1 sur **DBT / mammographie** (`Breast-Cancer-Screening-DBT`),
> décision du 2026-09-12, voir la section suivante. L'IRM multiphase (DCE-MRI, DICOM)
> reste la modalité de la brique de **localisation**, qui s'active après l'étape 1.
> **Cible** : démo / portfolio. **Pas d'usage clinique, pas de certification.**
> **Mention obligatoire, partout** : *Research Use Only — Not for diagnostic use*.

Ce document garde ce qui ne se déduit pas du code : la charte graphique appliquée à
l'app (Partie 3) et le journal daté de ce qui a été mesuré, y compris les échecs
(§4.1 à §4.3). Le reste — comment lancer la démo, où vivent les données, comment
tourne le pipeline — est dans [README.md](README.md), au plus près du code.

## Retrait de l'étape 2, focus exclusif sur l'étape 1 (2026-09-14)

Décision : retirer complètement du dépôt tout ce qui répondait à la question « cette
lésion est-elle bénigne ou maligne ? » — tabulaire (Wisconsin/Spark) et image
(`lesionclf`) — pour ne garder que la question « y a-t-il un cancer dans cet examen ? »
(étape 1). Pas une dépréciation en place : suppression, avec l'historique dans git.

**Pourquoi maintenant.** L'étape 2 n'a jamais manqué d'un modèle qui marche — le
tabulaire Wisconsin sert `/biopsie` à 99,9 % de ROC-AUC (avec une réserve connue et
non corrigée, la fuite de préprocessing avant split, restée ouverte depuis le
2026-08-18). L'étape 1, elle, vient d'accumuler son **troisième résultat négatif
d'affilée** — `lesionclf` (§4.5, AUC 0,513), `examclf` (§4.7, AUC 0,457), et
`examclf --bag-size 32` (§4.7, AUC 0,414, mesuré le 2026-09-14, un peu pire) — sans
qu'aucune piste n'ait encore été essayée à fond. Diviser l'attention entre une étape
qui fonctionne déjà et une étape qui accumule des échecs mesurés n'a plus de sens :
l'étape 1 est le seul goulot, elle mérite tout l'effort.

**Ce qui a été retiré** (fichiers, pas seulement du code mort) :

| Domaine | Fichiers |
|---|---|
| Pipeline tabulaire Wisconsin/Spark | `Main.py`, `AnalyzeData.py`, `train_tabular_model.py`, `tabular_export.py`, `Final_Report.md`, la fonction `extract_breast_cancer_wisconsin_diagnostic_data` d'`ExtractData.py`, les fonctions Spark/PCA de `TransformData.py` (`_get_spark`, `clean_data`, `apply_pca`, `transform_data`, etc.) |
| Route `/biopsie` | `app/templates/biopsy.html`, la section « Step 2 » d'`app/server.py`, la section tabulaire d'`inference.py` (`predict_tabular`, `predict_tabular_spark`, `load_tabular_scorer`) |
| Étape 2 en image | `imaging/lesionclf.py` |
| BreakHis (histopathologie, support image de l'étape 2) | `ExtractBreakHis.py` |
| Artefacts versionnés | `models/tabular/`, `models/lesionclf/`, `plots/model_comparison.png` |
| Tests | `tests/test_tabular_export.py`, `tests/test_lesionclf.py`, `tests/test_biopsy_route.py`, les deux tests biopsie de `test_result_page_claims.py` |
| Config / packaging | constantes Wisconsin/tabulaire de `config.py`, dépendances `pyspark`/`ucimlrepo`/`kaggle` de `pyproject.toml`, `COPY tabular_export.py`/`COPY models/tabular/` du `Dockerfile`, blocs correspondants de `.gitignore`/`.dockerignore` |

**Ce qui reste, vérifié partagé et donc gardé intégralement** : `TransformData.py`
(tout ce qui prétraite DBT/MRI), `validation.py`, `lineage.py`, `imaging/dataset.py`,
`imaging/metrics.py`, `imaging/sliceclf.py` (son encodeur sert aussi `examclf` et
`predict_dce_mri`), `app/predictor.py`, tout `inference.py` à partir de la section
imagerie. `reports/experiments/` (ablations U-Net DCE-MRI) n'a jamais contenu de
fichier tabulaire — seul `reports/model_results.csv`, jamais versionné, disparaît
avec son producteur.

**Vérifié après coup** : la suite passe (`ruff check .` propre), l'app Flask démarre et
rend `/` sans la carte « Étape 2 » ni erreur console, `/biopsie` répond 404. Les tests
retirés testaient exclusivement le code supprimé, aucune perte de couverture sur
l'étape 1 — `roc_auc`/`bootstrap_auc` restent couverts par `tests/test_examclf.py`.

> **Correction du 2026-09-15.** Cette ligne annonçait « 202 tests passent » et un
> `README.md` mis à jour « 232 → 202 ». Les deux chiffres sont faux : la suite en
> compte **176** aujourd'hui, et depuis ce jour-là des tests ont été *ajoutés*
> (`test_examclf.py` +128 lignes, `test_exambank.py` +12), jamais retirés — donc 202
> n'a jamais été mesuré, il a été estimé puis écrit comme une mesure. Ce n'est pas un
> accident isolé : le message du commit `acd4099`, deux jours plus tard, annonce « the
> local Windows run keeps all 209 » — un troisième chiffre, tout aussi impossible.
> C'est exactement
> la faute que l'incident de téléchargement du §4.10 décrit sous « compter des fichiers
> ne mesure pas un débit », commise sur un compte de tests. Les chiffres de cette ligne
> sont remplacés par la nature du constat, qui, elle, tient.

## Prochaines pistes pour l'étape 1 (recherche du 2026-09-14)

> **Verdict de la fin de journée, à lire avant la liste (§4.8 à §4.10).** Cette liste a
> été écrite le matin depuis la littérature ; l'après-midi l'a largement réfutée, et
> elle est gardée telle quelle parce que savoir *ce qu'on croyait* explique les mesures
> qui ont suivi. Ce qui tient désormais :
>
> - **Les pistes 1 et 3 sont mesurées et négatives** (§4.8, §4.10) : warm start
>   0,426 [0,338 – 0,513], score relatif 0,465 [0,382 – 0,551], contre 0,457 pour la
>   ligne de base. **La piste « Run B » (top-k seul) est annulée**, raison au §4.9.
> - **Les pistes 4 et 5** (échantillonnage de sac, pertes auxiliaires) tombent avec
>   elles : elles réarrangent une agrégation, alors que le §4.10 montre qu'il n'y a
>   rien à agréger — trois têtes entraînées font le même score qu'une statistique sans
>   modèle.
> - **La piste 2** (plus de patients) a abouti côté données — 86 patients cancer
>   disponibles au lieu de 56 — mais elle ne débloque rien seule.
> - **La piste 6** (auto-supervision) reste non testée et non prioritaire.
> - **Les pistes 7 et 8 remontent en tête**, et la 8 devient la suivante à essayer :
>   la question n'est pas la luminosité à une résolution ou à une autre — mesurée à
>   trois échelles, jusqu'au natif — mais la **forme**, qui demande des features
>   apprises sous la supervision la plus riche disponible, les **boîtes**.
>
> **Prochaine étape retenue** : reprendre le cadrage de Buda et al. — détection
> supervisée par boîtes en résolution native, puis agrégation en décision d'examen —
> sur les 86 patients cancer désormais téléchargés.

Revue de littérature ciblée sur le problème exact d'`examclf` (§4.7) : un label
d'examen, un signal utile sur <1 % des coupes, 56 patients cancer, une perte qui ne
descend dans aucun pli. Classée par effort attendu, pas par promesse — dans l'esprit
du reste de ce document, ce sont des pistes à mesurer, pas des solutions.

**1. Pré-entraîner l'encodeur d'`examclf` sur l'étiquette par coupe, au lieu de partir
de zéro à chaque pli.** Gain le moins cher de la liste : aucune donnée nouvelle, la
banque porte déjà ce qu'il faut. Piste la plus directement motivée par le diagnostic
du §4.7 — la perte n'y **descend dans aucun pli**, ce qui ressemble à un encodeur qui
n'apprend jamais à quoi ressemble une lésion depuis 107 sacs positifs dont le signal
fait ~5 coupes sur 68.

> **Correction du 2026-09-14, le jour même.** La première rédaction de cette piste
> disait « réutiliser le checkpoint `sliceclf` déjà entraîné » et citait son AUC
> intra-volume de 0,803 (§4.3). **C'était faux, et de la pire façon : la bonne
> justification pour le mauvais objet.** `models/sliceclf/sliceclf_best.pt` est
> entraîné sur `slice_bank_p2`, c'est-à-dire sur de la **DCE-MRI** — le 0,803
> appartient à une autre modalité que celle d'`examclf`. Le réutiliser tel quel serait
> un transfert inter-modalité, pas le « il sait déjà faire ça » que la phrase laissait
> entendre.

La version **dans le domaine** est meilleure et ne coûte pas plus cher : la banque
d'examens stocke déjà `has_lesion` par coupe, posé par les boîtes que
`preprocess_dbt_exams` peint. Mesuré sur la banque réelle avant d'écrire une ligne de
code :

| | |
|---|---:|
| Coupes de la banque | 59 529 |
| Coupes peintes (`has_lesion`) | **1 401** (2,35 %) |
| — dont dans un examen cancer | 583 |
| — dont dans un examen non-cancer (lésions bénignes) | 818 |
| Examens cancer ayant ≥ 1 coupe peinte | **107 / 107** |
| Profondeur moyenne d'un examen | 68,4 coupes |

1 401 étiquettes **localisées en profondeur**, contre 870 étiquettes d'examen : c'est
une supervision plus dense sur la seule question « à quoi ressemble une lésion ? », et
elle laisse à la phase MIL la question pour laquelle elle existe — cancer ou pas, à
laquelle une boîte peinte ne répond pas, puisque deux tiers des coupes peintes ici
sont des lésions **bénignes**.

**Implémenté le 2026-09-14** (`imaging/examclf.py`, `--warm-start-epochs`,
`WarmStartSliceDataset`, `warm_start_encoder`), avec deux garde-fous : les négatifs
sont rééchantillonnés à chaque époque (8 par positif, sinon une époque de warm start
coûterait quatre époques MIL pour une phase qui n'a qu'à initialiser un encodeur), et
surtout **seuls les examens d'entraînement du pli** sont lus. L'étiquette par coupe
vient des mêmes annotations que l'étiquette d'examen : pré-entraîner sur les coupes
d'un patient tenu à l'écart ferait fuiter sa réponse dans l'encodeur qui le note
ensuite — la faille exacte que la validation croisée existe pour empêcher,
réintroduite une couche plus bas. C'est la propriété qu'un test protège
(`test_the_warm_start_never_reads_a_held_out_patients_slice`).

**2. Pooler les trois splits BCS-DBT annotés.** Déjà chiffré dans ce document (ligne
P2 « reprendre l'étape 2 en version image », mais le même corpus alimente `examclf`) :
les boîtes du split test de la collection (`BCS-DBT-boxes-test`, 136 lignes, 60
patients, 30 cancers) n'ont jamais été utilisées ici. Les pooler porte le corpus
annoté de 141 à 201 patients, et surtout les cancers disponibles de 56 à 89 — tous
ceux que la collection contient. C'est le levier que le diagnostic du §4.7 nomme en
premier, pas encore tiré.

**3. Remplacer le max pur par un top-k moyenné.** `examclf` agrège un sac par
`logits.max(dim=1)` (`imaging/examclf.py:205`) : le gradient ne traverse que la
coupe la plus suspecte du sac, un choix documenté et défendable (une coupe saine ne
doit pas être diluée), mais aussi fragile à une coupe mal classée par excès de
confiance. Le top-k généralise le max (k=1) et la moyenne (k=N) : moyenner les k
scores les plus hauts est le compromis que plusieurs travaux de MIL médical
retiennent précisément pour les petites lésions rares, où le max seul est bruité par
construction (Avg-TopK, *Expert Systems with Applications* 2023 ; la même
justification revient dans DSMIL et les benchmarks de MIL en pathologie — la
prédiction instance-level d'une seule coupe surestime facilement, moyenner plusieurs
coupes hautes la stabilise). **Implémenté le 2026-09-14** (`--top-k`,
`examclf.bag_logit`), appliqué à l'identique à l'entraînement et à l'évaluation —
optimiser une moyenne de top-k puis noter au max pur ferait diverger les deux. `k=1`
reste le défaut et reproduit exactement le max mesuré jusqu'ici, ce qu'un test vérifie.

### Protocole de mesure de ces pistes, fixé avant de lancer quoi que ce soit

Le §4.5 et le §4.7 s'arrêtent tous les deux à une mesure, et disent pourquoi : régler
des hyper-paramètres contre cette même validation croisée jusqu'à ce que le chiffre
monte la viderait de son sens. Implémenter deux pistes d'un coup puis lancer cinq
variantes serait la version rapide de la même faute. Donc, **annoncé avant les
résultats** :

- **Run A** — warm start seul (`--warm-start-epochs 5 --top-k 1`), pour isoler la
  piste 1. Lancé le 2026-09-14 à 12h50.
- **Run B** — top-k seul (`--top-k 3 --warm-start-epochs 0`), pour isoler la piste 3.
- Une éventuelle **Run C** combinant les deux **seulement si** l'une des deux bouge
  l'AUC hors-pli au-delà de son IC actuel ; sinon on publie deux négatifs de plus et
  on passe à la piste 2 (plus de patients), qui est la seule que le diagnostic du §4.7
  désigne comme structurelle.
- Les chiffres fixés d'avance et non ajustés ensuite : 5 époques de warm start,
  8 négatifs par positif, k=3, 25 époques MIL, 5 plis, graine 42 — les mêmes que la
  ligne de base partout ailleurs.

**4. Échantillonnage de sac orienté plutôt qu'aléatoire.** Déjà une piste notée au
§4.7 (« sac plus intelligent »), maintenant recoupée par la littérature :
sur-échantillonner les coupes voisines d'une coupe déjà suspecte au sein d'un examen
positif (curriculum / hard-instance mining), au lieu du tirage uniforme actuel
(`ExamBagDataset.__getitem__`). Le facteur mesuré au §4.7 — un sac aléatoire de 16
coupes manque toutes les coupes peintes d'un examen cancer une fois sur quatre — est
exactement ce qu'un tirage orienté vers les coupes déjà suspectes réduirait.

**5. Pertes auxiliaires multi-tâches.** Prédire en plus du label cancer des
attributs disponibles sans coût (vue, latéralité, éventuellement densité) régularise
l'entraînement sur un signal principal rare — c'est une des techniques documentées
de la 1ʳᵉ place du concours Kaggle/RSNA de détection de cancer du sein en
mammographie 2023, sur un problème de prévalence comparable (~2 % d'images
positives ; ici ~13 % d'examens, mais 56 patients cancer seulement). Coût
d'implémentation modéré (têtes de sortie supplémentaires, `BCS-DBT-file-paths-*.csv`
porte déjà la vue), à essayer après les pistes 1-2 plutôt qu'avant.

**6. Pré-entraînement auto-supervisé sur les DBT normaux non annotés.** La
collection compte 4 581 patients normaux, dont 150 seulement sont téléchargés ici
(§« Ce que la cible coûte en données »). Un pré-entraînement contrastif (SimCLR,
DINO) sur ces volumes sans aucune étiquette, avant le fine-tuning MIL supervisé, est
documenté pour la mammographie par au moins un framework dédié (DITL, 2024) — mais
la preuve est mitigée : un benchmark 2026 sur la segmentation de densité mammaire
trouve un gain **négligeable ou négatif** du SSL générique face à un pré-entraînement
ImageNet nu, et seule une variante contrastive multi-vues fait mieux. À tester, pas à
supposer gagnant — le premier test bon marché est justement de comparer contre la
piste 1 (warm-start supervisé), qui pourrait suffire.

**7. Repenser l'architecture plutôt que le sac : un module global léger + un module
local haute capacité (style GMIC / 3D-GMIC).** La piste la plus proche du problème
exact posé ici, publiée par l'équipe NYU sur le screening mammographique : GMIC (MLMI
2019) puis sa variante 3D pour les volumes DBT/tomosynthèse (3D-GMIC, *IEEE TMI*
2023) évitent justement l'échantillonnage aléatoire de coupes qu'`examclf` pratique
— elles scorent le volume 3D entier via un module global bas-coût qui produit une
carte de saillance, puis affinent seulement les régions les plus suspectes avec un
module local haute capacité, entraînées avec le seul label d'examen. 3D-GMIC est
validé sur DBT externe (Duke) et atteint 0,831 d'AUC par sein sur la cohorte NYU —
mais entraîné sur 85 526 patients, 335× le corpus disponible ici. À considérer comme
direction à moyen terme (ré-architecturer, pas juste ré-entraîner), pas comme
prochain essai.

**8. Repère de référence, pas une piste à répliquer : le détecteur de Buda et al.**
Le papier qui a publié ce même jeu de données BCS-DBT (*JAMA Network Open* 2021)
fournit aussi son propre modèle — un DenseNet 2D par grille de cellules, perte focale
pour la rareté du positif, 65 % de sensibilité à 2 faux positifs par sein sur son
split test (460 études). C'est un cadrage différent (détection par boîte, pas
classification MIL par label d'examen) entraîné sur 4 838 études majoritairement
annotées — hors de portée du volume de boîtes disponible ici (299 lignes) — mais
c'est le chiffre auquel comparer n'importe quel futur modèle DBT de ce dépôt, et sa
perte focale est directement réutilisable dans le code existant (`imaging/metrics.py`
a déjà des pertes de segmentation dans cet esprit).

**Sources** : Buda et al., [*A Data Set and Deep Learning Algorithm for the Detection
of Masses and Architectural Distortions in Digital Breast Tomosynthesis Images*](https://arxiv.org/pdf/2011.07995),
JAMA Netw Open 2021 · Shen et al., [*GMIC*](https://arxiv.org/pdf/2002.07613), MLMI
2019 · [*3D-GMIC*](https://arxiv.org/abs/2210.08645v1), IEEE TMI 2023 · dangnh0611,
[1ʳᵉ place, RSNA Screening Mammography Breast Cancer Detection AI Challenge](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)
2023 · [*Avg-TopK: A new pooling method for CNNs*](https://www.sciencedirect.com/science/article/abs/pii/S0957417423003937),
Expert Systems with Applications 2023.

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
| P1 | Pooler les trois splits BCS-DBT annotés pour `examclf` | Porte le corpus annoté de 141 à 201 patients, les cancers de 56 à 89 — tous ceux de la collection. Chiffré, pas fait ; voir "Prochaines pistes pour l'étape 1" |
| P1 | Initialiser l'encodeur d'`examclf` depuis le checkpoint `sliceclf` | Changement bon marché (pas de nouvelle donnée) motivé par le diagnostic du §4.7 (la perte ne descend dans aucun pli) ; voir "Prochaines pistes pour l'étape 1" |
| P2 | Top-k pooling et échantillonnage de sac orienté pour `examclf` | Deux pistes bon marché supplémentaires, voir "Prochaines pistes pour l'étape 1" |
| P3 | Bug NaN fp16 non résolu | La divergence (§4.2, repoussée époque 11 → 15) est localisée dans le forward pass et corrigée, ou documentée comme acceptée. Rétrogradé de P2 : le U-Net DCE-MRI quitte le chemin critique de l'étape 1. Le checkpoint servi reste un instantané pré-divergence (époque ≤ 14 sur 30) |
| P3 | Retirer le code mort | **Fait le 2026-09-15** : supprimé, pas réécrit — `app/run_unet.py`, `DbtUNetPredictor`, la branche `unet` de `get_predictor`, `inference.predict_dbt`, `load_dbt_dicom`, `config.DBT_UNET_CKPT`, et la branche `raw_loader` de `_load_volume_and_offset` devenue inatteignable. Réécrire « pour la nouvelle tête DBT » n'avait plus d'objet : le §4.10 mesure que la suite du côté DBT n'est pas un U-Net de segmentation. `imaging.train` écrit toujours un checkpoint DBT si on en entraîne un ; le resservir redevient un acte délibéré |
| P3 | Réparer le paquet | **Fait le 2026-09-15** : `logging_setup`, `validation` et `lineage` ajoutés à `py-modules`. Vérifié sur la roue construite (`pip wheel . --no-deps`) et non sur une promesse — les trois fichiers y sont désormais, ils n'y étaient pas |
| P3 | Exécuter le lineage sur le corpus | **Fait — la ligne était périmée, corrigée le 2026-09-15.** Deux manifestes existent et viennent de vraies passes : `dbt/manifest.json` (260 cas, révision `c22bc81-dirty`, 2026-09-13) et `dbt_exams/manifest.json` (870 cas, révision `060b40f`, 0 avertissement de validation). Reste ouvert, plus petit : `dce_mri_p2/` n'en a pas (le corpus précède `lineage.py`), et le manifeste DBT porte `output_dir: data/preprocessed_data/dbt_join` alors qu'il est lu depuis `dbt/` — le dossier a été renommé après la passe, donc ce champ ne dit pas où le manifeste se trouve |
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

### Relevés et corrigés le 2026-09-15

Même principe : gardés une fois corrigés. Tous relevés en relisant la doc **contre le
dépôt qui tourne**, pas contre la doc.

| Constat | Où | État |
|---|---|---|
| `README.md` annonçait « 202 tests, ~25 s » ; il y en a **176**, en ~38 s. Le chiffre ne pouvait pas être juste : depuis la ligne qui l'a écrit (2026-09-14), des tests ont été **ajoutés** et aucun retiré | `README.md` §Development, et la ligne « 202 tests passent » du §Retrait de l'étape 2 | Corrigé le 2026-09-15 |
| `app/__init__.py` documentait le vrai backend comme `MRI_APP_BACKEND=unet` — un backend qui ne démarrait pas, puis qui n'existe plus | `app/__init__.py` | Corrigé : `dce_mri` |
| `inference.predict_dce_mri` documentait son checkpoint comme `results_mri_p2/unet_best.pt` ; `config.DCE_MRI_UNET_CKPT` dit `models/dce_mri_p2_negfix/unet_best.pt` | `inference.py` docstring | Corrigé le 2026-09-15 |
| La table « Ce qui reste ouvert » annonçait qu'aucun `manifest.json` n'existait sous `data/preprocessed_data/` ; deux existent depuis le 2026-09-13 | `plan.md` §Ce qui reste ouvert | Corrigé le 2026-09-15 — la doc était en retard sur le code, le sens inhabituel de l'écart |
| `app/README.md` listait trois backends servables ; le troisième (`unet`) nommait un checkpoint inexistant | `app/README.md` | Corrigé avec le retrait du code mort |
| `README.md` annonçait `~80 GB` de DICOM sous `data/raw_data/tcia/` ; le dossier en fait **138 Go**. L'écart est le coût du split test téléchargé le 2026-09-14 (§4.10) et des 150 patients normaux, jamais reporté sur cette ligne | `README.md` §Where things live | Corrigé le 2026-09-15, avec la date de la mesure dans le texte — un volume qui grossit à chaque téléchargement se périme, le dater dit quand le recompter |

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
| Parité annoncée « sur les 569 lignes, écart max 1 × 10⁻¹⁵ » ; le test compare **5 lignes** à 1e-9, et le dit dans son propre commentaire | `plan.md`, docstring `inference.predict_tabular` vs `tests/test_tabular_export.py` | Caduc le 2026-09-14 : code retiré, voir "Retrait de l'étape 2" |
| Les métriques tabulaires renvoient à `reports/model_results.csv`, **absent du disque** — comme `pca_info.csv`, `feature_contributions.csv`, `scree_plot.png` | `Final_Report.md` | Caduc le 2026-09-14 : `Final_Report.md` retiré |
| « 87 tests, ~7 s » ; il y en a **174**, tous passants (116 à l'audit, plus 9 pour le P0, 16 pour la colonne `Class`, 6 pour le split stratifié, 7 pour l'appariement et 20 pour l'étape 2 en image) | `README.md` §Development | Corrigé le 2026-09-13 : **198**, après les tests de la jointure et des labels |
| « 0,76 s par volume, 4,5 ms par coupe » ; l'artefact dit **0,825 s** et **4,83 ms** | `plan.md` §4.3 et `DEMO.md` vs `eval_report.json` | Ouvert |
| « temps de calcul ~110 ms » ; mesuré 69-73 ms à chaud, 585 ms au premier appel | `README.md`, `DEMO.md` | Ouvert |
| Checkpoint par défaut documenté `results_mri_p2/unet_best.pt` ; c'est `models/dce_mri_p2_negfix/unet_best.pt` | docstring `inference.predict_dce_mri` | Ouvert |
| Backend `unet` présenté comme disponible ; son checkpoint n'existe plus | `app/README.md`, `app/predictor.py` | Ouvert |
| Logs annonçant `data/transformed_data.csv`, `data/pca_info.csv`, `data/scree_plot.png` ; le code écrit dans `reports/` et `plots/` | `TransformData.transform_data` | Caduc le 2026-09-14 : fonction retirée |
| IC 95 % top-1 `[25,0 – 60,7]` : aucun code ni artefact versionné ne la produit | `plan.md` §4.3 | Ouvert |
| Fuite de préprocessing : imputation, scaler et PCA ajustés sur les 569 lignes **avant** le `randomSplit`, donc les 97,67 % / 99,89 % sont optimistes | `TransformData.transform_data` → `AnalyzeData.prepare_data` | Caduc le 2026-09-14 : les deux fonctions retirées |
| Étape 2 annoncée « livrée » ; la branche `ameliore-le-mvp` est locale, absente d'`origin` | `plan.md` §Livrés | Caduc le 2026-09-14 : étape 2 retirée du dépôt |
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

> **2026-09-15** : `load_dbt_dicom` a été supprimée avec le reste du chemin
> d'inférence DBT (voir §Écarts relevés le 2026-09-15). La contrainte, elle, ne
> disparaît pas — elle change seulement d'adresse : le jour où un chemin d'inférence
> DBT est réécrit, il doit appliquer la règle de latéralité de
> `TransformData.image_laterality`, sinon une moitié des examens arrivera dans le
> mauvais repère. C'est noté ici plutôt que résolu.

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
ligne du §4.4 ; la jointure ne la traite pas. *(La fonction est supprimée depuis le
2026-09-15 ; la contrainte reste, reportée sur le futur chemin d'inférence DBT — voir
la note du §4.4.)*

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
localisation ; elle ne dépend pas de cette tête. Son échec mesuré est la première fois
que l'étape 1 a un chiffre du tout — et c'est ce chiffre qui est publié, pas un chiffre
plus flatteur obtenu en cherchant. *(Mise à jour du 2026-09-14 : la deuxième moitié de
cette phrase, « le tabulaire Wisconsin pour l'étape 2 », ne vaut plus — l'étape 2 est
retirée du dépôt, voir la section en tête de document.)*

### 4.8 Run A — le warm start ne marche pas, et son diagnostic déplace le problème (2026-09-14)

Première des deux mesures pré-enregistrées plus haut (« Protocole de mesure de ces
pistes »). Piste 1 isolée : `--warm-start-epochs 5 --top-k 1`, tout le reste identique
à la ligne de base du §4.7. Rapport versionné dans
`models/examclf/warmstart5/cv_report.json`.

**Le résultat au niveau examen, celui qui était annoncé d'avance.**

| Mesure | Ligne de base (§4.7) | Run A (warm start) |
|---|---:|---:|
| ROC-AUC patient | 0,457 [0,369 – 0,544] | **0,426 [0,338 – 0,513]** |
| Sensibilité à 0,5 | 0,0 % | **0,0 %** |
| Exactitude | 79,4 % | 79,4 % |
| « Toujours pas de cancer » | 79,4 % | 79,4 % |

Aucune amélioration ; l'IC contient toujours 0,5 par sa borne haute, et le modèle ne
prédit toujours aucun cancer. **Négatif, publié comme tel.**

**Mais le diagnostic ajouté pour ce run vaut plus que la mesure elle-même.** Il pose au
modèle la question *par coupe* sur des coupes tenues à l'écart, là où `has_lesion`
fournit une vérité terrain — une fois après le warm start, une fois après la phase MIL.
Jamais utilisé pour sélectionner quoi que ce soit.

| Pli | AUC par coupe après warm start | après MIL |
|---|---:|---:|
| 0 | 0,514 | 0,464 |
| 1 | 0,517 | 0,537 |
| 2 | 0,532 | 0,557 |
| 3 | 0,433 | 0,457 |
| 4 | 0,516 | 0,535 |
| **Moyenne** | **0,502** | **0,510** |

**Le hasard, dans les cinq plis, avant comme après.** Et la perte du warm start le
confirme par un autre chemin : elle converge à **1,2364**, contre **1,2323** pour le
meilleur prédicteur *constant* possible à cet équilibre de classes (calculé, pas
estimé : avec `neg_per_pos=8` et `pos_weight=8` les deux classes se compensent
exactement, donc l'optimum constant est p = 0,5). L'encodeur ne trouve rien de mieux
qu'ignorer l'image.

**Ce que ça change dans la lecture du §4.7.** Le diagnostic y disait « la tâche est plus
dure que celle de `lesionclf` », en désignant l'objectif MIL et la rareté des sacs
positifs. C'était incomplet. Avec une étiquette **10 fois plus dense et localisée en
profondeur** (1 401 coupes peintes contre 870 étiquettes d'examen), le même encodeur
n'apprend toujours rien. Le goulot n'est donc **pas l'agrégation** — ce qui rétrograde
d'un coup les pistes 3, 4 et 5 (top-k, échantillonnage de sac, pertes auxiliaires) :
elles réarrangent toutes la façon d'agréger des features qui ne portent aucun signal.
Un top-k sur du bruit reste du bruit.

**La résolution, suspect n°1, est largement disculpée.** Mesuré sur les 299 boîtes
annotées, en propageant la géométrie du prétraitement (natif 2457 lignes → 384 dans
`dbt_exams` → 224 dans la banque, facteur 0,091) :

| | Médiane | p10 | Part de la trame |
|---|---:|---:|---:|
| Boîte en natif | 212 × 201 px | — | 0,91 % |
| à 384 (`dbt_exams`) | 33 × 31 px | 15 px | 0,91 % |
| à 224 (banque MIL) | **19 × 18 px** | 8,9 px | 0,91 % |

13 % des boîtes passent sous 8 px de côté à 224, **aucune** sous 4 px. Une lésion
médiane occupe donc ~19 px dans une trame de 224 : petit, mais pas au point d'expliquer
une AUC de 0,50 sur une tâche de présence.

**Le suspect qui reste, et c'est un défaut de définition de cible, pas de modèle.**
`preprocess_dbt_exams` peint `slice_margin=2`, soit 5 coupes autour de l'unique coupe
que BCS-DBT annote. Une masse en tomosynthèse reste visible bien au-delà : les coupes
juste en dehors de la bande peinte montrent **la même lésion** tout en portant
l'étiquette inverse. On demande à l'encodeur de séparer z+2 de z+3, deux images quasi
identiques. Mesuré sur la banque : une bande d'exclusion de ±10 coupes écarte 5 275 des
17 104 négatifs des examens annotés (30,8 %), et il en reste ~53 000 ailleurs — donc
l'hypothèse est testable sans manquer de négatifs.

**Test en cours** (`--warm-start-exclude-band`, implémenté et testé le même jour) : une
seule séparation 80/20 par patient, bande ±10, 15 époques au lieu de 5. Les deux
explications candidates — bruit d'étiquette en profondeur, budget d'entraînement — y
sont relâchées **ensemble et à dessein** : la question posée est binaire (« ce signal
est-il apprenable ici, oui ou non ? »), pas « laquelle des deux mérite le crédit ». Si
l'AUC décolle, on isolera ensuite ; si elle reste à 0,50, les deux hypothèses tombent
ensemble et la question suivante devient la capacité de l'encodeur et la géométrie du
prétraitement, pas l'agrégation.

### 4.9 Le signal est relatif à l'examen — et c'est pourquoi rien ne marchait (2026-09-14)

Le test annoncé au §4.8 a répondu, puis quatre mesures sur les pixels ont trouvé la
cause. C'est la première chose positive mesurée de toute cette chaîne, et elle désigne
un **défaut de conception de la tête**, pas un manque de données.

**Le test discriminant : les deux hypothèses tombent ensemble.** Bande d'exclusion
±10 coupes, 15 époques au lieu de 5, une séparation 80/20 par patient (684 examens
d'entraînement, 186 tenus à l'écart) :

| Époque | 1 | 3 | 5 | 8 | 12 | 15 |
|---|---:|---:|---:|---:|---:|---:|
| Perte d'entraînement | 1,2462 | 1,2356 | 1,2342 | **1,2323** | 1,2330 | 1,2327 |
| AUC par coupe hors-pli | 0,517 | 0,450 | 0,532 | 0,537 | 0,482 | 0,529 |

Plancher du prédicteur constant : **1,2323**, atteint exactement à l'époque 8. AUC
finale sur les seules coupes non ambiguës (211 peintes contre 11 290 éloignées) :
**0,532**. Le modèle **n'arrive pas à sur-ajuster son propre jeu d'entraînement** — ce
n'est pas un défaut de généralisation, c'est une incapacité à optimiser. Ni le bruit
d'étiquette en profondeur ni le budget d'époques n'expliquent donc quoi que ce soit.

**Quatre mesures sur les pixels, dont une qui ne prouvait rien.**

1. *Les masques sont-ils sur du tissu ?* Oui : intensité moyenne +1,529 dans la boîte
   contre −0,013 ailleurs, 0 boîte sur 117 plus sombre que sa coupe. **Mais cette
   comparaison était vide** — « ailleurs » contient 54 % d'air, donc elle dit seulement
   que la boîte est dans le sein. Le §4.4 posait ce test pour détecter des masques sur
   du fond ; le reprendre tel quel ici répondait à une autre question que la mienne.
2. *La lésion se distingue-t-elle du **tissu normal de sa propre coupe** ?* Oui, et
   nettement : +1,535 contre +0,725, soit un **d de Cohen de 1,15**, et la lésion est
   plus brillante que son tissu dans **117 examens sur 117**. Le signal existe, il est
   large, et il est visible par une simple intensité.
3. *Alors une statistique triviale devrait trouver la coupe peinte.* Elle la trouve —
   mais seulement **à l'intérieur d'un examen** :

   | Statistique par coupe | AUC intra-examen |
   |---|---:|
   | maximum | 0,501 |
   | **99ᵉ percentile** | **0,732** |
   | moyenne des 100 plus hauts | 0,534 |
   | aire au-dessus de 1,5 | 0,637 |

4. *Et mise en commun entre examens ?* **C'est là que tout s'effondre**, et c'est le
   chiffre central de cette section :

   | 99ᵉ percentile, 8 387 coupes de 117 examens annotés | AUC |
   |---|---:|
   | **mise en commun entre examens** | **0,532** |
   | **normalisée par examen** | **0,736** |

**Conclusion.** Le pouvoir discriminant est **entièrement relatif à l'examen**. Le
niveau absolu varie tellement d'un patient à l'autre qu'il noie une différence pourtant
large (d = 1,15). Or `examclf` produit un score **absolu** par coupe, en prend le max,
et compare ces maxima **entre patients** pour tracer une ROC patient. On lui demande
exactement ce que la donnée ne permet pas. 0,50 par coupe et 0,46 par examen ne sont
donc pas deux échecs, c'en est un seul, vu à deux niveaux.

**Une correction évidente, et réfutée par la mesure.** Si le niveau absolu varie, une
normalisation d'intensité mieux posée devrait le corriger : z-normaliser sur le
**tissu seul** plutôt que sur un volume à 54 % d'air. Mesuré avant d'écrire la moindre
ligne de production : AUC mise en commun **0,520**, contre 0,532 pour la normalisation
actuelle. **Aucun gain — hypothèse abandonnée.** Ce que le §4.6 disait de l'appariement
vaut ici : une correction plausible qui ne mesure pas mieux n'est pas une correction.

**Ce qui est implémenté à la place** (`--bag-relative`, `examclf.bag_logit`) : le score
d'un sac devient l'écart entre sa coupe la plus suspecte et **sa propre médiane**, à
l'entraînement comme à l'évaluation. La médiane plutôt que la moyenne, un sac étant
surtout fait de coupes banales. Deux conséquences à assumer : le score n'est plus une
probabilité calibrée mais un **score de rang** — le seuil 0,5 n'y veut plus rien dire,
et c'est de toute façon le P1 « choisir le point de fonctionnement sur la validation »
qui doit le fixer ; et `torch.median` renvoie la valeur inférieure des deux centrales
sur un sac de taille paire, ce qu'un test épingle parce que ça décale tous les scores.

**Sur le protocole.** Cette piste ne vient pas de la liste pré-enregistrée, et il faut
dire pourquoi ce n'est pas du réglage déguisé : elle sort de mesures faites **sur les
pixels et sur des statistiques écrites à la main**, qui n'ont jamais touché ni les plis
de la validation croisée ni un modèle entraîné. Le mécanisme a été identifié d'abord,
mesuré ensuite. **Run B (top-k seul) est abandonnée** plutôt que lancée : le §4.8 a
montré qu'il n'y a rien à agréger tant que les scores ne sont pas comparables, et
dépenser deux heures de GPU pour le confirmer n'apprendrait rien — c'est une
annulation motivée, écrite ici pour qu'elle ne passe pas pour un résultat non publié.

### 4.10 Le score relatif ne marche pas non plus — et une ligne de numpy bat le CNN (2026-09-14)

**La mesure.** `--bag-relative`, tout le reste identique à la ligne de base
(`models/examclf/relative/cv_report.json`) : ROC-AUC patient **0,465 [0,382 – 0,551]**,
contre 0,457 [0,369 – 0,544]. Inchangé. La perte d'entraînement descend un peu plus bas
qu'avant (1,21-1,22 contre ~1,35), donc l'objectif relatif est marginalement plus facile
à optimiser, mais **rien n'arrive jusqu'à l'AUC hors-pli**. Au seuil 0,5 : sensibilité
1,0, spécificité 0,0 — exactement ce que le §4.9 annonçait pour un score de rang, où ce
seuil ne veut rien dire.

**Le tableau qui manquait.** Le §4.9 comparait le CNN *mis en commun* (0,50) au p99
*intra-examen* (0,73) : deux métriques différentes, donc une comparaison qui ne valait
rien. Calculées sur le même modèle (checkpoint du pli 0) et les mêmes examens tenus à
l'écart :

| | AUC intra-examen | AUC mise en commun |
|---|---:|---:|
| **CNN entraîné** | **0,592** | 0,464 |
| **99ᵉ percentile, une ligne de numpy** | **0,732** | 0,532 |

Le CNN apprend donc *quelque chose* (0,592 > 0,5), et **se fait battre de 0,14 par une
statistique triviale** sur la tâche pour laquelle il est entraîné. Un modèle qui
n'atteint pas ce qu'une ligne de code capture ne souffre pas d'un manque de données.

**Et la mesure qui referme tout : la même statistique, sur la vraie question.** Aucun
modèle, aucun entraînement, 272 patients, 56 cancers, score patient = max sur ses
examens :

| Statistique par examen | AUC patient |
|---|---:|
| max absolu du p99 | 0,433 [0,350 – 0,523] |
| **max relatif (z par examen)** | **0,366 [0,292 – 0,444]** |
| max − médiane | 0,442 [0,365 – 0,526] |
| moyenne des 3 plus hauts, relatifs | 0,371 [0,296 – 0,447] |

**Rien ne bat le hasard, et deux sont significativement en dessous** — leur IC entier
est sous 0,5, donc la statistique est *anti*-corrélée au cancer. À rapprocher des
0,457 / 0,426 / 0,465 des trois têtes entraînées : elles font toutes, à peu près, ce
que fait une statistique sans modèle.

**Ce que ça veut dire, et c'est la leçon de la journée.** Le §4.9 avait trouvé un signal
intra-examen à 0,736 et j'en avais tiré un espoir mal placé. **Trouver la lésion annotée
n'est pas détecter un cancer** : deux tiers des coupes peintes de ce corpus sont des
lésions *bénignes*, donc ce que le p99 trie, c'est « il se passe quelque chose ici », et
« quelque chose » est plus souvent bénin que malin. Le signal mesuré et la cible visée
ne sont pas la même quantité, et aucun réglage d'agrégation ne transforme l'un en
l'autre.

**Conclusion provisoire, formulée pour être réfutable.** Au niveau de prétraitement
actuel — trame entière, 2457 lignes ramenées à 384 puis 224, z-normalisation par
volume — **il n'y a pas de signal de cancer au niveau examen mesurable**, ni par un CNN
entraîné de trois façons différentes, ni par des statistiques d'intensité écrites à la
main. Ce n'est pas un manque de patients : 56 cancers suffiraient largement à détecter
un effet de la taille de celui que le §4.9 mesure sur la lésion (d = 1,15). C'est la
représentation qui ne porte pas la question.

Ce qui recadre la piste 2 (plus de patients) : elle reste nécessaire, elle n'est plus
suffisante — et elle ne devrait pas être la prochaine dépense. Ce que la comparaison
avec Buda et al. suggérait déjà, sans que j'en tire les conséquences : leur 65 % de
sensibilité à 2 FP/sein est obtenu **en résolution native**, par un détecteur à grille
de cellules 96×96 supervisé par les boîtes — pas par une classification faiblement
supervisée d'une trame réduite à 224 px. La différence n'est pas l'architecture, c'est
ce qu'on donne à voir au modèle.

**Et ce n'est pas la banque qui perd le signal.** Vérifié, parce que c'était
l'explication la plus confortable : les mêmes statistiques calculées sur le corpus
stocké à **384 px**, avant le sous-échantillonnage de la banque, donnent 0,431 / 0,359 /
0,469 — identiques aux 0,433 / 0,366 mesurés à 224. La perte, s'il y en a une, se
produit plus tôt : au passage 2457 → 384, ou dans le cadrage lui-même.

**Prochaine dépense, dans l'ordre.** (1) Refaire cette mesure sans modèle **en
résolution native**, sur un échantillon de patients : si une statistique d'intensité y
sépare mieux, le prétraitement est le coupable et la correction est une géométrie, pas
un modèle. (2) Si elle ne sépare pas mieux non plus, passer au cadrage de Buda et al. —
détection supervisée par les boîtes, en natif, puis agrégation en décision d'examen —
plutôt que de continuer à entraîner une classification faiblement supervisée sur une
représentation dont on aura alors mesuré trois fois qu'elle ne porte pas la question.

**Fait le jour même, et c'est (2) qui l'emporte.** Mesure (1) sur le DICOM brut,
**60 patients cancer contre 60 normaux** tirés au hasard parmi ceux présents sur
disque, aucun modèle, aucun entraînement. La statistique n'est plus un percentile mais
un détecteur de tache grossier : le maximum, sur des blocs de **96 px** — la cellule de
grille de Buda et al. sur ce jeu de données précis, pas un choix arbitraire — de
l'écart entre la moyenne du bloc et la médiane du tissu de sa coupe, en écarts-types du
tissu.

| Résolution | AUC patient, sans modèle |
|---|---:|
| **native (2457 lignes, blocs 96 px)** | **0,451 [0,352 – 0,556]** |
| 384 px (corpus stocké) | 0,431 [0,347 – 0,522] |
| 224 px (banque MIL) | 0,433 [0,350 – 0,523] |

Moyennes : cancers **3,493 ± 0,835**, normaux **3,717 ± 0,917** — les examens cancer
sont même très légèrement *moins* contrastés que les normaux, cohérent avec
l'anti-corrélation relevée plus haut.

**L'hypothèse de la géométrie est donc réfutée.** Ce n'est pas le prétraitement qui a
détruit le signal : en résolution native, avec la fenêtre d'analyse du détecteur publié
sur ce jeu de données, une statistique d'intensité ne sépare toujours pas un examen
cancer d'un examen normal. Ce qui distingue les deux n'est pas une question de
*luminosité* à une échelle ou à une autre — c'est de la **forme** (spiculation,
distorsion architecturale), et une forme ne se lit pas dans une moyenne de bloc. Il
faut des features apprises, et pour les apprendre sur 86 patients cancer il faut la
supervision la plus riche disponible : les **boîtes**, pas une étiquette d'examen.

C'est exactement ce que fait Buda et al. (65 % de sensibilité à 2 FP/sein), et c'est ce
qui reste à essayer. Noter ce que cela implique pour l'ordre des pistes : la piste 2
(plus de patients) vient d'aboutir côté données — les 60 patients du split test sont
téléchargés, 121/121 séries, **30 patients cancer complets**, ce qui porte le corpus
annoté à **86 patients cancer** — mais elle ne sert à rien tant que la tâche reste une
classification faiblement supervisée d'une trame entière.

#### Incident de téléchargement (2026-09-14)

`tcia_utils.nbia` n'impose **aucun timeout** : le téléchargement des patients du split
test s'est figé sur une seule requête à 13 h 51 et y est resté **2 h 30**, sans erreur,
sans log, le processus vivant. Deux conséquences. La première est opérationnelle : il
faut relancer (les séries déjà acquises sont sautées, donc rien n'est perdu). La
seconde est une leçon sur la façon de rendre compte — j'ai rapporté à 14 h 33 un débit
de « ~1 min par série » calculé en divisant 73 séries par le temps écoulé, alors que le
processus était arrêté depuis 40 minutes. **Compter des fichiers ne mesure pas un
débit** ; il faut lire l'horodatage de la dernière ligne de log, ce que la suite de ce
document fera.
