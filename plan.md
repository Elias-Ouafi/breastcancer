# plan.md — décisions de conception et journal des mesures

> **Modalité** : IRM mammaire multiphase (DCE-MRI, DICOM).
> **Outil visé** : deux étapes — (1) une IRM en entrée, dire s'il y a un cancer ;
> (2) les retours de la biopsie en entrée, dire si c'est malin ou bénin.
> **Cible** : démo / portfolio. **Pas d'usage clinique, pas de certification.**
> **Mention obligatoire, partout** : *Research Use Only — Not for diagnostic use*.

Ce document garde ce qui ne se déduit pas du code : la charte graphique appliquée à
l'app (Partie 3) et le journal daté de ce qui a été mesuré, y compris les échecs
(§4.1 à §4.3). Le reste — comment lancer la démo, où vivent les données, comment
tourne le pipeline — est dans [README.md](README.md), au plus près du code.

## Où en est le projet (2026-08-18)

> **Relu contre la cible produit**, énoncée ici pour la première fois : (1) une IRM en
> entrée, dire s'il y a un cancer ; (2) les retours de la biopsie en entrée, dire si
> c'est malin ou bénin. Les mesures ci-dessous étaient justes ; c'est leur lecture qui
> change. Aucune ne répond à l'étape 1, et l'étape 2 ne sort pas du disque.

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

**Étape 2 — faite, mesurée, débranchée.** Wisconsin Diagnostic *est* l'étape 2 : 30
features morphologiques mesurées sur une cytoponction, 569 cas, label M/B, ROC-AUC
99,89 % (`Final_Report.md`). `train_tabular_model.py` persiste un `PipelineModel` unique
dans `models/tabular/`, et `inference.predict_tabular` sait scorer 30 features brutes.
Mais `app/` n'appelle jamais `predict_tabular` : pas de route, pas de formulaire, pas de
template. La moitié la plus performante du projet est un moteur sans pédale.

**BreakHis est une branche morte.** `ExtractBreakHis.py` télécharge et extrait ;
`BREAKHIS_DIR` n'apparaît nulle part ailleurs que dans `config.py`. Zéro modèle, zéro
test, zéro métrique. C'est pourtant le support image de l'étape 2 — les lames issues de
la biopsie. À assumer comme pendant image du Wisconsin, ou à retirer.

**Le dépôt, lui, tient.** Données en couches sous `data/` (raw → preprocessed →
curated), chemins centralisés dans `config.py`, checkpoints dans `models/`, pipeline
exécutable et reprenable via `python -m pipelines.dce_mri`, **87 tests** (~7 s, sans GPU
ni dataset) et ruff en CI. Docker, validation de schéma et manifeste de lineage sont
écrits, testés et commités le 2026-08-18.

## Ce qui reste ouvert

Ordonné par ce qui rapproche de la cible en deux étapes, pas par facilité.

| Priorité | Tâche | Critère de « fait » |
|---|---|---|
| P0 | Commiter Docker + validation + lineage | Les trois chantiers ci-dessous sont sur `main` ; un relecteur les voit |
| P1 | Brancher l'étape 2 dans l'app | Une route et un formulaire 30 champs (ou upload CSV) appellent `inference.predict_tabular` et affichent malin/bénin avec sa probabilité |
| P1 | Acquérir des examens négatifs | Le corpus contient des patients sans lésion ; un split conserve la proportion |
| P1 | Tête de détection au niveau volume | Sensibilité, **spécificité** et ROC-AUC patient publiées avec IC, comme §4.3 l'a fait pour la localisation |
| P2 | Trancher le sort de BreakHis | Un modèle bénin/malin entraîné et mesuré, ou le script et `BREAKHIS_DIR` supprimés |
| P2 | Trancher le sort du pipeline tabulaire Spark | Assumé et documenté comme démo Spark, ou retiré — aujourd'hui il impose une JVM à tout le dépôt pour 569 lignes |
| P2 | Bug NaN fp16 non résolu | La divergence (§4.2, repoussée époque 11 → 15) est localisée dans le forward pass et corrigée, ou documentée comme acceptée |
| P3 | Registre de traitement RGPD | Une page : base légale, nature des données, finalité, conservation, sécurité |
| P3 | Nom de produit + logo | Choisi et intégré au header de l'app |

**Livrés** (branche `amelioration-docker-preprocess-env`, commités le 2026-08-18) :

| Tâche | Où | Ce qui la rend faite |
|---|---|---|
| Packaging Docker | `Dockerfile`, `docker-compose.yml` | `docker compose up --build` rejoue la démo ; image étroite (ni Spark, ni JVM, ni ITK), torch CPU, `read_only`, port publié sur `127.0.0.1` seulement, healthcheck sur `run_demo.py --check` |
| Validation de schéma | `validation.py`, branchée dans `save_preprocessed` | Dimensions, dtype, finitude et binarité du masque vérifiés au point unique d'écriture ; le seuil `MIN_IN_PLANE = 128` est calibré sur l'incident de crop du §4.2, pas deviné. 13 tests |
| Manifeste de lineage | `lineage.py` | `manifest.json` par dossier prétraité : commit (suffixé `-dirty`), source, paramètres, stats par cas. Écrit en dernier — son absence signale une run interrompue. 8 tests |

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
