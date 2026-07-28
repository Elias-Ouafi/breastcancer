# Démo — 3 étapes

> **Research Use Only — Not for diagnostic use.** Outil de recherche, pas un
> dispositif médical. Aucune décision clinique ne doit en dépendre.

## 1. Lancer l'app

```bash
.venv/Scripts/python.exe -m app.run_dce_mri
```

Un seul processus, en local uniquement (`127.0.0.1:5000`) — rien n'est exposé sur
le réseau.

## 2. Ouvrir

<http://127.0.0.1:5000>

## 3. Analyser un cas

Glisser-déposer (ou cliquer sur la zone de dépôt) l'un des fichiers de
`demo_cases/`, puis **Analyser l'examen** :

| Fichier | Patient | Coupe | IoU vérifié |
|---------|---------|:-----:|:-----------:|
| `demo_1_Breast_MRI_135.npz` | Breast_MRI_135 | 52 | 0,830 |
| `demo_2_Breast_MRI_105.npz` | Breast_MRI_105 | 62 | 0,738 |
| `demo_3_Breast_MRI_079.npz` | Breast_MRI_079 | 104 | 0,728 |

Résultat attendu : verdict + confiance, coupe annotée avec le cadre sur la zone
de réhaussement, et le détail technique (coupe, cadre, moteur) dépliable.

---

## À dire pendant la démo

Ces trois cas fonctionnent parce que **la coupe a été choisie à l'avance par un
humain**, pas par le modèle (clé `forced_slice` dans le `.npz`). Sur un volume
complet uploadé librement, la sélection automatique de coupe échoue encore
(vérifié 0/186 sur les patients de test) : le modèle segmente correctement une
lésion **quand on lui montre la bonne coupe**, il ne sait pas encore la trouver
seul. C'est un problème de *sélection*, pas de *segmentation* — détaillé dans
[plan.md](plan.md) §4.2.

Le Dice (~0,55) n'est pas comparable à la littérature (~0,80) : les masques
d'entraînement sont des **boîtes englobantes** TCIA, pas des contours experts.

## Si ça ne marche pas

| Symptôme | Cause probable | Correctif |
|----------|----------------|-----------|
| `FileNotFoundError` sur le checkpoint | `results_mri_p2_negfix/unet_best.pt` absent | Ré-entraîner (`python -m imaging.train`) ou démarrer en mode mock : `.venv/Scripts/python.exe -m app.server` |
| `demo_cases/` vide | Dossier non versionné (données patient) | Régénérer avec `TransformData.make_demo_case` |
| Port déjà utilisé | Une instance tourne déjà | `$env:MRI_APP_PORT = "5001"` avant de relancer |
| Aucune coupe annotée affichée | Le fichier n'est pas un `.npz` prétraité | Utiliser un fichier de `demo_cases/` |

**Repli si le live échoue** : garder une capture d'écran d'un résultat réussi.
