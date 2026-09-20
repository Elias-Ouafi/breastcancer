"""Ce que l'image contiendrait suffit-il à faire tourner la démo ?

Le build lui-même demande un daemon Docker, absent de cette machine (§4.17). Deux
propriétés s'en détachent pourtant et se vérifient sans daemon — ce sont aussi les
deux qui cassent en silence quand on déplace un fichier ou qu'on ajoute un import :

* **La liste des `COPY` est complète.** Un module oublié ne se voit pas ici, où tout
  le dépôt est sur le disque ; il se voit dans le conteneur, au premier clic.
* **Les paquets déclarés suffisent.** `requirements-demo.txt` en installe quatre. Un
  cinquième import qui se glisse dans le chemin de la démo passe inaperçu sur une
  machine de développement, où les 68 paquets du pipeline sont là.

Les deux sont mesurées dans **un seul sous-processus** : il reconstruit l'arbre que
les `COPY` produiraient, s'y installe, refuse à l'import tout paquet tiers absent de
l'image, puis fait le clic de démo. Un sous-processus parce que le processus pytest a
déjà importé `config` depuis le dépôt — la mesure porterait alors sur le mauvais
arbre. Un seul parce que chaque passage charge le U-Net.

Ce que cela ne prouve pas : le build. Image de base, `apt-get`, torch depuis l'index
CPU, `chown` non-root, `read_only`, publication du port — rien de tout cela ne se
simule.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import config

DOCKERFILE = os.path.join(config.ROOT, "Dockerfile")
DEMO_REQUIREMENTS = os.path.join(config.ROOT, "requirements-demo.txt")

pytestmark = pytest.mark.skipif(
    not os.path.exists(config.DCE_MRI_UNET_CKPT),
    reason="checkpoint absent : rien à faire tourner",
)

# Nom de distribution -> module importable, quand les deux diffèrent.
DISTRIBUTION_TO_MODULE = {"pillow": "PIL"}

# Ce que pip installe *avec* les quatre paquets de `requirements-demo.txt`. Mesuré le
# 2026-09-20 par `pip install --dry-run --report` sur torch, Flask, numpy et Pillow :
# 17 paquets au total. Les voici moins les quatre, en noms de modules.
TRANSITIVE_MODULES = {
    "jinja2", "markupsafe", "werkzeug", "click", "itsdangerous", "blinker",
    "filelock", "fsspec", "sympy", "mpmath", "networkx", "typing_extensions",
    "setuptools", "pkg_resources", "_distutils_hack",
}

# Livrés à l'intérieur de la roue torch, donc présents dès que torch l'est.
BUNDLED_WITH_TORCH = {"torchgen"}

PROJECT_MODULES = ("config", "inference", "app.server", "app.predictor",
                   "imaging.unet", "logging_setup")


def _copy_pairs():
    """Les couples (source, destination) que les lignes `COPY` du Dockerfile décrivent."""
    pairs = []
    with open(DOCKERFILE, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("COPY "):
                continue
            *sources, destination = line[len("COPY "):].split()
            for source in sources:
                pairs.append((source, destination))
    return pairs


def _ignore_pycache(_directory, names):
    return [n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo"))]


def _build_mirror(destination_root):
    """Reconstruit l'arbre que les `COPY` produiraient, et le renvoie."""
    for source, destination in _copy_pairs():
        target = os.path.join(destination_root,
                              destination.lstrip("./").rstrip("/").replace("/", os.sep))
        origin = os.path.join(config.ROOT, source.rstrip("/").replace("/", os.sep))
        if os.path.isdir(origin):
            # `COPY rep/ dest/` copie le *contenu* de rep dans dest.
            shutil.copytree(origin, target, ignore=_ignore_pycache, dirs_exist_ok=True)
        else:
            os.makedirs(target or destination_root, exist_ok=True)
            shutil.copy2(origin, os.path.join(target or destination_root,
                                              os.path.basename(origin)))
    return destination_root


def _declared_modules():
    """Les modules qu'installe `requirements-demo.txt`, sous leur nom d'import."""
    modules = set()
    with open(DEMO_REQUIREMENTS, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                name = re.split(r"[<>=!~\[]", line)[0].strip().lower()
                modules.add(DISTRIBUTION_TO_MODULE.get(name, name))
    return modules


# Exécuté *dans* le miroir, avec les paquets hors image refusés à l'import.
PROBE = '''
import builtins, importlib.util, json, os, sys, sysconfig

AUTORISES = set(json.loads(os.environ["IMAGE_PACKAGES"]))
BLOQUES = []
_real_import = builtins.__import__
_roots = {os.path.normcase(sysconfig.get_paths()[k]) for k in ("purelib", "platlib")}


def _is_third_party(top):
    try:
        spec = importlib.util.find_spec(top)
    except Exception:
        return False
    origin = getattr(spec, "origin", None) or ""
    if not origin:
        locations = getattr(spec, "submodule_search_locations", None) or []
        origin = next(iter(locations), "")
    origin = os.path.normcase(origin)
    return any(origin.startswith(root) for root in _roots)


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    # level > 0 : import relatif. `name` y est relatif au paquet courant, donc son
    # premier segment n'est pas un nom de distribution -- werkzeug fait
    # `from .test import Client`, et lire "test" comme un paquet tiers refusait
    # werkzeug a lui-meme. Un import relatif ne peut de toute facon sortir du
    # paquet qui le fait, deja autorise pour qu'on en soit la.
    if level != 0 or name in sys.modules:
        return _real_import(name, globals, locals, fromlist, level)
    top = name.split(".")[0]
    if top not in AUTORISES and top not in sys.builtin_module_names and _is_third_party(top):
        BLOQUES.append(top)
        raise ImportError(name + " : absent de l'image (paquet non installe)")
    return _real_import(name, globals, locals, fromlist, level)


sys.path.insert(0, os.getcwd())
os.environ["MRI_APP_BACKEND"] = "dce_mri"
builtins.__import__ = guarded
try:
    import run_demo

    file_problems, cases = run_demo._check_files()
    model_problems, elapsed, result = run_demo._check_prediction(cases)

    import app.server as server

    client = server.app.test_client()
    page = client.post("/demo/1").get_data(as_text=True)
    home = client.get("/").status_code
    how = client.get("/comment-ca-marche").status_code
finally:
    builtins.__import__ = _real_import

origins = {}
for name in json.loads(os.environ["PROJECT_MODULES"]):
    module = sys.modules.get(name)
    origins[name] = getattr(module, "__file__", None)

# Le garde-fou sait-il encore refuser ? Ce qu'il a refuse au-dessus ne le dit pas :
# un environnement deja proche de l'image n'a rien a refuser, et zero refus y est la
# bonne reponse. On lui presente donc un paquet certainement present -- pytest, qui
# fait tourner ce fichier -- et certainement absent de l'image.
try:
    guarded("pytest")
    canari_refuse = False
except ImportError:
    canari_refuse = True

keep = ("best_slice", "n_slices", "slice_preselected", "backend")
print("<<<RAPPORT>>>" + json.dumps({
    "file_problems": file_problems,
    "model_problems": model_problems,
    "cases": cases,
    "result": {k: result[k] for k in keep} if result else None,
    "verdict_rendu": "R\\u00e9sultat de l\\'analyse" in page,
    "images": page.count("data:image/png;base64,"),
    "home": home,
    "how": how,
    "bloques": sorted(set(BLOQUES)),
    "canari_refuse": canari_refuse,
    "origins": origins,
}))
'''


@pytest.fixture(scope="module")
def image_report(tmp_path_factory):
    """Le rapport du sous-processus lancé dans le miroir. Une seule fois."""
    mirror = _build_mirror(str(tmp_path_factory.mktemp("image")))

    environment = dict(os.environ)
    environment["IMAGE_PACKAGES"] = json.dumps(
        sorted(_declared_modules() | TRANSITIVE_MODULES | BUNDLED_WITH_TORCH))
    environment["PROJECT_MODULES"] = json.dumps(list(PROJECT_MODULES))
    environment["PYTHONIOENCODING"] = "utf-8"
    # Le miroir doit se suffire : pas de PYTHONPATH hérité pointant vers le dépôt.
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run([sys.executable, "-c", PROBE], cwd=mirror, env=environment,
                               capture_output=True, text=True, encoding="utf-8")
    assert "<<<RAPPORT>>>" in completed.stdout, (
        "le sous-processus n'a pas produit de rapport\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    report = json.loads(completed.stdout.split("<<<RAPPORT>>>", 1)[1].splitlines()[0])
    report["mirror"] = mirror
    return report


def test_the_copy_list_carries_every_file_the_demo_opens(image_report):
    """Le préflight complet passe depuis le seul contenu de l'image."""
    assert image_report["file_problems"] == [], image_report["file_problems"]
    assert image_report["model_problems"] == [], image_report["model_problems"]
    assert len(image_report["cases"]) == 3, image_report["cases"]


def test_the_project_modules_come_from_the_image_not_the_checkout(image_report):
    """Sinon la mesure porterait sur le dépôt et ne prouverait rien.

    Le piège est réel : le dépôt est installé en mode éditable sur cette machine, donc
    `import config` trouve une cible même hors du miroir.
    """
    mirror = os.path.normcase(os.path.realpath(image_report["mirror"]))
    for name, path in image_report["origins"].items():
        assert path, f"{name} n'a pas été importé"
        assert os.path.normcase(os.path.realpath(path)).startswith(mirror), (
            f"{name} vient de {path}, pas de l'image"
        )


def test_the_demo_click_works_with_only_the_declared_packages(image_report):
    """Le clic, l'image annotée, le balayage des coupes et la vue MIP."""
    assert image_report["verdict_rendu"], "la page de résultat ne rend pas le verdict"
    # Une image par coupe du pavé, plus la coupe annotée et le MIP : bien plus que cinq.
    assert image_report["images"] > 5, image_report["images"]
    assert image_report["home"] == 200
    assert image_report["how"] == 200

    result = image_report["result"]
    assert result["backend"] == "dce_mri"
    assert result["slice_preselected"] is True
    assert 0 <= result["best_slice"] < result["n_slices"]


def test_the_guard_can_still_refuse(image_report):
    """Sans quoi les trois tests ci-dessus deviennent des tautologies vertes.

    Ce n'est **pas** « a-t-il refusé quelque chose ». Une première version l'affirmait
    et a échoué en CI : là-bas `tqdm`, `cffi`, `defusedxml` et `colorama` ne sont pas
    installés du tout, donc ne rien refuser est la bonne réponse — l'environnement est
    déjà proche de l'image. La propriété qui compte est que le mécanisme **sache**
    refuser, et on la mesure en lui présentant un paquet toujours présent (pytest fait
    tourner ce fichier) et jamais dans l'image.
    """
    assert image_report["canari_refuse"] is True, (
        "le garde-fou a laissé passer pytest : il ne protège plus rien, et les autres "
        "tests de ce fichier mesurent la machine au lieu de l'image"
    )


def test_the_copy_list_and_the_requirements_file_stay_in_step():
    """L'image installe la liste de la démo, et copie ce dont cette liste a besoin."""
    sources = {source for source, _ in _copy_pairs()}
    assert "requirements-demo.txt" in sources
    for expected in ("config.py", "inference.py", "run_demo.py", "app/", "imaging/"):
        assert expected in sources, f"{expected} n'est plus copié dans l'image"
