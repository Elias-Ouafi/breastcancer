"""One command to run the investor demo: ``python run_demo.py``.

Checks that everything the demo needs is present *and works*, wires the real DCE-MRI
model in, and starts the local Flask app. It replaces the "set MRI_APP_BACKEND,
remember the right module, hope the checkpoint is there" dance -- and, more
importantly, it fails with a sentence you can act on instead of a stack trace
mid-pitch.

Everything it needs is committed (checkpoint + the three curated cases), so a fresh
``git clone`` + ``pip install -e .`` is enough. ``--check`` runs
the preflight and exits, which is the fast way to confirm a machine is demo-ready
without occupying a port.

**The preflight scores a real case** (~2 s), it does not merely stat the files. A
truncated checkpoint, a torch build that cannot start, a case whose keys moved: all
of those pass an existence check and die on the first click, which is the one moment
they must not. Skip it with ``--fast-check`` when you only want the file inventory.

Local-only, like ``app.server``: binds to 127.0.0.1 and nothing else.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import config

ROOT = config.ROOT
CHECKPOINT = os.path.relpath(config.DCE_MRI_UNET_CKPT, ROOT)
DEMO_DIR = os.path.relpath(config.DEMO_CASES_DIR, ROOT)

INSTALL_HINT = "    -> pip install -e ."


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing is listening on ``host:port``.

    Asked by connecting, not by binding. On Windows a second bind to a port werkzeug
    already holds *succeeds* -- SO_REUSEADDR lets it -- so the launcher printed its
    "Running on http://127.0.0.1:5000" banner while the OS kept routing every request
    to the first process. A demo then shows stale pages with no error anywhere.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) != 0


def _check_files():
    """Problems with what a clone must have received. Empty means the files are there."""
    problems = []

    try:
        import torch  # noqa: F401
    except ImportError:
        problems.append("PyTorch n'est pas installé. Le modèle ne peut pas être chargé.\n"
                        + INSTALL_HINT)

    try:
        import flask  # noqa: F401
    except ImportError:
        problems.append("Flask n'est pas installé.\n" + INSTALL_HINT)

    ckpt = os.path.join(ROOT, CHECKPOINT)
    if not os.path.exists(ckpt):
        problems.append(
            f"Checkpoint absent : {CHECKPOINT}\n"
            "    Il est versionné dans le dépôt ; un fichier manquant signifie un\n"
            "    clone incomplet. -> git checkout -- " + CHECKPOINT
        )

    demo_dir = os.path.join(ROOT, DEMO_DIR)
    cases = sorted(f for f in os.listdir(demo_dir) if f.endswith(".npz")) \
        if os.path.isdir(demo_dir) else []
    if not cases:
        problems.append(
            f"Aucun cas de démo dans {DEMO_DIR}/\n"
            "    Ils sont versionnés ; sinon régénérer :\n"
            "    -> python scripts/make_demo_cases.py (nécessite les volumes complets)"
        )

    return problems, cases


def _check_prediction(cases):
    """Score the first bundled case. Returns (problems, seconds, result-or-None).

    This is the check that matters: it exercises the exact call the first click makes
    -- load the checkpoint, read the case, run the U-Net -- so anything that would
    turn that click into a 500 surfaces here instead, with time to react.
    """
    if not cases:
        return [], None, None

    os.environ.setdefault("MRI_APP_BACKEND", "dce_mri")
    case = os.path.join(ROOT, DEMO_DIR, cases[0])
    started = time.perf_counter()
    try:
        from app.predictor import get_predictor

        result = get_predictor().predict(case)
    except Exception as exc:
        return ([f"Le modèle n'a pas pu analyser {cases[0]} :\n"
                 f"    {type(exc).__name__}: {exc}\n"
                 "    Le fichier existe mais le calcul échoue — checkpoint tronqué,\n"
                 "    installation de PyTorch cassée, ou cas de démo corrompu.\n"
                 "    -> git status puis git checkout -- " + CHECKPOINT + " " + DEMO_DIR],
                None, None)

    elapsed = time.perf_counter() - started
    if result.get("best_slice") is None:
        return ([f"{cases[0]} a été analysé mais aucune coupe n'a été retenue.\n"
                 "    La démo afficherait un résultat vide.\n"
                 "    -> python scripts/make_demo_cases.py (nécessite les volumes complets)"],
                elapsed, result)
    return [], elapsed, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Vérifier que la machine est prête, sans démarrer le serveur")
    parser.add_argument("--fast-check", action="store_true",
                        help="Contrôler seulement les fichiers, sans charger le modèle")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MRI_APP_PORT", "5000")),
                        help="Port local (défaut : 5000, ou $MRI_APP_PORT)")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Ouvrir le navigateur sur la démo une fois le serveur prêt")
    args = parser.parse_args(argv)

    problems, cases = _check_files()
    elapsed = None
    if not problems and not args.fast_check:
        model_problems, elapsed, _ = _check_prediction(cases)
        problems += model_problems

    if not problems and not args.check and not _port_is_free(args.port):
        problems.append(
            f"Le port {args.port} est déjà utilisé.\n"
            "    Une démo y tourne peut-être déjà : ouvrir\n"
            f"    http://127.0.0.1:{args.port} avant de relancer.\n"
            f"    -> python run_demo.py --port {args.port + 1}"
        )

    if problems:
        print("La démo ne peut pas démarrer :\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}\n", file=sys.stderr)
        return 1

    print("Écoute — démo locale (Research Use Only, pas un dispositif médical)")
    print(f"  modèle    : {CHECKPOINT}")
    print(f"  cas prêts : {', '.join(cases)}")
    if elapsed is not None:
        print(f"  essai     : {cases[0]} analysé en {elapsed:.1f} s")
    elif args.fast_check:
        print("  essai     : ignoré (--fast-check) — le modèle n'a pas été chargé")
    if args.check or args.fast_check:
        print("\nMachine prête. Lancer sans --check pour démarrer.")
        return 0

    demo_dir_display = DEMO_DIR.replace("\\", os.sep)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Ouvrir    : {url}")
    print(f"  Déposer   : un fichier .npz de {demo_dir_display}{os.sep} puis « Analyser l'examen »")
    print("  Arrêter   : Ctrl+C\n")

    os.environ["MRI_APP_BACKEND"] = "dce_mri"
    os.environ["MRI_APP_PORT"] = str(args.port)
    sys.path.insert(0, ROOT)

    if args.open_browser:
        # After the print above, before the blocking serve(): the browser retries a
        # refused connection, so a page opened a beat early still lands.
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    from app.server import main as serve

    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
