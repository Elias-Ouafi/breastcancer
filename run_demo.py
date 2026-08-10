"""One command to run the investor demo: ``python run_demo.py``.

Checks that everything the demo needs is present, wires the real DCE-MRI model in,
and starts the local Flask app. It replaces the "set MRI_APP_BACKEND, remember the
right module, hope the checkpoint is there" dance -- and, more importantly, it fails
with a sentence you can act on instead of a stack trace mid-pitch.

Everything it needs is committed (checkpoint + the three curated cases), so a fresh
``git clone`` + ``pip install -r requirements.txt`` is enough. ``--check`` runs the
preflight and exits, which is the fast way to confirm a machine is demo-ready
without occupying a port.

Local-only, like ``app.server``: binds to 127.0.0.1 and nothing else.
"""
from __future__ import annotations

import argparse
import os
import sys

import config

ROOT = config.ROOT
CHECKPOINT = os.path.relpath(config.DCE_MRI_UNET_CKPT, ROOT)
DEMO_DIR = os.path.relpath(config.DEMO_CASES_DIR, ROOT)


def _preflight():
    """Return a list of human-readable problems; empty means ready to run."""
    problems = []

    try:
        import torch  # noqa: F401
    except ImportError:
        problems.append(
            "PyTorch n'est pas installé. Le modèle ne peut pas être chargé.\n"
            "    -> pip install -r requirements.txt"
        )

    try:
        import flask  # noqa: F401
    except ImportError:
        problems.append(
            "Flask n'est pas installé.\n"
            "    -> pip install -r requirements.txt"
        )

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Vérifier que la machine est prête, sans démarrer le serveur")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MRI_APP_PORT", "5000")),
                        help="Port local (défaut : 5000, ou $MRI_APP_PORT)")
    args = parser.parse_args(argv)

    problems, cases = _preflight()
    if problems:
        print("La démo ne peut pas démarrer :\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}\n", file=sys.stderr)
        return 1

    print("Écoute — démo locale (Research Use Only, pas un dispositif médical)")
    print(f"  modèle    : {CHECKPOINT}")
    print(f"  cas prêts : {', '.join(cases)}")
    if args.check:
        print("\nMachine prête. Lancer sans --check pour démarrer.")
        return 0

    print(f"\n  Ouvrir    : http://127.0.0.1:{args.port}")
    print(f"  Déposer   : un fichier de {DEMO_DIR}/ puis « Analyser l'examen »")
    print("  Arrêter   : Ctrl+C\n")

    os.environ["MRI_APP_BACKEND"] = "dce_mri"
    os.environ["MRI_APP_PORT"] = str(args.port)
    sys.path.insert(0, ROOT)

    from app.server import main as serve

    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
