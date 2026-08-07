"""Makes the repository root importable for the test suite.

Tests import top-level modules (``config``) and the ``imaging`` package. Under
pytest's default import mode, a test file in ``tests/`` (which has no ``__init__.py``)
gets *its own* directory prepended to ``sys.path``, not the repository root -- so
``import config`` would only resolve when the project happens to be pip-installed.

pytest imports the root ``conftest.py`` before collecting anything and prepends its
directory, so simply existing here is what fixes it. That keeps ``pytest`` working on
a bare checkout, which is how CI runs it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
