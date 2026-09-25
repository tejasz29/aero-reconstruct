"""Pytest bootstrap: make the project root importable without install.

This lets `pytest` resolve `import src.…` whether or not
`pip install -e .` was run.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
