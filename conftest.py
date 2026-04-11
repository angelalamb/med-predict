"""
Root conftest.py — adds the project root to sys.path so that pytest can
resolve imports like `from config import ...` and `from tracking import ...`
without a package install.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
