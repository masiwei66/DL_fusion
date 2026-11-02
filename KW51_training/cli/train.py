"""Backward-compatible KW51 CLI delegating to the shared launcher."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from dl_fusion.train import main


if __name__ == "__main__":
    main()
