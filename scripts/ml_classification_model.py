#!/usr/bin/env python3
"""
Compatibility script for running the central Health Risk Classification model.

The core implementation now lives in `health_risk_model.core_model` so that the
encrypted healthcare risk model can be controlled from a single location.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health_risk_model.core_model import main


if __name__ == "__main__":
    main()
