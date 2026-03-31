#!/usr/bin/env python3
"""Compatibility wrapper for the relocated Experiment 2 LaTeX generator."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[1] / "results" / "experiment2" / "reports" / "generate_sweep.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
