"""
experiment_config.py -- COMPATIBILITY SHIM.

The real configuration now lives in config/experiment_config.py.
This file re-exports everything so existing imports still work.
"""
from config.experiment_config import *  # noqa: F401,F403
