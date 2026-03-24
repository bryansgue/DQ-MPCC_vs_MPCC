"""
tuning_config.py -- COMPATIBILITY SHIM.

The real configuration now lives in config/tuning_config.py.
This file re-exports everything so existing imports still work.
"""
from config.tuning_config import *  # noqa: F401,F403
