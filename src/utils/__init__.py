"""src/utils package.

Note: device.py (torch-dependent) is NOT imported here to keep the database,
dataset, and benchmark CLIs runnable without torch installed.
Import device utilities directly when needed in inference code:
    from src.utils.device import resolve_device
"""
from src.utils.config_loader import (
    load_audio_config,
    load_benchmark_config,
    load_datasets_config,
    load_models_config,
    get_data_dir,
    get_model_cache_dir,
    get_outputs_dir,
    get_reports_dir,
)
from src.utils.logger import get_logger, log_inference_event

__all__ = [
    "load_audio_config",
    "load_benchmark_config",
    "load_datasets_config",
    "load_models_config",
    "get_data_dir",
    "get_model_cache_dir",
    "get_outputs_dir",
    "get_reports_dir",
    "get_logger",
    "log_inference_event",
]
