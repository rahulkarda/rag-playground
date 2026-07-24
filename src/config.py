"""
YAML config loader utility for rag-playground.

Provides:
- load_yaml_config(path): parses YAML file and returns a dict.

Usage:
    from src.config import load_yaml_config
    config = load_yaml_config('config.yaml')

This utility is used for experiment configuration, model selection, and pipeline parameters.
"""
import yaml
from typing import Any, Dict

def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return as a dict.
    Args:
        path (str): Path to YAML file.
    Returns:
        dict: Parsed config dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
