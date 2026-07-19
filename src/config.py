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
