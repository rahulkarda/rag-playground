import tempfile
import os
from src.config import load_yaml_config

if __name__ == "__main__":
    # Create a minimal YAML config file
    yaml_content = """
model: all-MiniLM-L6-v2
retriever:
  type: bm25
  k: 10
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name
    try:
        config = load_yaml_config(tmp_path)
        print("Loaded config:")
        print(config)  # Should print a dict with model and retriever keys
        assert config["model"] == "all-MiniLM-L6-v2"
        assert config["retriever"]["type"] == "bm25"
        assert config["retriever"]["k"] == 10
        print("Config loader test passed.")
    finally:
        os.remove(tmp_path)
