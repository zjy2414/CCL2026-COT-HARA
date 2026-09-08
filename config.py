"""
Configuration for the HARA evaluation task.

Values in config.yaml (if present) override the defaults defined in this file.
"""

import os

class Config:
    # Data paths
    TRAIN_DATA_PATH = "data/train.json"
    VAL_DATA_PATH = "data/val.json"
    OUTPUT_DIR = "outputs"

    # Model configuration
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_SOURCE = "huggingface"  # Model source: "huggingface" or "modelscope"
    HUGGINGFACE_CACHE_DIR = None  # HuggingFace cache directory; None uses the default cache path
    MODELSCOPE_CACHE_DIR = None   # ModelScope cache directory; None uses the default cache path
    MAX_LENGTH = 8192
    TEMPERATURE = 0.7
    TOP_P = 0.9

    # Training configuration
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    WARMUP_STEPS = 100
    TRAIN_SPLIT_RATIO = 0.95  # Fraction of train.json used for training (the rest is used for validation)

    # LoRA configuration
    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1

    # Inference configuration
    INFERENCE_BATCH_SIZE = 8
    NUM_BEAMS = 4

    # LLM-as-a-Judge evaluation configuration (HLM/PCC/RCC/SGA scoring, optional)
    # Enable with: evaluate.py --llm-judge
    # Supports OpenAI-compatible APIs (e.g., Qwen3.6-Max, GPT-4o)
    LLM_JUDGE_API_KEY = None      # API key; can also be set via the environment variable LLM_JUDGE_API_KEY
    LLM_JUDGE_API_BASE = None     # API base URL; can also be set via the environment variable LLM_JUDGE_API_BASE
    LLM_JUDGE_MODEL = "qwen-max"  # Model name
    LLM_JUDGE_CACHE_DIR = None    # Scoring cache directory; None uses outputs/.llm_cache

    # ASIL level mapping
    ASIL_LEVELS = ["QM", "A", "B", "C", "D"]

    # Risk parameter ranges
    EXPOSURE_RANGE = [0, 1, 2, 3, 4]  # E: 0-4
    SEVERITY_RANGE = [0, 1, 2, 3]  # S: 0-3 (ISO 26262: S0-S3)
    CONTROLLABILITY_RANGE = [0, 1, 2, 3]  # C: 0-3 (ISO 26262: C0-C3)


def _load_yaml_config():
    """Load configuration from config.yaml and override Config class attributes."""
    _yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(_yaml_path):
        return

    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML is not installed; skipping config.yaml loading. Install with: pip install pyyaml")
        return

    with open(_yaml_path, "r", encoding="utf-8") as f:
        _yaml_config = yaml.safe_load(f)

    if not _yaml_config:
        return

    for key, value in _yaml_config.items():
        if hasattr(Config, key):
            setattr(Config, key, value)
        else:
            print(f"Warning: '{key}' in config.yaml is not a valid configuration item and was ignored")

    print("Loaded override configuration from config.yaml")


_load_yaml_config()
