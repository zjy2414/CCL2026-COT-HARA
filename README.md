# CoT-HARA: Automating Hazard Analysis and Risk Assessment for Autonomous Driving via Chain-of-Thought

> CCL26-Eval Task 11: Automated Hazard Analysis and Risk Assessment for Autonomous Driving

## Introduction

This project is built for the **CCL 2026 evaluation task on automated hazard analysis and risk assessment for autonomous driving**. It leverages large language models (LLMs) to automate the HARA (Hazard Analysis and Risk Assessment) process defined under the ISO 26262 standard.

### Background

As automotive electrical/electronic (E/E) architectures evolve toward deeper intelligence and connectivity, functional safety has become a critical cornerstone for the deployment of autonomous driving technology. HARA serves the core function of risk identification and top-level safety requirement definition. By systematically modeling vehicle operating scenarios, potential functional failure modes, and environmental factors, it quantifies risk along the following three dimensions:

| Dimension | Description |
|:---|:---|
| Severity (S) | Severity of potential harm (S0-S3) |
| Exposure (E) | Frequency of occurrence of the scenario (E0-E4) |
| Controllability (C) | Ability of the driver to avoid the hazard (C0-C3) |

The S/E/C scores are mapped through a lookup table to determine the **ASIL level** (QM / A / B / C / D), which is then translated into top-level Safety Goals and Fault Tolerant Time Intervals (FTTI).

### Method Overview

- **Model**: Qwen2.5-7B-Instruct (parameters < 35B, meeting evaluation requirements)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) + 8-bit quantization
- **Inference strategy**: Chain-of-Thought (CoT) prompting for step-by-step reasoning: S/E/C → lookup table → ASIL
- **System Prompt**: `CoT.md`, containing ISO 26262 standard definitions, the ASIL lookup table, and output format constraints

### Dataset

**Source**: [CCL 2026 HARA Evaluation Dataset](https://ccl2026-hara.github.io/dataset.html)

The evaluation dataset is derived from desensitized cases in real industrial scenarios, focusing on a core high-risk failure mode of the powertrain system — **unintended driving force / torque output**.

| Dataset | Samples | Usage |
|:---|:---|:---|
| Train | 1,600 | Model training |
| Validation | 400 | Leaderboard A |
| Test | 1,000 | Leaderboard B |

---

## Project Structure

```
ccl2026-cot-hara/
├── data/                         # Data directory
│   ├── train.json                # Training set
│   ├── val.json                  # Validation set
│   └── test.json                 # Test set
├── config.py                     # Default configuration (Python class)
├── config.yaml                   # Local override config (not tracked by git)
├── config.yaml.example           # Example configuration file
├── CoT.md                        # System Prompt (chain-of-thought definition)
├── data_loader.py                # Data loading, ASIL lookup, CoT reasoning construction
├── model.py                      # Model training (LoRA) and batched inference
├── train.py                      # Training entry script
├── inference.py                  # Inference entry script
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation (this file)
```

| File | Description |
|:---|:---|
| `CoT.md` | System prompt defining the ISO 26262 standard, ASIL lookup table, chain-of-thought steps, and output format |
| `config.py` | Default configuration with all model, training, and inference parameters |
| `config.yaml` | Local override config; tune parameters without modifying `config.py` |
| `data_loader.py` | Dataset loading, ASIL lookup functions, automatic CoT reasoning construction |
| `model.py` | Model training (LoRA + 8-bit quantization) and batched inference |
| `train.py` | Training entry point; supports CLI argument overrides |
| `inference.py` | Inference entry point; loads a checkpoint for batched prediction |
| `evaluate.py` | Evaluation script computing per-field accuracy |
| `format_submission.py` | Formats output into the CCL 2026 submission format with validation |

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8 (GPU recommended; VRAM >= 24GB)

### Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Basic Usage

```bash
# Train with default config (falls back to config.py defaults if no config.yaml exists)
python train.py

# Use a ModelScope model
# First set MODEL_SOURCE: "modelscope" in config.yaml
python train.py
```

### CLI Arguments

| Argument | Type | Default | Description |
|:---|:---|:---|:---|
| `--model_name` | str | `config.MODEL_NAME` | Model name |
| `--batch_size` | int | `config.BATCH_SIZE` | Batch size |
| `--learning_rate` | float | `config.LEARNING_RATE` | Learning rate |
| `--num_epochs` | int | `config.NUM_EPOCHS` | Number of epochs |

### Training Workflow

1. Load `data/train.json` and split into training and validation sets according to `TRAIN_SPLIT_RATIO` (default 9:1)
2. Load the base model (8-bit quantization) and apply LoRA
3. Start training; log loss every 10 steps and run evaluation every 100 steps
4. Automatically save the checkpoint with the best eval_loss to `outputs/best/`
5. Training logs are written to `outputs/logs_YYYYMMDD_HHMMSS/training_log.csv`

### Training Output

```
outputs/
├── best/                          # Best checkpoint (lowest eval_loss)
├── checkpoint-100/                # Checkpoint at step 100
├── checkpoint-200/                # Checkpoint at step 200
├── ...                            # Other checkpoints
├── logs_20260701_141731/
│   └── training_log.csv           # step,epoch,loss,eval_loss,lr,grad_norm
└── final_model/                   # Final model at the end of training
```

---

## Inference

### Basic Usage

```bash
# Inference with the fine-tuned model (LoRA checkpoint)
python inference.py \
    --model_path outputs/best \
    --data_path data/val.json

# Inference with the pure base model (zero-shot, without fine-tuned weights)
python inference.py \
    --no-lora \
    --data_path data/val.json

# Inference on the test set with a custom output path
python inference.py \
    --model_path outputs/best \
    --data_path data/test.json \
    --output_path outputs/test_predictions.json
```

### CLI Arguments

| Argument | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `--model_path` | str | No | — | Path to the fine-tuned model (LoRA adapter). Not needed when using `--no-lora` |
| `--data_path` | str | Yes | — | Path to the data file for inference |
| `--output_path` | str | No | `outputs/predictions.json` | Output path for results |
| `--no-lora` | flag | No | `False` | Inference with the pure base model, without any fine-tuned weights |

### Batched Inference

Inference uses `INFERENCE_BATCH_SIZE` to control the number of samples per batch (default 8), with GPU-parallel acceleration:

- Left padding + attention masks ensure batched inference produces results identical to per-sample inference
- Reduce `INFERENCE_BATCH_SIZE` if GPU memory is insufficient; setting it to 1 degenerates to per-sample inference
- Configure in `config.yaml`: `INFERENCE_BATCH_SIZE: 16`

### Inference Output

The approximate structure of `predictions.json` is as follows:

```json
[
   {
      "id": "sample_id",
      "input": { "Malfunction": "...", ... },
      "output": {
         "hazardous event": "Full causal-chain description",
         "Possible Hazard": "Type of immediate hazard consequence",
         "Persons at risk": "Groups at risk of injury",
         "Exposure or Frequency 'E'": "0-4",
         "Severity 'S'": "0-3",
         "Control ability 'C'": "0-3",
         "Resulting A SIL": "QM/A/B/C/D",
         "Safety Goal": "Safety goal or /",
         "FTTI": "Fault tolerant time interval or /"
      },
      "reasoning": "**Step 1: Scenario understanding**\n..."
   },
   ...
]
```

---

## Configuration

### Two-Layer Configuration Mechanism

The project uses a two-layer configuration of **config.py defaults + config.yaml overrides**:

1. `config.py`: defines default values for all configuration items
2. `config.yaml.example`: example file demonstrating configurable options

```bash
# Using configuration
cp config.yaml.example config.yaml   # Create a local config
nano config.yaml                       # Modify as needed
```

### Full Configuration Reference

#### Model Configuration

| Option | Default | Description |
|:---|:---|:---|
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace or ModelScope model name |
| `MODEL_SOURCE` | `huggingface` | Model source: `huggingface` or `modelscope` |
| `HUGGINGFACE_CACHE_DIR` | `null` | HuggingFace cache directory (`null` = default path) |
| `MODELSCOPE_CACHE_DIR` | `null` | ModelScope cache directory |
| `MAX_LENGTH` | `8192` | Maximum input token count |
| `TEMPERATURE` | `0.7` | Generation temperature (0 = greedy decoding) |
| `TOP_P` | `0.9` | Nucleus sampling threshold |

#### Training Configuration

| Option | Default | Description |
|:---|:---|:---|
| `BATCH_SIZE` | `1` | Batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Gradient accumulation steps (effective batch = 1 × 4 = 4) |
| `LEARNING_RATE` | `2e-5` | Learning rate |
| `NUM_EPOCHS` | `10` | Number of epochs |
| `WARMUP_STEPS` | `100` | Learning rate warmup steps |
| `TRAIN_SPLIT_RATIO` | `0.9` | Proportion of train.json used for training (the rest is used for eval) |

#### LoRA Configuration

| Option | Default | Description |
|:---|:---|:---|
| `USE_LORA` | `true` | Whether to use LoRA fine-tuning |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA scaling factor |
| `LORA_DROPOUT` | `0.1` | LoRA dropout |

#### Inference Configuration

| Option | Default | Description |
|:---|:---|:---|
| `INFERENCE_BATCH_SIZE` | `8` | Number of samples per inference batch (reduce if GPU memory is insufficient) |
| `NUM_BEAMS` | `4` | Number of beams for beam search |

### config.yaml Example

```yaml
# Model configuration
MODEL_NAME: "Qwen/Qwen2.5-7B-Instruct"
MODEL_SOURCE: "huggingface"
MAX_LENGTH: 8192
TEMPERATURE: 0.7
TOP_P: 0.9

# Training configuration
BATCH_SIZE: 1
GRADIENT_ACCUMULATION_STEPS: 4
LEARNING_RATE: 2e-5
NUM_EPOCHS: 10
TRAIN_SPLIT_RATIO: 0.9

# Inference configuration
INFERENCE_BATCH_SIZE: 8
```

---

## FAQ

### Q: What if I run out of GPU memory?

- Reduce `BATCH_SIZE`

### Q: What if inference is too slow?

- Increase `INFERENCE_BATCH_SIZE` (e.g., 8 → 16)
