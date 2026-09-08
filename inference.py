"""
Inference script.
"""

import argparse
import os
from config import Config
from data_loader import HARADataset
from model import HARAModel


def main():
    parser = argparse.ArgumentParser(description='HARA model inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the fine-tuned model (LoRA adapter). Omit when using --no-lora with the pure base model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to the output file')
    parser.add_argument('--no-lora', action='store_true', default=False,
                        help='Inference with the pure base model (without fine-tuned weights)')

    args = parser.parse_args()

    # Validate arguments
    if not args.no_lora and args.model_path is None:
        parser.error("You must specify either --model_path (fine-tuned model) or --no-lora (pure base model inference)")

    # Update configuration
    config = Config()

    # Set the default output path
    if args.output_path is None:
        mode_suffix = "base" if args.no_lora else "finetuned"
        args.output_path = os.path.join(config.OUTPUT_DIR, f"predictions_{mode_suffix}.json")

    print("=" * 50)
    print("HARA Model Inference")
    print("=" * 50)
    print(f"Inference mode: {'pure base model (no fine-tuning)' if args.no_lora else 'fine-tuned model (LoRA)'}")
    if args.model_path:
        print(f"Model path: {args.model_path}")
    print(f"Base model: {config.MODEL_NAME}")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print("=" * 50)

    # Load data
    print("\nLoading data...")
    dataset = HARADataset(args.data_path, tokenizer=None)
    print(f"Dataset size: {len(dataset)}")

    # Initialize the model
    print("\nInitializing model...")
    model = HARAModel(config)
    if args.no_lora:
        model.load_base_model()
    else:
        model.load_model(args.model_path)

    # Run inference (results are saved incrementally)
    print("\nStarting inference...")
    model.predict(dataset, args.output_path)

    print("\nInference complete!")


if __name__ == "__main__":
    main()
