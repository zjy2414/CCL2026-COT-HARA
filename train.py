"""
Training script.
"""

import argparse
from config import Config
from data_loader import load_datasets
from model import HARAModel


def main():
    parser = argparse.ArgumentParser(description='HARA model training')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')

    args = parser.parse_args()

    # Update configuration
    config = Config()
    if args.model_name:
        config.MODEL_NAME = args.model_name
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.num_epochs:
        config.NUM_EPOCHS = args.num_epochs

    print("=" * 50)
    print("HARA Model Training")
    print("=" * 50)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 50)

    # Initialize the model
    print("\nInitializing model...")
    model = HARAModel(config)

    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset = load_datasets(config, tokenizer=model.tokenizer)

    # Train the model
    print("\nStarting training...")
    model.train(train_dataset, val_dataset)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
