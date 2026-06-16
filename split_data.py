"""
Split data into train, validation, and test sets.
Supports both MFCC and phonetic features, for Experiment 1 and Experiment 2.

Usage:
    # MFCC - Experience 1
    python split_data.py --features mfcc --exp 1
    
    # MFCC - Experience 2
    python split_data.py --features mfcc --exp 2
    
    # Phonetic 7 features - Experience 1
    python split_data.py --features phonetic --exp 1 --subfolder 7features
    
    # Phonetic 5 features - Experience 1
    python split_data.py --features phonetic --exp 1 --subfolder 5features
    
    # Phonetic 5 features - Experience 2
    python split_data.py --features phonetic --exp 2
"""

import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Split data into train/val/test")
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        choices=["mfcc", "phonetic"],
        help="Type of features: mfcc or phonetic"
    )
    parser.add_argument(
        "--exp",
        type=int,
        required=True,
        choices=[1, 2],
        help="Experiment number: 1 (5 languages) or 2 (ES vs PT)"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder for phonetic features (e.g., 7features, 5features). Required for exp 1."
    )
    return parser.parse_args()


def get_data_paths(features, exp, subfolder=None):
    """
    Determine input and output directories based on arguments.
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    if features == "mfcc":
        input_dir = data_dir / "processed_mfcc" / f"exp{exp}"
        output_dir = input_dir  # Save splits in same directory
        file_prefix = "mfcc"
    else:  # phonetic
        if exp == 1:
            if subfolder is None:
                raise ValueError("--subfolder required for exp 1 (e.g., 7features or 5features)")
            input_dir = data_dir / "processed_phonetic" / "exp1" / subfolder
            output_dir = input_dir
            file_prefix = "phonetic"
        else:  # exp 2
            input_dir = data_dir / "processed_phonetic" / "exp2"
            output_dir = input_dir
            file_prefix = "phonetic"
    
    return input_dir, output_dir, file_prefix


def split_data(X, y, random_state=42, test_size=0.3, val_size=0.5):
    """
    Split data into train (70%), validation (15%), test (15%).
    
    Args:
        X: Features array
        y: Labels array
        random_state: Random seed for reproducibility
        test_size: Proportion for temp set (default 0.3 = 30%)
        val_size: Proportion of temp set for validation (default 0.5 = 15% of total)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train (70%), temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: validation (50% of temp = 15%), test (50% of temp = 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    args = parse_args()
    
    # Get paths
    input_dir, output_dir, file_prefix = get_data_paths(
        args.features, args.exp, args.subfolder
    )
    
    print("=" * 50)
    print("DATA SPLIT")
    print("=" * 50)
    print(f"Features: {args.features}")
    print(f"Experiment: {args.exp}")
    if args.subfolder:
        print(f"Subfolder: {args.subfolder}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check input directory exists
    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist!")
        print("Please run feature extraction first.")
        return
    
    # Load data
    try:
        X = np.load(input_dir / "all_mfcc.npy") if args.features == "mfcc" else np.load(input_dir / "all_phonetic.npy")
        y = np.load(input_dir / "all_labels.npy")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load data: {e}")
        print(f"Make sure the following files exist in {input_dir}:")
        if args.features == "mfcc":
            print("  - all_mfcc.npy")
        else:
            print("  - all_phonetic.npy")
        print("  - all_labels.npy")
        return
    
    print(f"Loaded: X {X.shape}, y {y.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Print distribution
    print(f"\nTrain: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Save splits
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    print(f"\n✅ Splits saved to {output_dir}")
    print("   - X_train.npy, y_train.npy")
    print("   - X_val.npy, y_val.npy")
    print("   - X_test.npy, y_test.npy")
    
    # Verify class distribution
    print("\n--- Class distribution in splits ---")
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        print(f"\n{split_name}:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count}")


if __name__ == "__main__":
    main()