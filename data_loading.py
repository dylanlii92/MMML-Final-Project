"""
Sample exactly N files per language from raw data to create a balanced dataset.
Supports both Experience 1 and Experience 2.


Usage:
    # Experience 1: 5 languages, 1200 samples each
    python data_loading.py --exp 1 --seed 42
    
    # Experience 2: Spanish vs Portuguese, 2000 samples each
    python data_loading.py --exp 2 --seed 42
"""

import random
import shutil
import argparse
from pathlib import Path

# Default configuration
DEFAULT_RANDOM_SEED = 42

# Language configurations per experiment
EXP_CONFIGS = {
    1: {
        "name": "Experience 1 - 5 Languages",
        "langs": {
            "en": "en",
            "fr": "fr",
            "de": "de",
            "zh": "zh",
            "es": "es"
        },
        "output_dir": "exp1",
        "target_samples": 1200
    },
    2: {
        "name": "Experience 2 - Spanish vs Portuguese",
        "langs": {
            "es": "es",
            "pt": "pt"
        },
        "output_dir": "exp2",
        "target_samples": 2000
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sample audio files per language")
    parser.add_argument(
        "--exp",
        type=int,
        required=True,
        choices=[1, 2],
        help="Experiment number: 1 (5 languages) or 2 (ES vs PT)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})"
    )
    return parser.parse_args()


def sample_language(source_path, target_path, target_count, verbose=True):
    """
    Randomly sample target_count files from source directory
    and copy them to target directory.
    """
    # Get all wav files from source
    all_files = list(source_path.glob("*.wav"))
    all_files += list(source_path.glob("*.mp3"))
    all_files += list(source_path.glob("*.flac"))
    all_files = list(set(all_files))
    
    if verbose:
        print(f"  Found {len(all_files)} total files")
    
    # Sample randomly
    if len(all_files) >= target_count:
        selected = random.sample(all_files, target_count)
    else:
        if verbose:
            print(f"  WARNING: Only {len(all_files)} available, taking all")
        selected = all_files
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files
    for existing_file in target_path.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()
    
    # Copy selected files
    for src_file in selected:
        dst_file = target_path / src_file.name
        shutil.copy2(src_file, dst_file)
    
    if verbose:
        print(f"  Copied {len(selected)} files to {target_path}")
    
    return len(selected)


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Get experiment configuration
    exp_config = EXP_CONFIGS[args.exp]
    lang_folders = exp_config["langs"]
    output_subdir = exp_config["output_dir"]
    target_samples = exp_config["target_samples"]
    
    # Define paths - script is at project root
    base_dir = Path(__file__).parent  # project root
    data_dir = base_dir / "data"
    
    source_dir = data_dir / "raw_data"
    target_dir = data_dir / "sampled_data" / output_subdir
    
    print("=" * 50)
    print("DATA LOADING AND SAMPLING")
    print("=" * 50)
    print(f"Experiment: {exp_config['name']}")
    print(f"Languages: {', '.join(lang_folders.keys())}")
    print(f"Target samples per language: {target_samples}")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Check source directory exists
    if not source_dir.exists():
        print(f"ERROR: Source directory '{source_dir}' does not exist!")
        print("Please create it with the following structure:")
        print("  data/")
        print("    raw_data/")
        for lang in lang_folders.keys():
            print(f"      {lang}/")
        return
    
    # Create target base directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    for lang_code, lang_folder in lang_folders.items():
        print(f"\nProcessing {lang_code}...")
        
        source_path = source_dir / lang_folder
        target_path = target_dir / lang_code
        
        if not source_path.exists():
            print(f"  ERROR: {source_path} does not exist!")
            continue
        
        copied = sample_language(source_path, target_path, target_samples)
        total_copied += copied
    
    # Verification
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    
    for lang_code in lang_folders.keys():
        target_path = target_dir / lang_code
        if target_path.exists():
            count = len(list(target_path.glob("*.wav")))
            status = "✓" if count == target_samples else "⚠"
            print(f"{status} {lang_code}: {count} files")
        else:
            print(f"✗ {lang_code}: directory not found")
    
    print(f"\nTotal files in {target_dir}: {total_copied}")
    print(f"Expected total: {len(lang_folders) * target_samples}")
    
    if total_copied == len(lang_folders) * target_samples:
        print("\n✓ Dataset is perfectly balanced!")
    else:
        print("\n⚠ Warning: Dataset is not perfectly balanced. Check above.")


if __name__ == "__main__":
    main()