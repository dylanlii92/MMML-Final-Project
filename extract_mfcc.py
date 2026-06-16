"""
Extract MFCC features for both Experiment 1 and Experiment 2.


Usage:
    # Experience 1: 5 languages (EN, FR, DE, ZH, ES)
    python extract_mfcc.py --exp 1
    
    # Experience 2: Spanish vs Portuguese
    python extract_mfcc.py --exp 2
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse

# Configuration
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 13
HOP_LENGTH = int(0.01 * SAMPLE_RATE)
N_FFT = int(0.025 * SAMPLE_RATE)
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

# Language configurations per experiment
EXP_CONFIGS = {
    1: {
        "name": "Experience 1 - 5 Languages",
        "langs": {
            "en": 0,
            "fr": 1,
            "de": 2,
            "zh": 3,
            "es": 4
        },
        "input_subdir": "exp1",
    },
    2: {
        "name": "Experience 2 - Spanish vs Portuguese",
        "langs": {
            "es": 0,
            "pt": 1
        },
        "input_subdir": "exp2",
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract MFCC features")
    parser.add_argument(
        "--exp",
        type=int,
        required=True,
        choices=[1, 2],
        help="Experiment number: 1 (5 languages) or 2 (ES vs PT)"
    )
    return parser.parse_args()


def extract_mfcc_from_file(file_path):
    """Extract MFCCs from a single audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        if len(y) >= TARGET_LENGTH:
            y = y[:TARGET_LENGTH]
        else:
            padding = TARGET_LENGTH - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        mfcc = mfcc.T
        return mfcc
    
    except Exception as e:
        print(f"Error with {file_path.name}: {e}")
        return None


def process_language(lang_dir, lang_name, label, output_dir):
    """Process all files for one language."""
    print(f"\nProcessing {lang_name}...")
    
    all_mfccs = []
    failed = 0
    
    wav_files = list(Path(lang_dir).glob("*.wav"))
    
    for file_path in tqdm(wav_files, desc=f"{lang_name}"):
        mfcc = extract_mfcc_from_file(file_path)
        if mfcc is not None:
            all_mfccs.append(mfcc)
        else:
            failed += 1
    
    if failed > 0:
        print(f"  {failed} files failed")
    
    if len(all_mfccs) == 0:
        print(f"  ERROR: No features extracted for {lang_name}")
        return None, None
    
    X = np.array(all_mfccs)
    y = np.full(len(all_mfccs), label)
    
    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")
    
    np.save(os.path.join(output_dir, f"{lang_name}_mfcc.npy"), X)
    np.save(os.path.join(output_dir, f"{lang_name}_labels.npy"), y)
    
    return X, y


def main():
    args = parse_args()
    
    # Get experiment configuration
    exp_config = EXP_CONFIGS[args.exp]
    lang_to_label = exp_config["langs"]
    input_subdir = exp_config["input_subdir"]
    
    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    input_dir = data_dir / "sampled_data" / input_subdir
    output_dir = data_dir / "processed_mfcc" / input_subdir  # <- Changé ici
    
    print("=" * 50)
    print("MFCC EXTRACTION")
    print("=" * 50)
    print(f"Experiment: {exp_config['name']}")
    print(f"Languages: {', '.join(lang_to_label.keys())}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check input directory exists
    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist!")
        print("Please run data_loading.py first.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_X = []
    all_y = []
    
    for lang_name, label in lang_to_label.items():
        lang_dir = input_dir / lang_name
        if not lang_dir.exists():
            print(f"Warning: {lang_dir} doesn't exist")
            continue
        
        X, y = process_language(lang_dir, lang_name, label, output_dir)
        if X is not None:
            all_X.append(X)
            all_y.append(y)
    
    if len(all_X) == 0:
        print("\nERROR: No data processed!")
        return
    
    # Combine all languages
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    print("\n" + "=" * 50)
    print("FULL DATASET")
    print("=" * 50)
    print(f"X_all shape: {X_all.shape}")
    print(f"y_all shape: {y_all.shape}")
    print(f"Total samples: {len(X_all)}")
    
    np.save(output_dir / "all_mfcc.npy", X_all)
    np.save(output_dir / "all_labels.npy", y_all)
    
    print("\n--- Class distribution ---")
    for lang_name, label in lang_to_label.items():
        count = np.sum(y_all == label)
        print(f"{lang_name}: {count} samples")
    
    print(f"\n✅ MFCC extraction complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()