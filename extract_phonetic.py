"""
Extract phonetic features for both Experiment 1 and Experiment 2.
Supports 7 features (with pitch) and 5 features (without pitch).

Usage:
    # Experience 1: 5 languages, 7 features (with pitch)
    python extract_phonetic.py --exp 1 --features 7
    
    # Experience 1: 5 languages, 5 features (without pitch)
    python extract_phonetic.py --exp 1 --features 5
    
    # Experience 2: Spanish vs Portuguese, 5 features (without pitch)
    python extract_phonetic.py --exp 2 --features 5
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse
import parselmouth
from parselmouth.praat import call

# Configuration
SAMPLE_RATE = 16000
DURATION = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
HOP_LENGTH = int(0.01 * SAMPLE_RATE)
FRAME_LENGTH = int(0.025 * SAMPLE_RATE)

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
    parser = argparse.ArgumentParser(description="Extract phonetic features")
    parser.add_argument(
        "--exp",
        type=int,
        required=True,
        choices=[1, 2],
        help="Experiment number: 1 (5 languages) or 2 (ES vs PT)"
    )
    parser.add_argument(
        "--features",
        type=int,
        required=True,
        choices=[5, 7],
        help="Number of features: 5 (without pitch) or 7 (with pitch)"
    )
    return parser.parse_args()


def extract_formants_praat(y, sr):
    """Extract formants (F1, F2, F3) using Praat."""
    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        
        num_frames = formant.get_number_of_frames()
        if num_frames == 0:
            return np.zeros((1, 3))
        
        formants_list = []
        for frame_idx in range(1, num_frames + 1):
            time = formant.get_time_from_frame_number(frame_idx)
            f1 = formant.get_value_at_time(1, time)
            f2 = formant.get_value_at_time(2, time)
            f3 = formant.get_value_at_time(3, time)
            
            f1 = f1 if not np.isnan(f1) else 0.0
            f2 = f2 if not np.isnan(f2) else 0.0
            f3 = f3 if not np.isnan(f3) else 0.0
            
            formants_list.append([f1, f2, f3])
        
        return np.array(formants_list)
    
    except Exception as e:
        print(f"  Formant extraction warning: {e}")
        return np.zeros((1, 3))


def extract_pitch_librosa(y, sr):
    """Extract pitch (F0) using librosa PYIN algorithm."""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=75, fmax=450, sr=sr,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        delta_f0 = np.gradient(f0)
        return f0, delta_f0
    
    except Exception as e:
        print(f"  Pitch extraction warning: {e}")
        num_frames = int(len(y) / HOP_LENGTH) + 1
        return np.zeros(num_frames), np.zeros(num_frames)


def extract_energy_and_zcr(y, sr):
    """Extract RMS energy and zero-crossing rate per frame."""
    try:
        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]
        return rms, zcr
    
    except Exception as e:
        print(f"  Energy/ZCR extraction warning: {e}")
        return np.zeros(1), np.zeros(1)


def extract_phonetic_features_from_file(file_path, num_features):
    """
    Extract phonetic features from a single audio file.
    
    Args:
        file_path: Path to audio file
        num_features: 5 (no pitch) or 7 (with pitch)
    
    Returns:
        features: (time_frames, num_features)
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        if len(y) >= TARGET_LENGTH:
            y = y[:TARGET_LENGTH]
        else:
            padding = TARGET_LENGTH - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        
        # Extract formants (always needed)
        formants = extract_formants_praat(y, sr)
        
        # Extract pitch if needed (7 features)
        if num_features == 7:
            f0, delta_f0 = extract_pitch_librosa(y, sr)
            num_frames = len(f0)
        else:
            num_frames = len(formants)
            f0 = None
            delta_f0 = None
        
        # Extract energy and ZCR (always needed)
        rms, zcr = extract_energy_and_zcr(y, sr)
        
        # Resample formants if needed
        if len(formants) != num_frames and len(formants) > 1:
            formants_interp = np.zeros((num_frames, 3))
            for i in range(3):
                formants_interp[:, i] = np.interp(
                    np.linspace(0, len(formants)-1, num_frames),
                    np.arange(len(formants)),
                    formants[:, i]
                )
            formants = formants_interp
        elif len(formants) == 1:
            formants = np.tile(formants[0], (num_frames, 1))
        
        # Resample RMS and ZCR if needed
        if len(rms) != num_frames and len(rms) > 1:
            rms = np.interp(
                np.linspace(0, len(rms)-1, num_frames),
                np.arange(len(rms)),
                rms
            )
            zcr = np.interp(
                np.linspace(0, len(zcr)-1, num_frames),
                np.arange(len(zcr)),
                zcr
            )
        elif len(rms) == 1:
            rms = np.full(num_frames, rms[0])
            zcr = np.full(num_frames, zcr[0])
        
        # Ensure all arrays have same length
        if num_features == 7:
            min_len = min(len(formants), len(f0), len(delta_f0), len(rms), len(zcr))
            formants = formants[:min_len]
            f0 = f0[:min_len]
            delta_f0 = delta_f0[:min_len]
            rms = rms[:min_len]
            zcr = zcr[:min_len]
            
            features = np.column_stack([
                formants[:, 0],  # F1
                formants[:, 1],  # F2
                formants[:, 2],  # F3
                f0,              # Pitch
                delta_f0,        # Delta pitch
                rms,             # Energy
                zcr              # Zero-crossing rate
            ])
        else:  # 5 features
            min_len = min(len(formants), len(rms), len(zcr))
            formants = formants[:min_len]
            rms = rms[:min_len]
            zcr = zcr[:min_len]
            
            features = np.column_stack([
                formants[:, 0],  # F1
                formants[:, 1],  # F2
                formants[:, 2],  # F3
                rms,             # Energy
                zcr              # Zero-crossing rate
            ])
        
        return features
    
    except Exception as e:
        print(f"Error with {file_path.name}: {e}")
        return None


def process_language(lang_dir, lang_name, label, output_dir, num_features):
    """Process all files for one language."""
    print(f"\nProcessing {lang_name}...")
    
    all_features = []
    failed = 0
    
    wav_files = list(Path(lang_dir).glob("*.wav"))
    
    for file_path in tqdm(wav_files, desc=f"{lang_name}"):
        features = extract_phonetic_features_from_file(file_path, num_features)
        if features is not None:
            all_features.append(features)
        else:
            failed += 1
    
    if failed > 0:
        print(f"  {failed} files failed")
    
    if len(all_features) == 0:
        print(f"  ERROR: No features extracted for {lang_name}")
        return None, None
    
    X = np.array(all_features)
    y = np.full(len(all_features), label)
    
    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")
    
    np.save(os.path.join(output_dir, f"{lang_name}_phonetic.npy"), X)
    np.save(os.path.join(output_dir, f"{lang_name}_labels.npy"), y)
    
    return X, y


def main():
    args = parse_args()
    
    # Get experiment configuration
    exp_config = EXP_CONFIGS[args.exp]
    lang_to_label = exp_config["langs"]
    input_subdir = exp_config["input_subdir"]
    
    # Determine output directory based on experiment and features
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    input_dir = data_dir / "sampled_data" / input_subdir
    
    if args.exp == 1:
        if args.features == 7:
            output_dir = data_dir / "processed_phonetic" / "exp1" / "7features"
        else:  # 5 features
            output_dir = data_dir / "processed_phonetic" / "exp1" / "5features"
    else:  # exp 2
        output_dir = data_dir / "processed_phonetic" / "exp2"
    
    # Features description
    if args.features == 7:
        feature_names = "F1, F2, F3, F0, delta_F0, RMS, ZCR"
    else:
        feature_names = "F1, F2, F3, RMS, ZCR"
    
    print("=" * 50)
    print("PHONETIC FEATURES EXTRACTION")
    print("=" * 50)
    print(f"Experiment: {exp_config['name']}")
    print(f"Languages: {', '.join(lang_to_label.keys())}")
    print(f"Features: {args.features} ({feature_names})")
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
        
        X, y = process_language(lang_dir, lang_name, label, output_dir, args.features)
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
    
    # Save combined dataset
    np.save(output_dir / "all_phonetic.npy", X_all)
    np.save(output_dir / "all_labels.npy", y_all)
    
    # Class distribution
    print("\n--- Class distribution ---")
    for lang_name, label in lang_to_label.items():
        count = np.sum(y_all == label)
        print(f"{lang_name}: {count} samples")
    
    print(f"\n✅ Phonetic features extraction complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()