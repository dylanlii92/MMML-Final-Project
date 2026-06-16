"""
Unified training script for TitaNet-inspired model.
Supports MFCC and phonetic features, Experiment 1 and Experiment 2.

Usage:
    # MFCC - Experience 1 (5 languages)
    python train_model.py --features mfcc --exp 1
    
    # MFCC - Experience 2 (Spanish vs Portuguese)
    python train_model.py --features mfcc --exp 2
    
    # Phonetic 7 features - Experience 1
    python train_model.py --features phonetic --exp 1 --subfolder 7features
    
    # Phonetic 5 features - Experience 1
    python train_model.py --features phonetic --exp 1 --subfolder 5features
    
    # Phonetic 5 features - Experience 2
    python train_model.py --features phonetic --exp 2
    
    # With normalization (for phonetic features)
    Add --normalize at the end of the previous commands
    
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse

from model import TitaNetInspired, count_parameters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TitaNet-inspired model")
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
        help="Subfolder for phonetic features (e.g., 7features, 5features)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply StandardScaler normalization to features"
    )
    return parser.parse_args()


def get_config(features, exp, subfolder=None):
    """Get configuration based on arguments."""
    if features == "mfcc":
        input_dim = 13
        data_dir = Path("data") / "processed_mfcc" / f"exp{exp}"
        model_name = f"mfcc_exp{exp}"
    else:  # phonetic
        if exp == 1:
            if subfolder is None:
                raise ValueError("--subfolder required for exp 1 (e.g., 7features or 5features)")
            data_dir = Path("data") / "processed_phonetic" / "exp1" / subfolder
            input_dim = 7 if subfolder == "7features" else 5
            model_name = f"phonetic_exp1_{subfolder}"
        else:  # exp 2
            data_dir = Path("data") / "processed_phonetic" / "exp2"
            input_dim = 5
            model_name = "phonetic_exp2"
    
    return data_dir, input_dim, model_name


def get_language_names(exp):
    """Get language names for plotting."""
    if exp == 1:
        return ['English', 'French', 'German', 'Mandarin', 'Spanish']
    else:
        return ['Spanish', 'Portuguese']


def get_num_languages(exp):
    """Get number of languages."""
    return 5 if exp == 1 else 2


def normalize_data(X_train, X_val, X_test):
    """Apply StandardScaler normalization."""
    n_samples_train, n_frames, n_feats = X_train.shape
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(
        X_train.reshape(-1, n_feats)
    ).reshape(n_samples_train, n_frames, n_feats)
    
    X_val_norm = scaler.transform(
        X_val.reshape(-1, n_feats)
    ).reshape(X_val.shape)
    
    X_test_norm = scaler.transform(
        X_test.reshape(-1, n_feats)
    ).reshape(X_test.shape)
    
    return X_train_norm, X_val_norm, X_test_norm


def main():
    args = parse_args()
    
    # Get configuration
    data_dir, input_dim, model_name = get_config(
        args.features, args.exp, args.subfolder
    )
    lang_names = get_language_names(args.exp)
    num_languages = get_num_languages(args.exp)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    print(f"Features: {args.features}")
    print(f"Experiment: {args.exp}")
    if args.subfolder:
        print(f"Subfolder: {args.subfolder}")
    print(f"Input dim: {input_dim}")
    print(f"Languages: {num_languages}")
    print(f"Normalization: {args.normalize}")
    print(f"Data directory: {data_dir}")
    print(f"Device: {DEVICE}")
    print()
    
    # Check data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory '{data_dir}' does not exist!")
        print("Please run feature extraction and split first.")
        return
    
    # Load data
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Input shape: {X_train.shape[1:]}")

    # Normalize if requested
    if args.normalize:
        print("\n" + "=" * 50)
        print("NORMALIZATION")
        print("=" * 50)
        X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
        print("Features normalized (mean=0, std=1)")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # DataLoaders
    BATCH_SIZE = 32
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("\n" + "=" * 50)
    print("MODEL INITIALIZATION")
    print("=" * 50)
    
    model = TitaNetInspired(
        input_dim=input_dim,
        num_languages=num_languages,
        num_blocks=3,
        num_repeats=2,
        channels=128,
        kernel_sizes=[7, 11, 15],
        dropout=0.3
    ).to(DEVICE)
    
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model size: {count_parameters(model) * 4 / 1024:.1f} KB")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    
    EPOCHS = 50
    PATIENCE = 10
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_{model_name}.pt")
            print(f"  --> New best model saved! (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Test evaluation
    print("\n" + "=" * 50)
    print("TEST EVALUATION")
    print("=" * 50)
    
    model.load_state_dict(torch.load(f"best_model_{model_name}.pt"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    test_acc = 100 * correct / total
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=lang_names, yticklabels=lang_names, cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix (Test Acc: {test_acc:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    
    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=lang_names))
    
    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Learning Curves - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'learning_curves_{model_name}.png')
    plt.close()
    
    print(f"\n✅ Graphs saved:")
    print(f"   - confusion_matrix_{model_name}.png")
    print(f"   - learning_curves_{model_name}.png")
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Features: {args.features}")
    print(f"Input dim: {input_dim}")
    print(f"Normalization: {args.normalize}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Languages: {', '.join(lang_names)}")


if __name__ == "__main__":
    main()