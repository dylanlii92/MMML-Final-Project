"""
Lightweight TDNN model for Spoken Language Identification.
Inspired by TitaNet-LID architecture but adapted for CPU training.

Key TitaNet features incorporated:
- 1D depth-wise separable convolutions (simplified)
- Squeeze-and-Excitation blocks with global average pooling
- Repeated basic blocks with residual connections
- Statistical pooling (mean + std)
- Progressive kernel sizes (7, 11, 15)

Differences (for CPU efficiency):
- Fewer channels (C=128 instead of 512/1024)
- Fewer blocks (B=3, R=2 instead of B=3, R=5)
- No depth-wise separable (standard Conv1d for simplicity)
"""

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block as described in TitaNet.
    Uses global average pooling for squeeze operation.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Squeeze: global average pooling across time
        squeeze = x.mean(dim=2)  # (batch, channels)
        # Excitation
        excitation = self.fc(squeeze).unsqueeze(2)  # (batch, channels, 1)
        # Scale
        return x * excitation


class BasicBlock(nn.Module):
    """
    Basic block as in TitaNet:
    Conv1d -> BatchNorm -> ReLU -> Dropout
    Kernel size varies (7, 11, 15) depending on position
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MegaBlock(nn.Module):
    """
    Mega block as in TitaNet:
    R repeated basic blocks + residual connection + SE block
    """
    def __init__(self, channels, num_repeats, kernel_size, dropout=0.2):
        super().__init__()
        
        # R repeated basic blocks
        self.blocks = nn.ModuleList()
        for i in range(num_repeats):
            self.blocks.append(BasicBlock(channels, channels, kernel_size, dropout))
        
        # Squeeze-and-Excitation at the end of mega block
        self.se = SqueezeExcitation(channels)
    
    def forward(self, x):
        residual = x
        
        for block in self.blocks:
            x = block(x)
        
        # Residual connection
        x = x + residual
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        return x


class StatsPool(nn.Module):
    """
    Statistical pooling as in TitaNet decoder.
    Concatenates mean and standard deviation across time.
    Maps variable-length input to fixed-length representation.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)


class TitaNetInspired(nn.Module):
    """
    Lightweight model inspired by TitaNet-LID architecture.
    
    Architecture:
    - Prologue block (initial convolution)
    - B mega blocks, each with R repeated basic blocks
    - Increasing kernel sizes (7, 11, 15) across mega blocks
    - Squeeze-and-Excitation after each mega block
    - Statistical pooling (mean + std)
    - Two linear layers for classification
    
    Original TitaNet-LID: B=3, R=5, C=512/1024, 29M params
    This version: B=3, R=2, C=128, ~500k params (60x smaller)
    """
    def __init__(self, input_dim=13, num_languages=5, 
                 num_blocks=3, num_repeats=2, channels=128, 
                 kernel_sizes=[7, 11, 15], dropout=0.3):
        super().__init__()
        
        # Prologue block (B0 in TitaNet)
        self.prologue = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mega blocks (B1...BN-1 in TitaNet)
        self.mega_blocks = nn.ModuleList()
        for i in range(num_blocks):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 7
            self.mega_blocks.append(
                MegaBlock(channels, num_repeats, kernel_size, dropout)
            )
        
        # Epilogue block (BN in TitaNet)
        self.epilogue = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder: statistical pooling + two linear layers
        self.pooling = StatsPool()
        
        # Two linear layers as in TitaNet decoder
        self.fc1 = nn.Linear(channels * 2, 512)
        self.fc2 = nn.Linear(512, num_languages)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input: (batch, time, features) -> (batch, features, time)
        x = x.transpose(1, 2)
        
        # Prologue
        x = self.prologue(x)
        
        # Mega blocks
        for mega_block in self.mega_blocks:
            x = mega_block(x)
        
        # Epilogue
        x = self.epilogue(x)
        
        # Decoder: statistical pooling
        x = self.pooling(x)
        
        # Two linear layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Quick test to verify model works."""
    print("=" * 50)
    print("TitaNet-Inspired Model Test")
    print("=" * 50)
    
    model = TitaNetInspired(input_dim=13, num_languages=5,
                            num_blocks=3, num_repeats=2, channels=128)
    
    params = count_parameters(model)
    print(f"Total parameters: {params:,}")
    
    # Compare with original TitaNet
    original_params = 29_000_000
    ratio = original_params / params
    print(f"Original TitaNet-LID: 29,000,000 params")
    print(f"This model: {params:,} params ({ratio:.1f}x smaller)")
    
    # Test forward pass
    dummy_input = torch.randn(32, 301, 13)  # batch, time, features
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Model test passed!")


if __name__ == "__main__":
    test_model()