# Lightweight Spoken Language Identification

---

## 1. Introduction

This project implements a lightweight spoken language identification (LID) system inspired by TitaNet-LID, with approximately 1.3 million parameters (vs 29M for the original model). The system compares two feature representations:

- **MFCC** (39 features): The standard spectral representation
- **Phonetic features** (5 or 7 features): F1, F2, F3, F0, delta_F0, RMS, ZCR

Two experiments are conducted:

| Experiment | Languages | Samples per language | Goal |
|------------|-----------|---------------------|------|
| **Exp 1** | English, French, German, Mandarin, Spanish | 1200 | Compare MFCC vs phonetic features on diverse languages |
| **Exp 2** | Spanish, Portuguese | 2000 | Test phonetic features on closely related languages |

---

## 2. Data Preparation

### 2.1 Directory Structure

Place your raw audio files in the following structure:

data/
└── raw_data/
├── en/ # English
├── fr/ # French
├── de/ # German
├── zh/ # Mandarin
├── es/ # Spanish
└── pt/ # Portuguese (for Exp 2)



Supported audio formats: `.wav`, `.mp3`, `.flac`

### 2.2 Sampling

Run the data loading script to create balanced datasets:

```bash
# Experience 1: 5 languages, 1200 samples each
python data_loading.py --exp 1 --seed 42

# Experience 2: Spanish vs Portuguese, 2000 samples each
python data_loading.py --exp 2 --seed 42
