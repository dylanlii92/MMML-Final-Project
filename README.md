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
## 2. Requirements
This project can be run with any version of Python >= 3.10. 
Before running any script, install the required Python packages:

pip install numpy librosa torch torchvision scikit-learn matplotlib seaborn tqdm parselmouth praat-parselmouth

---


## 3. Data Preparation

### 3.1 Dataset

We use the **VoxLingua107** dataset ([Valk and Alumäe, SLT 2021](https://arxiv.org/abs/2011.12998)), which contains YouTube speech data for 107 languages with approximately 6628 hours of training data. Individual language archives can be downloaded from:

[https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/)

### 3.2 Directory Structure

Place your downloaded audio files in the following structure:
data/
└── raw_data/
├── en/ # English (en.zip)
├── fr/ # French (fr.zip)
├── de/ # German (de.zip)
├── zh/ # Mandarin Chinese (zh.zip)
├── es/ # Spanish (es.zip)
└── pt/ # Portuguese (pt.zip)

**Note**: Extract the downloaded zip files directly into the corresponding language folders.

---

## 4. Data Sampling

For computational reasons, randomly extract 1200 files from each language for Experience 1, and 2000 files from each language for Experience 2. 

Execute the following scripts: 

python data_loading.py --exp 1 --seed 42  # For Experience 1 
python data_loading.py --exp 2 --seed 42  # For Experience 2

---

## 5. MFCC features extraction






