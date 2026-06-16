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

## Scripts

To run the project, execute the following commands in your terminal in strict sequential order:


### 4.1 Data Sampling

Randomly extract 1200 files from each language for Experience 1, and 2000 files from each language for Experience 2. 

Execute the following scripts: 

python data_loading.py --exp 1 --seed 42  # For Experience 1 

python data_loading.py --exp 2 --seed 42  # For Experience 2



### 4.2  MFCC features extraction
Extract 13 MFCC features using librosa library. 

Execute the following scripts:

python extract_mfcc.py --exp 1  # For Experience 1
    
python extract_mfcc.py --exp 2  # For Experience 2



### 4.3 Phonetic features extraction
Extract 5 or 7 phonetic feature using librosa and parselmouth libraries.

Execute the following scripts:

python extract_phonetic.py --exp 1 --features 7  # For Experience 1 with 7 features
    
python extract_phonetic.py --exp 1 --features 5  # For Experience 1 withouth fundamental frequency F0 and its derivative deltaF0  
    
python extract_phonetic.py --exp 2 --features 5  # For Experience 2


### 4.4 Data split
Split data into train, validation, and test sets (repartition is 70-15-15).

Execute the following scripts: 

python split_data.py --features mfcc --exp 1  # For Experience 1 with MFCC features

python split_data.py --features mfcc --exp 2  # For Experience 2 with MFCC features

python split_data.py --features phonetic --exp 1 --subfolder 7features # For Experience 1 with 7 phonetic features

python split_data.py --features phonetic --exp 1 --subfolder 5features # For Experience 1 with 5 phonetic features

python split_data.py --features phonetic --exp 2  For Experience 2 with 5 phonetic features


### 4.5 Model training 












