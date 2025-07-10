# CT-TADB: A Method for Predicting TAD Boundaries Using CNN and Transformer Frameworks

## Introduction

CT-TADB is a deep learning framework that integrates convolutional neural networks with attention mechanisms, using DNA sequences, histone modification signals, and transcription factor binding site signals as inputs to predict TAD boundaries with high efficiency and accuracy. The CT-TADB model demonstrates that sequence-encoded information, histone modification signals, and key transcription factor binding sites largely determine chromatin architecture, providing a practical tool and conceptual framework for understanding three-dimensional genome organization.

## Key Features

- **Input**: Integrates histone modification signals and transcription factor binding site information alongside DNA sequence inputs.
- **Cell-Type-Specific Prediction**: Supports predictions across six human cell lines (IMR90, NHEK, K562, HUVEC, HMEC, GM12878).
- **Superior Performance**: Achieves outstanding accuracy and discriminative ability on independent test sets.
- **Cross-Cell-Line and Cross-Species Generalizability**: Maintains high predictive performance across diverse cellular contexts and species.
- **Interpretable Results**: Attention mechanisms reveal critical sequence features and transcription factor motifs.
- **Motif Discovery**: Identifies multiple human-associated transcription factors, including MEF2A, NFATC4, NFATC1, PLAGL1, HIC2, and FoxE.

## Steps to Install and Run CT-TADB

### 1. Clone the CT-TADB repository

```bash
git clone https://github.com/Tong-Chen-ct/CT-TADB.git
cd CT-TADB
```

### 2. Install the required dependencies

```
tensorflow>=2.4.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
biopython>=1.78
pybedtools>=0.8.0
matplotlib>=3.3.0
```

### 3. Prepare your input data

The data folder contains TAD boundaries for various cell lines and TAD region data for independent test sets. It is necessary to convert TAD regions into TAD boundaries.

Download bigwig files of histone modification signals and transcription factor binding site signals (CTCF、H2A.Z、H3k9ac、H3k4me3、H3k4me2、H3k9me3、H3k27me3、H3k36me3、H3k27ac、H3k79me2、H3k4me1、H4k20me1﻿) from the ENCOD database

### 4. Train the CT-TADB model

```bash
python Model.py
```

### 5. Evaluate the trained model

```bash
python Test.py
```

## Model Architecture

CT-TADB include:

- The input layer receives a 10,000 bp DNA sequence encoded as a one-hot matrix and 12 modification signals.
- Two convolutional layers, kernel size = 9, num_kernels = 64
- Each convolutional layer is followed by a ReLU activation function and a max-pooling layer.
- A multi-head attention mechanism is employed. num_heads = 4, ff_dim = Transformer_fim = 32
- An output layer with a sigmoid activation function computes boundary probabilities.

## File Structure

```
CT-TADB/
├── README.md
├── CT-TADB_Model.py
├── Test.py
├── SHAPE.py
├── deeplift.py
├── Model.hdf5
├── data
├── DNA
    ├── GM12878_manually_annotated_TADs.bed
    ├── GM12878_TAD_boundaries.bed
    ├── HMEC_TAD_boundaries.bed
    ├── HUVEC_TAD_boundaries.bed
    ├── IMR90_TAD_boundaries.bed
    ├── K562_TAD_boundaries.bed
    └── NHEK_TAD_boundaries.bed
```
