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

This data folder contains the TAD boundary information for different cell lines as well as the TAD region data of the independent test set. To ensure the effective utilization of the data, you need to follow the following steps for data processing and conversion.
1. Convert TAD Regions to TAD Boundaries

For the independent test set containing TAD region data, use the TAD_change_Boundaries.py script to convert the TAD region data into TAD boundary information.
```bash
python TAD_change_Boundaries.py
```
2. One-hot Encoding to Generate Final Input Data
   
All TAD boundary information should be processed using the one_hot.py script to apply one-hot encoding, generating the final input files. This step converts the boundary information into a format suitable for model input, producing .npy files.
Command example:
```bash
python one_hot.py
```
3.After one-hot encoding, the resulting .npy files will serve as the model input data for further analysis and training.

Download bigwig files of histone modification signals and transcription factor binding site signals (CTCF、H2A.Z、H3k9ac、H3k4me3、H3k4me2、H3k9me3、H3k27me3、H3k36me3、H3k27ac、H3k79me2、H3k4me1、H4k20me1﻿) from the ENCOD database

### 4. Train the CT-TADB model

```bash
python Model.py
```

### 5. Evaluate the trained model

```bash
python Test.py
```

### 6. Output Result Analysis  
For example, as shown in the figure, for the position indicated by the arrow within the region chr8:29,000,000 - 30,000,000, the model weight of GM12878 predicts that the probability of this region being a TAD boundary is 0.895; The K562 model predicts that the probability of this region being a TAD boundary is only 0.256. The prediction results are consistent with the Hi-C results.

## Model Architecture

CT-TADB include:

- The input layer receives a 10,000 bp DNA sequence encoded as a one-hot matrix and 12 modification signals.
- Two convolutional layers, kernel size = 9, num_kernels = 64
- Each convolutional layer is followed by a ReLU activation function and a max-pooling layer.
- A multi-head attention mechanism is employed. num_heads = 4, ff_dim = Transformer_fim = 32
- An output layer with a sigmoid activation function computes boundary probabilities.
- Experiments were conducted on NVIDIA A800-SXM4-80GB (CUDA 12.2) and NVIDIA GeForce RTX 3090 (CUDA 11.7) platforms. 

Under the default 10 kb input setting with batch size = 16, each training epoch requires approximately 1 min 47 s, and the complete training process takes approximately 1 hour on an NVIDIA GeForce RTX 3090 (24 GB GPU memory). Peak GPU memory usage was monitored using nvidia-smi. 

## File Structure

```
CT-TADB/
├── README.md
├── CT-TADB_Model.py
├── Test.py
├── SHAPE.py
├── deeplift.py
├── Model.hdf5
├── fimo.txt
├── data
    ├── GM12878_manually_annotated_TADs.bed
    ├── GM12878_TAD_boundaries.bed
    ├── HMEC_TAD_boundaries.bed
    ├── HUVEC_TAD_boundaries.bed
    ├── IMR90_TAD_boundaries.bed
    ├── K562_TAD_boundaries.bed
    └── NHEK_TAD_boundaries.bed
    └── TAD_change_Boundaries.py
    └── one_hot.py
```
