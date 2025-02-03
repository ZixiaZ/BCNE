# Brain-dynamic Convolutional-Network-based Embedding (BCNE)

This repository contains the implementation of BCNE (Brain-dynamic Convolutional-Network-based Embedding), a self-supervised manifold learning method for analyzing dynamic brain data. BCNE is designed to reveal neurocognitive and behavioral patterns through dimensionality reduction and visualization of complex neural recordings.

## Overview

BCNE is a novel approach that combines convolutional neural networks with manifold learning to analyze time-varying neural activity patterns. The method has been successfully applied to three different types of neural recordings:

1. Human fMRI data (Sherlock dataset)
2. Rat hippocampal recordings
3. Monkey Area2 recordings

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- SciPy
- Pandas
- Matplotlib
- scikit-learn


## Usage

### Training Models

```python
from train_model import main_sherlock, main_rat, main_monkey

# For Sherlock fMRI data
main_sherlock(subject_num=1, n_components=2, ROI_number=0, recur=4, 
              balance=4, train_mode=0, round_seed=0)

# For rat hippocampal data
main_rat(rat_number=0, n_components=2, recur=4, balance=3, 
         train_mode=0, round_seed=0)

# For monkey motor cortex data
main_monkey(mode_number=0, n_components=2, recur=4, balance=1, 
            train_mode=0, round_seed=0)
```

### Testing Models

```python
from test_model import test_sherlock, test_rat, test_monkey

# Test trained models
test_sherlock(subject_num=1, ROI_number=0, balance=4, round_seed=0)
test_rat(rat_number=0, balance=3, round_seed=0)
test_monkey(mode_number=0, balance=1, round_seed=0)
```

### Evaluation

```python
from evaluations import evaluate_sherlock, evaluate_rat, evaluate_monkey

# Evaluate model performance
evaluate_sherlock(subject_num=1, ROI_number=0, balance=4, round_seed=0)
evaluate_rat(rat_number=0, balance=3, round_seed=0)
evaluate_monkey(mode_number=0, balance=1, round_seed=0)
```

## Key Features

1. **Self-Supervised Learning**: BCNE learns meaningful representations without requiring labeled data.

2. **Spatiotemporal Processing**:
   - Spatial projection using optimal transport
   - Temporal correlation analysis
   - Recursive manifold learning

3. **Multi-Scale Analysis**:
   - Support for different temporal scales
   - Adaptive batch processing
   - Multiple recursion levels

4. **Evaluation Metrics**:
   - KNN classification accuracy
   - Pearson correlation analysis
   - Trustworthiness and continuity measures
   - Event segmentation analysis

## Model Architecture

The BCNE model consists of:

1. **Convolutional Layers**: Process spatial patterns in neural data
2. **Dense Layers**: Transform features into low-dimensional embeddings
3. **Manifold Learning Component**: Preserves data structure during dimensionality reduction

## Parameters

Key parameters for model training:

- `n_components`: Dimensionality of the embedding (default: 2)
- `recur`: Number of recursive iterations (default: 4)
- `balance`: Batch size balancing factor
- `train_mode`: Training strategy (0: patient strategy, 1: fixed epochs)
- `round_seed`: Random seed for reproducibility


## Contact

For questions and feedback, please contact Zixia Zhou (zixia@stanford.edu).
