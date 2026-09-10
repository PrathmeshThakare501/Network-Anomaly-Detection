# Network-Anomaly-Detection

An AI/ML project investigating deep-learning approaches for **network anomaly detection** using the **CICIDS 2017** dataset.

The project began with a Generative Adversarial Network (GAN) approach and later evaluated a 1D Convolutional Neural Network (CNN) as an alternative discriminative model. The work focuses on practical ML experimentation, large-scale tabular data, class imbalance, computational constraints, and model selection.

## Project at a glance

| Area                       | Details                                                           |
| -------------------------- | ----------------------------------------------------------------- |
| Domain                     | Cybersecurity / Network Intrusion Detection                       |
| Dataset                    | CICIDS 2017                                                       |
| Data type                  | Network-flow / tabular traffic features                           |
| Initial modelling approach | GAN                                                               |
| Alternative model          | 1D CNN                                                            |
| Key challenges             | Class imbalance, high dimensionality, GAN stability, compute cost |
| Compute                    | GAN on GPU server; CNN on local machine                           |
| Main objective             | Detect benign vs anomalous network traffic                        |

## What the project explores

- Cleaning and preparing a multi-million-row network-traffic dataset.
- Handling missing values, infinite values, duplicates, extreme values and non-informative columns.
- Working with an imbalanced cybersecurity classification problem.
- Investigating a GAN with separate Generator and Discriminator networks.
- Exploring **SMOTE**, **PCA**, **stratified sampling** and **cross-validation** during model development.
- Moving computationally intensive GAN experiments from local CPU work to a GPU environment.
- Refactoring the experiment for GPU execution with PyTorch/CUDA support.
- Building a 1D CNN using Conv1D, Batch Normalization, MaxPooling and dense layers.
- Comparing modelling approaches based on the observed behaviour of the data rather than assuming the more complex model is automatically better.

## Data pipeline

```text
CICIDS 2017
    │
    ▼
Data consolidation / preprocessed CSV
    │
    ▼
Duplicate removal
    │
    ▼
Missing + infinite value handling
    │
    ▼
Extreme-value handling
    │
    ▼
Column / string cleanup
    │
    ▼
Remove non-informative columns
    │
    ▼
Select benign / anomalous target
    │
    ├───────────────┐
    ▼               ▼
   GAN             1D CNN
    │
    ├── SMOTE
    ├── PCA experiments
    ├── Stratified sampling
    ├── Cross-validation
    └── GPU training
```

## GAN implementation

The GAN implementation is based on the dissertation experiment and uses **PyTorch**.

### Generator

- Dense layers: `input → 128 → 256 → 512 → 256 → 128 → input`
- Batch Normalization between dense layers
- ReLU hidden activations
- Tanh output

### Discriminator

- Dense layers: `input → 512 → 256 → 128 → 64 → 1`
- Batch Normalization
- LeakyReLU activations
- Dropout (`0.5`)
- Sigmoid output

The training loop uses Adam optimisers, gradient clipping, label smoothing, and **five generator updates per discriminator update**, following the original experiment.

## CNN implementation

The CNN implementation is based on the dissertation architecture and uses **TensorFlow/Keras**.

```text
Input: (features, 1)
   │
Conv1D(64, kernel=6) + ReLU
   │
Batch Normalization
   │
MaxPooling1D
   │
Conv1D(64, kernel=6) + ReLU
   │
Batch Normalization
   │
MaxPooling1D
   │
Conv1D(64, kernel=6) + ReLU
   │
Batch Normalization
   │
MaxPooling1D
   │
Flatten
   │
Dense(64) + ReLU
   │
Dense(64) + ReLU
   │
Softmax output
```

The CNN is trained using Adam and categorical cross-entropy for the binary benign/anomalous task.

## Compute environment

The experiments were split across two environments according to computational requirements:

- **GAN:** executed on a GPU-enabled ML server because adversarial training on the large network-traffic dataset was computationally intensive.
- **CNN:** developed and trained on a local personal machine.

The project therefore also involved moving from CPU-based experimentation to GPU execution and working with a Linux/CUDA-enabled ML environment.

## Dataset

The project uses **CICIDS 2017**, produced by the Canadian Institute for Cybersecurity. The dataset contains labelled network-traffic records covering benign traffic and multiple attack categories.

Because the dataset is large, it is intentionally not stored in this repository. Obtain it from the dataset provider and prepare the expected `preprocessed_data.csv` input before running the scripts.

Citation:

1. Canadian Institute for Cybersecurity (2017). IDS 2017 | Datasets | Research | Canadian Institute for Cybersecurity | UNB. [online] Www.unb.ca. Available at: https://www.unb.ca/cic/datasets/ids-2017.html.
2. Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

## Engineering notes

This project was developed as an applied research exercise rather than a production intrusion-detection system. Several practical constraints shaped the implementation, including:

- large dataset size
- class imbalance
- high-dimensional input data
- GAN training stability
- local CPU limitations
- GPU availability and resource management

A key lesson from the project was that model selection should be driven by the characteristics of the data and the behaviour observed during experimentation, not simply by choosing the most complex architecture.

## Future directions

Potential extensions identified by the research include:

- hybrid GAN + classifier approaches
- stronger handling of imbalanced network data
- explainability using methods such as SHAP or LRP
- richer evaluation beyond accuracy
- real-time anomaly detection
- more efficient deployment for resource-constrained SME environments

## Author

**Prathmesh Thakare**  
Data Science Project
