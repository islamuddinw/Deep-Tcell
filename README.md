# Deep-Tcell
====================================================================
 ABOUT THIS PAPER
====================================================================
Title: A Novel Computational Framework for Tumor-Specific T Cell Antigen Identification Using a Deep Neural Network
Authors: Islam Uddin, et al.
Journal: Submitted to Scientific Reports, 2025
Objective: To develop and evaluate a Deep Neural Network (DNN) framework
           that integrates hybrid feature extraction (K-mer, PSSM, PWM,
           ProtBERT) with SHAP-based feature selection to identify tumor
           T-cell antigens. The model is benchmarked against traditional
           machine learning algorithms (SVM, KNN, Random Forest, XGBoost).
Key Contributions:
  - Proposed DNN achieves 97.78% accuracy, 0.982 AUC, 0.94 MCC
  - Demonstrates superiority over SVM, RF, XGBoost, and KNN baselines
  - Provides interpretability using SHAP
  - Addresses class imbalance with SMOTE
  - Offers clinical relevance for in silico screening of neoantigens
    and integration into vaccine design pipelines

====================================================================
 PROJECT README (CODE & REPRODUCIBILITY)
====================================================================
Version: 1.0
Date: 2025-09-06
Repository: https://github.com/islamuddinw/Deep-Tcell
License: MIT
====================================================================

1) QUICK START
--------------------------------------------------------------------
Step 1: Clone the repository
  git clone https://github.com/<your-user>/<your-repo>.git
  cd <your-repo>

Step 2: Create environment (Windows 11 + Anaconda recommended)
  conda create -n ttca python=3.11 -y
  conda activate ttca

Step 3: Install dependencies
  pip install -r requirements.txt

Step 4: Prepare dataset
  - Place your CSV inside the "data" folder
  - Must contain column: label (0 = Non-Tumor, 1 = Tumor)
  - Optional: sequence column OR precomputed features
  - Supported features: PSSM_*, ProtBERT_*, kmer_*, PWM_*

--------------------------------------------------------------------
2) DATASET DETAILS
--------------------------------------------------------------------
- Original: 592 positive and 393 negative sequences
- Balanced with SMOTE (train-only) to 1184 samples (592/592)
- Data split: 80% train, 20% test
- 5-fold stratified cross-validation on training split
- Scaling with StandardScaler applied on train only

--------------------------------------------------------------------
3) DIRECTORY STRUCTURE
--------------------------------------------------------------------
data/                   your_dataset.csv
notebooks/              ttca_dnn_pipeline.ipynb, svm_pipeline.ipynb
scripts/                dnn_pipeline.py, svm_pipeline.py, knn_pipeline.py
outputs/                auto-saved metrics, plots, models
requirements.txt        dependencies
README_Notepad.txt      this file

--------------------------------------------------------------------
4) DNN PIPELINE
--------------------------------------------------------------------
Run Jupyter Notebook:
  jupyter notebook notebooks/ttca_dnn_pipeline.ipynb

Parameters to configure inside the notebook:
  CSV_PATH = data/your_dataset.csv
  LABEL_COL = label
  USE_KMER = True
  USE_PWM = True
  USE_SMOTE = True
  USE_SHAP = True

DNN Architecture:
  Dense(256, ReLU, L2=0.01) -> Dropout(0.5)
  Dense(128, ReLU, L2=0.01) -> Dropout(0.3)
  Dense(64, ReLU, L2=0.01) -> Sigmoid(1)

Training details:
  Optimizer: Adam (lr=0.001)
  Epochs: 50
  Batch size: 32
  Loss: Binary Cross-Entropy
  Regularization: Dropout, L2, EarlyStopping

Outputs:
  cv_fold_metrics.csv, cv_summary.csv
  holdout_metrics.json
  confusion_matrix.png, roc_curve.png
  ttca_dnn_model.h5
  shap_selected_indices.npy

--------------------------------------------------------------------
5) BASELINE MODELS
--------------------------------------------------------------------
All baselines follow same rules: SMOTE train-only, scaling train-only,
5-fold CV on training split, then holdout evaluation.

SVM (RBF):
  python scripts/svm_pipeline.py --csv_path data/your_dataset.csv --label_col label --out_dir ./outputs/svm --smote true --C 1.0 --gamma scale

KNN:
  python scripts/knn_pipeline.py --csv_path data/your_dataset.csv --label_col label --out_dir ./outputs/knn --smote true --n_neighbors 15 --weights distance --metric minkowski --p 2

Outputs (per model):
  cv_fold_metrics.csv, cv_summary.csv
  holdout_metrics.json
  confusion_matrix.png, roc_curve.png
  trained model file (*.joblib)

--------------------------------------------------------------------
6) FEATURES AND SELECTION
--------------------------------------------------------------------
K-mer: 32 dimensions (hashed, k=3,4)
PSSM: 100 dimensions (alignment-based)
ProtBERT: 120 dimensions (pretrained embeddings)
PWM: 100 dimensions (positional histogram)
Fusion: concatenation of all feature blocks
SHAP: KernelExplainer, select top 10 percent features by mean |SHAP|

--------------------------------------------------------------------
7) METRICS AND STATISTICS
--------------------------------------------------------------------
Metrics:
  ACC, SN (Sensitivity), SP (Specificity), AUC, MCC
Statistics:
  Report mean ± SD over 5-fold CV
  Paired t-test or Wilcoxon signed-rank for significance (p < 0.05)

--------------------------------------------------------------------
8) SYSTEM REQUIREMENTS
--------------------------------------------------------------------
Hardware:
  GPU: NVIDIA RTX 3090 (24 GB VRAM)
  CPU: Intel Xeon Silver 4216 @ 2.10 GHz
  RAM: 32 GB
  Storage: 1 TB SSD

Software:
  Windows 11 (64-bit)
  Python 3.11 (Anaconda)
  TensorFlow 2.11 + Keras
  scikit-learn, imbalanced-learn, shap
  numpy, pandas, matplotlib, joblib

--------------------------------------------------------------------
9) REPRODUCIBILITY
--------------------------------------------------------------------
- Fixed random seed: 42
- Dataset schema documented in paper
- All outputs saved to ./outputs/
- Requirements pinned in requirements.txt
- Models saved: DNN .h5, scikit-learn models .joblib
--------------------------------------------------------------------
10) LICENSE AND ACKNOWLEDGMENTS
--------------------------------------------------------------------
License: MIT
Acknowledgment: Research Supporting Project Number RSPD2025R585,
King Saud University, Riyadh, Saudi Arabia
--------------------------------------------------------------------
11) Contact

Maintainer: Islam Uddin (GitHub: islamuddinw)

Issues: please open a ticket in the repository’s Issues tab.

### requirements.txt (paste this as a file in your repo)
```txt
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
imbalanced-learn>=0.12
tensorflow==2.11.*
keras==2.11.*
matplotlib>=3.8
shap>=0.45
joblib>=1.3
jupyter>=1.0


====================================================================
END OF FILE
====================================================================
