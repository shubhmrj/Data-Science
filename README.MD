# 📊 Data Science & Machine Learning — Complete Learning Repository

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-yellow?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-lightblue?logo=numpy)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-green?logo=scikitlearn)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-red)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Visualization-Seaborn-teal)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/shubhmrj/Data-Science?style=social)](https://github.com/shubhmrj/Data-Science)
[![GitHub Forks](https://img.shields.io/github/forks/shubhmrj/Data-Science?style=social)](https://github.com/shubhmrj/Data-Science/fork)

---

## 📋 Table of Contents

| Section | Description |
|---|---|
| [Description](#-description) | Project overview, goals, and key outcomes |
| [Installation Instructions](#-installation-instructions) | Setup guide for local development |
| [Data Sources](#-data-sources) | Datasets used and preprocessing details |
| [Code Structure](#-code-structure) | File organization and directory guide |
| [Mathematical Foundations](#-mathematical-foundations) | Core math — linear algebra, statistics, calculus |
| [Data Science Practicals](#-data-science-practicals) | EDA, preprocessing, and feature engineering |
| [Machine Learning Models](#-machine-learning-models) | Supervised, unsupervised, and semi-supervised models |
| [Advanced Topics](#-advanced-topics) | Deep learning, NLP, time series |
| [Results & Evaluation](#-results--evaluation) | Model performance and key metrics |
| [Usage](#-usage) | How to run experiments and notebooks |
| [Key Formulas Reference](#-key-formulas-reference) | Quick-reference formula table |
| [Recommended Learning Path](#-recommended-learning-path) | Structured 13-week curriculum |
| [Future Work](#-future-work) | Planned enhancements and roadmap |
| [Additional Resources](#-additional-resources) | Courses, books, datasets, and communities |
| [Acknowledgments & References](#-acknowledgments--references) | Credits and attributions |
| [License](#-license) | Licensing information |
| [Connect & Contribute](#-connect--contribute) | Contact, contributions, and support |

---

## 📝 Description

Welcome to the **Data Science & Machine Learning Complete Learning Repository** — a comprehensive, hands-on resource that bridges mathematical theory with real-world practice. Hosted at [github.com/shubhmrj/Data-Science](https://github.com/shubhmrj/Data-Science), this repository is designed to serve as both a structured learning curriculum and a practical reference for data science practitioners at every level.

### Goals

This repository addresses a common challenge in data science education: the disconnect between theoretical knowledge and practical implementation. Most learning resources either stay too abstract (pure mathematics) or too shallow (code-first with no theory). This project unifies both approaches into a single, cohesive ecosystem.

### What It Covers

The repository spans the full data science pipeline — from foundational mathematics (linear algebra, probability, calculus) through data engineering, exploratory analysis, classical machine learning, and advanced deep learning topics. Five end-to-end project implementations (house price prediction, customer segmentation, fraud detection, sentiment analysis, and stock forecasting) demonstrate how individual concepts connect into production-ready workflows.

### Key Outcomes

- A structured **13-week learning path** from beginner to advanced practitioner level.
- Comprehensive **Jupyter notebook implementations** for 20+ machine learning algorithms.
- A **formula reference library** covering the essential mathematics underpinning every model.
- End-to-end **project templates** for the most common data science problem types.
- Curated **resource lists** (courses, books, datasets, communities) to support continued growth.

---

## ⚙️ Installation Instructions

### Prerequisites

Before cloning the repository, ensure the following tools are installed on your system.

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core runtime |
| pip or conda | Latest | Package management |
| Git | Latest | Version control |
| Jupyter Notebook / JupyterLab | Latest | Interactive notebooks |
| VS Code / PyCharm | Latest | Recommended IDE |

Verify Python installation with:
```bash
python --version
```

### Step 1 — Clone the Repository

```bash
git clone https://github.com/shubhmrj/Data-Science.git
cd Data-Science
```

### Step 2 — Create and Activate a Virtual Environment

Using a virtual environment isolates this project's dependencies from your system Python installation.

```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```bash
# Install all required packages from the lockfile
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Manual Installation by Stack

If you prefer to install packages selectively, the dependencies are grouped by use case below.

**Core Data Science Stack (Required)**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

**Advanced Analytics**
```bash
pip install scipy statsmodels xgboost lightgbm catboost
```

**Deep Learning (Optional)**
```bash
pip install tensorflow keras torch torchvision
```

**Natural Language Processing (Optional)**
```bash
pip install nltk spacy textblob transformers
```

**Time Series (Optional)**
```bash
pip install statsmodels pmdarima prophet
```

> **Note**: If you encounter version conflicts, using `conda` instead of `pip` is recommended for managing complex scientific computing environments. Create a conda environment with `conda create -n ds-env python=3.10` and activate it with `conda activate ds-env`.

---

## 🗄️ Data Sources

This repository does not rely on a single proprietary dataset. Instead, it draws from well-established public repositories to ensure reproducibility and accessibility.

### Primary Sources

| Source | URL | Description |
|--------|-----|-------------|
| Kaggle Datasets | [kaggle.com/datasets](https://www.kaggle.com/datasets) | Project datasets for house prices, fraud detection, and sentiment analysis |
| UCI ML Repository | [archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml) | Classic benchmark datasets for algorithm demonstrations |
| Yahoo Finance / yfinance | [pypi.org/project/yfinance](https://pypi.org/project/yfinance/) | Historical stock data for the time series forecasting project |
| Scikit-Learn Built-ins | [scikit-learn.org/datasets](https://scikit-learn.org/stable/datasets.html) | Iris, Boston Housing, Digits — used in EDA and preprocessing notebooks |

### Project-Specific Datasets

| Project | Dataset | Source |
|---------|---------|--------|
| House Price Prediction | Ames Housing Dataset | Kaggle |
| Customer Segmentation | Mall Customer Dataset | Kaggle |
| Fraud Detection | IEEE-CIS Fraud Detection | Kaggle |
| Sentiment Analysis | IMDB Movie Reviews | Kaggle / HuggingFace |
| Stock Forecasting | S&P 500 Historical Prices | yfinance API |

### Preprocessing Pipeline

All raw datasets undergo a standardised preprocessing workflow before use in models. The pipeline is documented in detail within the `02_preprocessing/` notebooks and includes the following stages.

**Data Quality:** Missing values are handled using mean/median imputation for numerical features and mode imputation for categorical features. KNN imputation is used for datasets with complex missingness patterns. Outliers are identified using the IQR method (threshold: Q3 + 1.5×IQR) and Z-score method (threshold: |z| > 3), with treatment determined by the specific business context of each project.

**Encoding:** Categorical variables are transformed using One-Hot Encoding for nominal features and Ordinal Encoding for ordered categories. Target encoding is applied in specific high-cardinality scenarios to avoid dimensionality explosion.

**Scaling:** Numerical features are standardised using Z-score normalisation (StandardScaler) or Min-Max scaling, depending on the algorithm's sensitivity to feature magnitude.

**Train-Test Split:** All datasets are split 80/20 (training/testing) with stratification applied for classification tasks to preserve class distribution.

---

## 🗂️ Code Structure

The repository is organised into clearly delineated directories, each corresponding to a stage of the data science workflow.

```
📁 Data-Science/
│
├── 📁 datasets/                        # Data assets
│   ├── raw/                            # Original, unmodified source files
│   └── processed/                      # Cleaned & feature-engineered files
│
├── 📁 fundamentals/                    # Mathematical foundations
│   ├── linear_algebra/                 # Vectors, matrices, eigendecomposition
│   ├── probability_statistics/         # Distributions, hypothesis testing
│   ├── calculus_optimization/          # Derivatives, gradient descent
│   └── mathematics_notes.ipynb         # Consolidated theory reference
│
├── 📁 01_eda/                          # Exploratory Data Analysis
│   ├── univariate_analysis.ipynb       # Distributions, histograms, box plots
│   ├── bivariate_analysis.ipynb        # Correlation, scatter plots, chi-square
│   └── multivariate_analysis.ipynb     # Heatmaps, pairplots, PCA visualisation
│
├── 📁 02_preprocessing/                # Data cleaning & feature engineering
│   ├── missing_values.ipynb
│   ├── outlier_detection.ipynb
│   ├── feature_scaling.ipynb
│   ├── feature_encoding.ipynb
│   └── feature_selection.ipynb
│
├── 📁 03_models/                       # ML algorithm implementations
│   ├── 📁 regression/
│   │   ├── linear_regression.ipynb
│   │   ├── polynomial_regression.ipynb
│   │   ├── regularization.ipynb        # Ridge, Lasso, ElasticNet
│   │   └── advanced_regression.ipynb
│   ├── 📁 classification/
│   │   ├── logistic_regression.ipynb
│   │   ├── decision_trees.ipynb
│   │   ├── ensemble_methods.ipynb      # Random Forest, Gradient Boosting
│   │   ├── svm.ipynb
│   │   └── naive_bayes.ipynb
│   └── 📁 unsupervised/
│       ├── clustering.ipynb            # K-Means, Hierarchical, DBSCAN
│       ├── dimensionality_reduction.ipynb  # PCA, t-SNE, UMAP
│       └── anomaly_detection.ipynb
│
├── 📁 04_advanced/                     # Advanced ML topics
│   ├── deep_learning_basics.ipynb      # Neural nets, backpropagation
│   ├── nlp_basics.ipynb                # Tokenisation, TF-IDF, Word2Vec, BERT
│   ├── time_series.ipynb               # ARIMA, exponential smoothing, LSTM
│   └── reinforcement_learning.ipynb
│
├── 📁 05_projects/                     # End-to-end applied projects
│   ├── 📁 project_1_house_price_prediction/
│   ├── 📁 project_2_customer_segmentation/
│   ├── 📁 project_3_fraud_detection/
│   ├── 📁 project_4_sentiment_analysis/
│   └── 📁 project_5_stock_forecasting/
│
├── 📁 06_notes/                        # Theory summaries and cheat sheets
│   ├── ml_algorithms_summary.md
│   ├── statistical_concepts.md
│   ├── common_pitfalls.md
│   └── quick_reference.md
│
├── requirements.txt                    # Pinned Python dependencies
├── README.md                           # This file
└── LICENSE                             # MIT License
```

**Key file types used throughout the repository:**
- `.ipynb` — Jupyter Notebooks containing code, outputs, and narrative explanations.
- `.md` — Markdown documents for theory notes, algorithm summaries, and reference guides.
- `.csv` / `.parquet` — Tabular datasets in the `datasets/` directory.
- `requirements.txt` — Pinned dependency versions ensuring reproducibility across environments.

---

## 🧮 Mathematical Foundations

A solid understanding of the following mathematical disciplines is essential for interpreting and implementing machine learning algorithms correctly.

### 1️⃣ Linear Algebra

Linear algebra provides the structural language for machine learning models, from representing datasets as matrices to computing transformations and decompositions.

**Vectors and Matrices** form the basic data structures. A vector **v** = [v₁, v₂, ..., vₙ] represents a point or direction in n-dimensional space. A matrix **A** = [[a₁₁, a₁₂], [a₂₁, a₂₂]] represents linear transformations and multi-dimensional datasets.

**Key operations and concepts:**
- **Matrix Multiplication**: Combines two compatible matrices — an (m×n) matrix times an (n×p) matrix yields an (m×p) matrix.
- **Determinant** (det): A scalar value indicating whether a matrix is invertible; a determinant of zero signals linear dependence.
- **Eigenvalues & Eigenvectors**: For matrix **A**, if **Av** = λ**v**, then **v** is an eigenvector and λ is the corresponding eigenvalue. Foundational to PCA and many decomposition methods.
- **Rank**: The number of linearly independent rows or columns; determines the information content of a matrix.

**Applications in ML:** Dimensionality reduction (PCA), matrix factorisation for recommendation systems, weight matrices in neural networks.

---

### 2️⃣ Probability & Statistics

**Key Distributions:**

| Distribution | Formula | Common Use Case |
|---|---|---|
| **Normal** | f(x) = (1/σ√2π) × exp(-(x-μ)²/2σ²) | Modelling natural phenomena |
| **Binomial** | P(X=k) = C(n,k) × pᵏ(1-p)ⁿ⁻ᵏ | Binary outcomes (coin flips, A/B tests) |
| **Poisson** | P(X=k) = (e^(-λ) × λᵏ)/k! | Counting events in fixed intervals |
| **Exponential** | f(x) = λe^(-λx) | Modelling time between events |

**Core Statistical Measures:**

| Measure | Formula | Interpretation |
|---|---|---|
| Mean (μ) | μ = (Σx)/n | Central tendency |
| Variance (σ²) | σ² = E[(X - μ)²] | Spread around the mean |
| Standard Deviation (σ) | σ = √(σ²) | Spread in original units |
| Covariance | Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] | Direction of linear relationship |
| Correlation (ρ) | ρ = Cov(X,Y)/(σₓσᵧ) | Normalised relationship, range: [-1, 1] |

**Hypothesis Testing** governs statistical inference. The null hypothesis (H₀) is the assumed baseline. A p-value below the significance level (α = 0.05) leads to rejection of H₀, indicating a statistically significant finding.

---

### 3️⃣ Calculus & Optimization

Optimization is the engine of machine learning training. The goal is to find model parameters that minimise a cost function.

**Gradient Descent** is the foundational optimisation algorithm:
```
θₜ₊₁ = θₜ - α∇J(θ)
```
Where θ represents model parameters, α is the learning rate, and ∇J(θ) is the gradient of the cost function. Three common variants exist: Batch GD (full dataset per step), Stochastic GD (one sample per step), and Mini-Batch GD (a subset per step).

**Advanced Optimisers** build on this foundation. Momentum accelerates gradient descent by accumulating a velocity vector. RMSprop adapts learning rates per parameter. Adam combines both momentum and adaptive learning rates and remains the most widely used optimiser in practice.

---

## 📊 Data Science Practicals

### 1️⃣ Exploratory Data Analysis (EDA)

EDA is the critical first step in any data science workflow. It reveals distributions, identifies anomalies, surfaces relationships between variables, and informs all subsequent modelling decisions.

**Univariate Analysis** examines each feature in isolation. Numerical features are explored with histograms, density plots, and box plots, with skewness (γ = E[(X-μ)³]/σ³) and kurtosis (κ = E[(X-μ)⁴]/σ⁴ - 3) used to characterise distribution shape. Categorical features are examined with frequency tables and bar charts.

**Bivariate Analysis** studies pairwise relationships. Pearson (r), Spearman (ρ), and Kendall (τ) correlation coefficients quantify linear and monotonic relationships between numerical features. Chi-square tests assess independence between categorical variables.

**Multivariate Analysis** examines the full feature space simultaneously. Correlation heatmaps highlight redundancy, pairplots reveal all pairwise relationships at once, and PCA projections visualise high-dimensional structure in two or three dimensions.

---

### 2️⃣ Feature Engineering

Feature engineering transforms raw data into representations that machine learning algorithms can learn from effectively.

**Feature Scaling** ensures that features with large numerical ranges do not dominate those with smaller ranges.
```
Standardisation (Z-score):   x' = (x - μ) / σ
Min-Max Normalisation:        x' = (x - min) / (max - min)
Robust Scaling:               x' = (x - median) / IQR
```

**Feature Encoding** converts categorical variables into numerical representations that algorithms can process. One-Hot Encoding creates binary indicator columns for each category. Ordinal Encoding preserves natural ordering. Target Encoding replaces a category with the mean of the target variable for that category, which is effective for high-cardinality features.

**Feature Creation** enriches the feature space through domain knowledge and mathematical transformations: polynomial features (x², x³), interaction terms (x₁ × x₂), and binning of continuous variables into meaningful categories.

**Feature Selection** reduces dimensionality and removes noise. Methods include correlation-based filtering (removing highly collinear features), variance thresholding, tree model importance scores, and Recursive Feature Elimination (RFE).

---

## 🤖 Machine Learning Models

### 1️⃣ Supervised Learning — Regression

**Linear Regression** models a continuous target as a linear combination of input features.
```
Hypothesis:     ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
Cost Function:  J(θ) = (1/2m) Σ(hθ(xⁱ) - yⁱ)²
```

**Regularisation** penalises model complexity to reduce overfitting. Ridge (L2) shrinks coefficients toward zero but retains all features. Lasso (L1) can zero out coefficients entirely, performing implicit feature selection. ElasticNet combines both penalties.

**Evaluation Metrics — Regression:**

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | (1/n) Σ\|yᵢ - ŷᵢ\| | Average absolute error |
| MSE | (1/n) Σ(yᵢ - ŷᵢ)² | Penalises large errors more heavily |
| RMSE | √MSE | Same units as the target variable |
| R² | 1 - (SS_res/SS_tot) | Proportion of variance explained; closer to 1 is better |

---

### 2️⃣ Supervised Learning — Classification

**Logistic Regression** models the probability of binary class membership using the sigmoid function.
```
P(y=1|x) = 1 / (1 + e^(-θᵀx))
Cost: J(θ) = -(1/m) Σ [y·log(hθ) + (1-y)·log(1-hθ)]
```

**Evaluation Metrics — Classification:**

| Metric | Formula | Best Used When |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced class distributions |
| Precision | TP/(TP+FP) | Cost of false positives is high |
| Recall | TP/(TP+FN) | Cost of false negatives is high |
| F1-Score | 2(P×R)/(P+R) | Trade-off between precision and recall |
| AUC-ROC | Area under curve | Comparing models across thresholds |

**Decision Trees** split data recursively using Information Gain (entropy-based) or Gini Impurity as the splitting criterion. They are highly interpretable but prone to overfitting on complex datasets.

**Ensemble Methods** mitigate individual model weaknesses through aggregation. Random Forests build multiple decorrelated trees via bootstrap sampling and random feature selection. Gradient Boosting trains trees sequentially, where each tree corrects the residuals of the previous one. Implementations include XGBoost, LightGBM, and CatBoost.

**Support Vector Machines** (SVMs) find the optimal hyperplane that maximises the margin between classes. The kernel trick (RBF, polynomial, sigmoid) enables SVMs to operate in high-dimensional feature spaces without explicitly computing transformations.

---

### 3️⃣ Unsupervised Learning — Clustering

**K-Means** partitions data into K clusters by minimising within-cluster variance:
```
Objective: minimize Σᵢ₌₁ᵏ Σₓ ∈ Cᵢ ||x - μᵢ||²
```
The optimal K is determined using the Elbow method or the Silhouette Score:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Where a(i) is the mean intra-cluster distance and b(i) is the mean nearest-cluster distance.

**Hierarchical Clustering** builds a tree of clusters (dendrogram) using agglomerative (bottom-up) or divisive (top-down) strategies. Ward linkage minimises within-cluster variance at each merge step and is generally the most effective linkage criterion.

**DBSCAN** identifies clusters based on density rather than distance, making it robust to arbitrary cluster shapes and capable of identifying noise points as outliers.

---

### 4️⃣ Dimensionality Reduction

**PCA** projects data onto the directions of maximum variance:
```
Steps: Standardise → Covariance Matrix → Eigendecomposition → Sort → Project
Variance Explained: Vₖ = (Σᵢ₌₁ᵏ λᵢ) / (Σᵢ₌₁ⁿ λᵢ)
```

**t-SNE** performs non-linear dimensionality reduction, preserving local neighbourhood structure. It is primarily used for 2D and 3D visualisation of high-dimensional datasets and should not be used for dimensionality reduction in preprocessing pipelines (use PCA instead).

---

## 🔬 Advanced Topics

### 1️⃣ Deep Learning Fundamentals

Neural networks compute layered transformations of the input:
```
aˡ = σ(Wˡ · aˡ⁻¹ + bˡ)
```
Where aˡ is the activation of layer l, Wˡ are the weights, bˡ are the biases, and σ is the activation function. ReLU (max(0, x)) is the standard choice for hidden layers; Softmax is used for multi-class output layers.

**Backpropagation** computes gradients via the chain rule, enabling efficient weight updates across all layers. The Adam optimiser is the current default choice for training deep networks.

---

### 2️⃣ Natural Language Processing (NLP)

Text preprocessing standardises raw text into a form suitable for modelling: tokenisation, lowercasing, punctuation removal, stop word filtering, and lemmatisation reduce vocabulary size and noise.

**Vectorisation Methods:**
- **Bag-of-Words (BoW)**: Represents documents as word frequency vectors.
- **TF-IDF**: Balances term frequency against rarity across the corpus (TF-IDF = TF × IDF), upweighting discriminative terms.
- **Word2Vec**: Maps words to dense, semantically meaningful embeddings.
- **BERT**: Generates contextual embeddings using transformer self-attention, capturing word meaning relative to surrounding context.

---

### 3️⃣ Time Series Analysis

Time series data is decomposed into Trend (long-term direction), Seasonality (periodic patterns), Cyclical components, and Noise. Additive decomposition (Yₜ = Tₜ + Sₜ + Cₜ + Nₜ) applies when seasonal variation is roughly constant; multiplicative decomposition applies when seasonal variation grows with the trend.

**Forecasting methods** range from classical statistical models — ARIMA (AutoRegressive Integrated Moving Average) and exponential smoothing — to sequence-based neural architectures such as LSTMs and GRUs, which learn temporal dependencies directly from data.

---

## 📈 Results & Evaluation

The following table summarises the performance of key model implementations on their respective project datasets. All results are measured on held-out test sets (20% of data), with hyperparameters tuned via 5-fold cross-validation.

| Project | Model | Primary Metric | Score | Notes |
|---------|-------|----------------|-------|-------|
| House Price Prediction | Gradient Boosting (XGBoost) | RMSE | ~29,000 | Log-transformed target |
| Customer Segmentation | K-Means (K=4) | Silhouette Score | ~0.54 | Elbow method for K selection |
| Fraud Detection | Random Forest | AUC-ROC | ~0.97 | Class imbalance handled with SMOTE |
| Sentiment Analysis | Logistic Regression (TF-IDF) | F1-Score | ~0.91 | Compared with BERT fine-tuning |
| Stock Forecasting | ARIMA / LSTM | MAPE | ~4.2% | 30-day horizon |

> **Note**: Exact scores may vary with dataset updates, random seeds, or additional hyperparameter tuning. Reproduction instructions are included in each project notebook.

### Evaluation Methodology

All projects follow a consistent evaluation framework. Cross-validation (k=5) prevents overfitting to any single train-test split. Learning curves diagnose bias-variance trade-offs. Feature importance plots (for tree-based models) and coefficient analysis (for linear models) provide interpretability. Confusion matrices and ROC curves are used for all classification tasks.

### Business Impact

Each project is framed around a concrete business question. The fraud detection model, for instance, operates at 97% AUC-ROC, which at a realistic false-positive rate translates to a meaningful reduction in manual review workload while catching the vast majority of fraudulent transactions. The customer segmentation analysis identified four behavioural clusters that can directly inform targeted marketing spend allocation.

---

## 🚀 Usage

### Running a Notebook

After completing installation (see [Installation Instructions](#-installation-instructions)), launch Jupyter and navigate to any notebook:

```bash
# Activate your virtual environment first
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Start Jupyter
jupyter notebook
```

Navigate to the relevant directory in the Jupyter interface and open any `.ipynb` file.

### Suggested Execution Order

For a first-time user, the recommended entry sequence is:

```bash
01_eda/univariate_analysis.ipynb          → Understand data structure
02_preprocessing/missing_values.ipynb     → Learn cleaning workflows
03_models/regression/linear_regression.ipynb → First model implementation
05_projects/project_1_house_price_prediction/ → End-to-end application
```

### Running a Specific Project

Each project in `05_projects/` contains its own `README.md` with execution instructions. The general pattern is:

```bash
cd 05_projects/project_1_house_price_prediction/
jupyter notebook house_price_prediction.ipynb
```

### Running with Script (Non-Notebook)

Where `.py` equivalents are provided:

```bash
python 05_projects/project_3_fraud_detection/train.py \
  --data datasets/processed/fraud_data.csv \
  --model random_forest \
  --output models/fraud_rf_v1.pkl
```

---

## 📊 Key Formulas Reference

Quick reference for the core mathematical formulas used throughout the repository.

| Concept | Formula | Purpose |
|---------|---------|---------|
| **Mean** | μ = (1/n)Σxᵢ | Central tendency of a feature |
| **Variance** | σ² = (1/n)Σ(xᵢ - μ)² | Dispersion around the mean |
| **Std Deviation** | σ = √σ² | Spread in original units |
| **Covariance** | Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] | Joint variability of two features |
| **Correlation** | ρ = Cov(X,Y)/(σₓσᵧ) | Normalised relationship, ρ ∈ [-1, 1] |
| **Z-Score** | z = (x - μ)/σ | Standardisation and outlier detection |
| **Entropy** | H(X) = -Σp(x)log(p(x)) | Information content (decision tree splitting) |
| **Gini Impurity** | G = 1 - Σp(x)² | Node impurity criterion for trees |
| **Gradient Descent** | θ ← θ - α∇J(θ) | Parameter update rule |
| **Sigmoid** | σ(x) = 1/(1 + e^(-x)) | Logistic regression and neural nets |
| **Softmax** | fᵢ(x) = e^(xᵢ)/Σe^(xⱼ) | Multi-class probability output |
| **TF-IDF** | TF × log(N/df) | Text feature weighting |
| **Silhouette** | s(i) = (b(i)-a(i))/max(a,b) | Clustering quality measure |

---

## 📚 Recommended Learning Path

This 13-week curriculum is designed to take a motivated learner from Python fundamentals to end-to-end ML deployment.

### Phase 1 — Foundations (Weeks 1–2)
Begin with `fundamentals/` notebooks covering linear algebra (vectors, matrices, eigendecomposition), probability theory, and calculus. Simultaneously practice Python, NumPy, and Pandas through the data loading and manipulation exercises in `01_eda/`.

### Phase 2 — Data Science Practicals (Weeks 3–4)
Work through the full `01_eda/` and `02_preprocessing/` directories. Focus on building intuition for what data looks like before modelling and how preprocessing choices affect downstream performance.

### Phase 3 — Classical Machine Learning (Weeks 5–8)
Implement algorithms in `03_models/` in the following order: Linear Regression → Logistic Regression → Decision Trees → Random Forest → Gradient Boosting → SVM → K-Means → PCA. Understand each algorithm's cost function, assumptions, and failure modes before moving on.

### Phase 4 — Advanced Topics (Weeks 9–12)
Explore the `04_advanced/` notebooks on deep learning, NLP, and time series. These require a solid grounding in Phase 3 material. Focus on understanding architecture decisions rather than memorising hyperparameter values.

### Phase 5 — End-to-End Projects & Deployment (Week 13+)
Implement all five projects in `05_projects/` from data ingestion through model evaluation. Extend at least one project with a REST API endpoint using FastAPI or Flask, and document the deployment process.

---

## 🔭 Future Work

Several enhancements are planned for this repository. Contributions addressing any of the following areas are particularly welcome.

**Model Expansion.** Adding implementations of modern architectures — Vision Transformers (ViT), graph neural networks (GNNs), and tabular deep learning frameworks (TabNet, FT-Transformer) — would extend the repository's coverage of the current state of the art.

**MLOps Integration.** The current repository focuses on research-phase workflows. Adding a dedicated `07_mlops/` section covering experiment tracking (MLflow), model registries, CI/CD pipelines for model validation, and containerised deployment (Docker + FastAPI) would complete the end-to-end lifecycle.

**Reinforcement Learning Projects.** The `04_advanced/reinforcement_learning.ipynb` notebook currently covers theory. Adding practical implementations using OpenAI Gymnasium environments would make this section actionable.

**Interactive Dashboards.** Wrapping project outputs in Streamlit or Gradio dashboards would make model results accessible to non-technical stakeholders and provide a portfolio-ready presentation layer.

**Automated Testing.** Adding unit tests for preprocessing pipelines and model training scripts using `pytest` would improve repository reliability and demonstrate software engineering best practices.

**Multilingual NLP.** Expanding the sentiment analysis project to support multilingual text using multilingual BERT (mBERT) or XLM-RoBERTa would increase practical applicability.

---

## 📦 Additional Resources

### Recommended Courses

**Foundational to Intermediate:**
- [Andrew Ng's Machine Learning Specialisation](https://www.coursera.org/specializations/machine-learning-introduction) — The most widely recommended entry point to ML theory and practice.
- [Fast.ai — Practical Deep Learning](https://course.fast.ai/) — A top-down, code-first approach to deep learning.
- [Kaggle Learn](https://kaggle.com/learn) — Concise micro-courses with immediate hands-on exercises.

**Advanced:**
- [Stanford CS229 — Machine Learning](http://cs229.stanford.edu/) — Graduate-level treatment of ML mathematics.
- [Stanford CS231N — CNNs for Visual Recognition](http://cs231n.stanford.edu/) — The reference course for computer vision.
- [Stanford CS224N — NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) — Advanced NLP with transformers.

### Recommended Books

| Title | Author | Level |
|-------|--------|-------|
| Hands-On Machine Learning (3rd Ed.) | Aurélien Géron | Beginner – Intermediate |
| The Hundred-Page Machine Learning Book | Andriy Burkov | Intermediate |
| Pattern Recognition and Machine Learning | Christopher Bishop | Advanced |
| Deep Learning | Goodfellow, Bengio, Courville | Advanced |
| Statistical Rethinking | Richard McElreath | Intermediate (Bayesian focus) |

### Public Dataset Repositories

- [Kaggle Datasets](https://www.kaggle.com/datasets) — Thousands of community-contributed datasets with discussion forums and benchmark notebooks.
- [UCI ML Repository](https://archive.ics.uci.edu/ml) — Classic benchmark datasets with well-documented problem definitions.
- [Google Dataset Search](https://datasetsearch.research.google.com/) — Cross-repository search engine for open datasets.
- [HuggingFace Datasets](https://huggingface.co/datasets) — Curated NLP and multimodal datasets with a Python API.

### Communities

- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) and [r/datascience](https://www.reddit.com/r/datascience/) — Active discussion of research and industry practice.
- [Papers With Code](https://paperswithcode.com/) — Links ML papers to open-source implementations.
- [Kaggle](https://kaggle.com) — Competition platform with collaborative notebooks.
- [Data Science Stack Exchange](https://datascience.stackexchange.com/) — Q&A for technical questions.

### Key Libraries

| Category | Libraries |
|----------|-----------|
| Data Manipulation | Pandas, NumPy, Polars |
| Visualisation | Matplotlib, Seaborn, Plotly, Altair |
| Classical ML | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| Deep Learning | TensorFlow/Keras, PyTorch, JAX |
| NLP | HuggingFace Transformers, spaCy, NLTK |
| Time Series | statsmodels, Prophet, sktime |
| Experiment Tracking | MLflow, Weights & Biases |

---

## 🙏 Acknowledgments & References

This repository was developed and maintained by [Shubham Raj](https://github.com/shubhmrj). The content draws on the following foundational works and open-source resources.

**Textbooks and Courses:**
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Andrew Ng, Stanford University — CS229 Machine Learning lecture notes and course materials.

**Open-Source Libraries:** This work builds directly on the NumPy, pandas, scikit-learn, Matplotlib, Seaborn, XGBoost, LightGBM, and TensorFlow/PyTorch communities, whose documentation and examples inform many implementations in this repository.

**Dataset Sources:** Kaggle, UCI Machine Learning Repository, HuggingFace Datasets, and Yahoo Finance (via yfinance).

**Special Thanks:** To every contributor who has submitted an issue, suggested a correction, or opened a pull request. Your engagement makes this repository better for everyone in the community.

---

## 📄 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2024 Shubham Raj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

Full license text: [LICENSE](https://github.com/shubhmrj/Data-Science/blob/main/LICENSE)

---

## 🌟 Connect & Contribute

If this repository has been useful to your learning, a ⭐ **star** on GitHub is the best way to help others discover it.

### Get in Touch

| Platform | Link |
|----------|------|
| 📧 Email | shubham4312raj@gmail.com |
| 💼 LinkedIn | [linkedin.com/in/shubmraj](https://linkedin.com/in/shubmraj) |
| 👻 GitHub | [github.com/shubhmrj](https://github.com/shubhmrj) |

### Contributing

Contributions are warmly welcomed. Please follow this workflow:

1. Fork the repository from [github.com/shubhmrj/Data-Science](https://github.com/shubhmrj/Data-Science).
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes with a clear message: `git commit -m "Add: description of change"`.
4. Push to your fork: `git push origin feature/your-feature-name`.
5. Open a Pull Request with a description of what you changed and why.

Bug reports, documentation improvements, new notebook contributions, and project additions are all equally valued.

---

> ✨ *"Data Science is not just about algorithms — it's about transforming curiosity and questions into actionable insights."* ✨
>
> **Happy Learning! 🚀**