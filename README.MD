# 📊 Data Science & Machine Learning Complete Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org/) [![Pandas](https://img.shields.io/badge/Library-Pandas-yellow?logo=pandas)](https://pandas.pydata.org/) [![NumPy](https://img.shields.io/badge/Library-NumPy-lightblue?logo=numpy)](https://numpy.org/) [![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-green?logo=scikitlearn)](https://scikit-learn.org/) [![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-red?logo=plotly)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/Visualization-Seaborn-teal)](https://seaborn.pydata.org/) [![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)](docs/Contributing.md) [![GitHub issues](https://img.shields.io/github/issues/shubhmrj/Data-Science)](https://github.com/shubhmrj/Data-Science/issues) [![GitHub forks](https://img.shields.io/github/forks/shubhmrj/Data-Science)](https://github.com/shubhmrj/Data-Science/network) [![GitHub stars](https://img.shields.io/github/stars/shubhmrj/Data-Science)](https://github.com/shubhmrj/Data-Science/stargazers) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to the **Comprehensive Data Science & ML Learning Repository** 🚀 — a deep dive into hands-on experiments, mathematical foundations, advanced algorithms, and real-world projects.

This is a **complete learning ecosystem** where theory meets practice, covering everything from statistics fundamentals to production-ready ML systems.

---

## 📚 Table of Contents

1. [What's Inside](#-whats-inside)
2. [Mathematical Foundations](#-mathematical-foundations)
3. [Data Science Practicals](#-data-science-practicals)
4. [Machine Learning Models](#-machine-learning-models)
5. [Advanced Topics](#-advanced-topics)
6. [Learning Path](#-recommended-learning-path)
7. [Repository Structure](#-repository-structure)
8. [Getting Started](#-getting-started)

---

## 🧭 What's Inside?

### 🔹 Complete Learning Curriculum
- **Foundations**: Statistics, Probability, Linear Algebra
- **Data Engineering**: Collection, Cleaning, Transformation
- **Exploratory Analysis**: Visualization, Pattern Discovery
- **Modeling**: Supervised, Unsupervised, Reinforcement Learning
- **Deployment**: Model serving, monitoring, optimization
- **Real-World Projects**: End-to-end implementations

---

## 🧮 Mathematical Foundations

### 1️⃣ Linear Algebra Essentials

#### Vectors & Matrices
- **Vector**: A 1D array of numbers: **v** = [v₁, v₂, ..., vₙ]
- **Matrix**: A 2D grid of numbers: **A** = [[a₁₁, a₁₂], [a₂₁, a₂₂]]
- **Operations**: Addition, Multiplication, Transposition, Inversion

**Key Concepts:**
- **Matrix Multiplication**: (m×n) × (n×p) = (m×p)
- **Determinant** (det): Scalar value indicating matrix invertibility
- **Eigenvalues & Eigenvectors**: For matrix **A**: **Av** = λ**v**
- **Rank**: Maximum number of linearly independent rows/columns

**Applications in ML:**
- Dimensionality reduction (PCA)
- Matrix factorization
- Feature transformations

---

### 2️⃣ Probability & Statistics

#### Fundamental Concepts

**Probability Distribution:**
- Discrete: P(X = x)
- Continuous: f(x) (Probability Density Function)

**Key Distributions:**

| Distribution | PDF/PMF | Use Case |
|-------------|---------|----------|
| **Normal** | ![Normal](https://wikimedia.org/api/rest_v1/media/math/render/svg/0dd8de6871c759a283182bf653baccc147b61b58) | Many natural phenomena |
| **Binomial** | P(X=k) = C(n,k)p^k(1-p)^(n-k) | Binary outcomes |
| **Poisson** | P(X=k) = (e^(-λ)λ^k)/k! | Event counting |
| **Exponential** | f(x) = λe^(-λx) | Time between events |

**Statistical Measures:**
- **Mean (μ)**: μ = (Σx)/n
- **Variance (σ²)**: σ² = E[(X - μ)²]
- **Standard Deviation (σ)**: σ = √(σ²)
- **Covariance**: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
- **Correlation**: ρ = Cov(X,Y)/(σₓσᵧ), range: [-1, 1]

**Hypothesis Testing:**
- **Null Hypothesis (H₀)**: Assumed true initially
- **P-value**: Probability of observing data given H₀
- **Significance Level (α)**: Rejection threshold (typically 0.05)

---

### 3️⃣ Calculus & Optimization

#### Derivatives & Gradients
- **Derivative**: Rate of change: f'(x) = df/dx
- **Partial Derivative**: Change w.r.t. one variable: ∂f/∂x
- **Gradient (∇f)**: Vector of all partial derivatives
- **Hessian Matrix**: Matrix of second-order partial derivatives

#### Optimization Methods

**Gradient Descent:**
```
θₜ₊₁ = θₜ - α∇J(θ)
```
Where:
- θ: Parameters
- α: Learning rate
- ∇J(θ): Gradient of cost function

**Variants:**
- Batch GD: Uses entire dataset
- Stochastic GD: Uses one sample
- Mini-batch GD: Uses subset of samples

**Advanced Optimizers:**
- **Momentum**: Accelerates GD with velocity
- **RMSprop**: Adaptive learning rates
- **Adam**: Combines momentum + RMSprop (most popular)

---

## 📊 Data Science Practicals

### 1️⃣ Data Collection & Preparation

#### Sources
- APIs & Web Scraping
- Databases (SQL, NoSQL)
- Public datasets (Kaggle, UCI, etc.)
- Sensors & IoT devices

#### Data Quality Issues
- **Missing Values**: Mean imputation, KNN imputation, Multiple imputation
- **Outliers**: Z-score (|z| > 3), IQR method (Q3 + 1.5×IQR)
- **Duplicates**: Row-level & column-level deduplication
- **Inconsistencies**: Typos, format issues, encoding problems

---

### 2️⃣ Exploratory Data Analysis (EDA)

#### Univariate Analysis
- **Numerical**: Histogram, Box plot, Density plot
  - Skewness: γ = E[(X-μ)³]/σ³
  - Kurtosis: κ = E[(X-μ)⁴]/σ⁴ - 3
- **Categorical**: Bar chart, Pie chart, Value counts

#### Bivariate Analysis
- **Correlation Matrix**: Pearson (r), Spearman (ρ), Kendall (τ)
- **Scatter plots**: Linear relationships
- **Cross-tabulation**: Categorical relationships
- **Chi-square test**: Independence between categories

#### Multivariate Analysis
- **Heatmaps**: Correlation visualization
- **Pairplots**: All pairwise relationships
- **Dimensionality reduction**: PCA visualization

---

### 3️⃣ Feature Engineering

#### Feature Scaling
```
Standardization (Z-score):     x' = (x - μ)/σ
Normalization (Min-Max):       x' = (x - min)/(max - min)
Robust Scaling:                x' = (x - median)/IQR
```

#### Feature Encoding
- **One-Hot Encoding**: Categorical → Binary columns
- **Label Encoding**: Categorical → Integer values
- **Ordinal Encoding**: Ordered categories → Integers
- **Target Encoding**: Replace with target mean

#### Feature Creation
- **Polynomial Features**: x² , x³, xy
- **Interaction Terms**: x₁ × x₂
- **Binning**: Continuous → Categorical buckets
- **Domain Knowledge**: Business-driven features

#### Feature Selection
- **Correlation-based**: Remove highly correlated features
- **Variance-based**: Remove low-variance features
- **Model-based**: Importance from tree models
- **Recursive Elimination**: Iteratively remove weak features

---

## 🤖 Machine Learning Models

### 1️⃣ Supervised Learning

#### Regression

**Linear Regression**
```
Hypothesis:     ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
Cost Function:  J(θ) = (1/2m)Σ(hθ(xⁱ) - yⁱ)²
```
- **Pros**: Simple, interpretable, fast
- **Cons**: Assumes linear relationship, sensitive to outliers
- **Use Case**: Housing prices, trend forecasting

**Polynomial Regression**
```
ŷ = θ₀ + θ₁x + θ₂x² + θ₃x³ + ...
```
- **Pros**: Captures non-linear patterns
- **Cons**: Risk of overfitting with high degrees
- **Degree Selection**: Use cross-validation

**Regularization Techniques**

1. **Ridge Regression (L2)**
   ```
   J(θ) = (1/2m)Σ(hθ(xⁱ) - yⁱ)² + (λ/2m)Σθⱼ²
   ```
   - Shrinks coefficients toward zero
   - Never eliminates features

2. **Lasso Regression (L1)**
   ```
   J(θ) = (1/2m)Σ(hθ(xⁱ) - yⁱ)² + (λ/m)Σ|θⱼ|
   ```
   - Can eliminate features (sparse solutions)
   - Feature selection built-in

3. **ElasticNet (L1 + L2)**
   ```
   J(θ) = (1/2m)Σ(hθ(xⁱ) - yⁱ)² + λ₁Σ|θⱼ| + λ₂Σθⱼ²
   ```

**Evaluation Metrics (Regression):**
- **MAE** (Mean Absolute Error): (1/n)Σ|yᵢ - ŷᵢ|
- **MSE** (Mean Squared Error): (1/n)Σ(yᵢ - ŷᵢ)²
- **RMSE** (Root MSE): √MSE
- **R² Score**: 1 - (SS_res/SS_tot), range: [0, 1]

---

#### Classification

**Logistic Regression**
```
Hypothesis:     P(y=1|x) = 1/(1 + e^(-θᵀx))  [Sigmoid]
Cost Function:  J(θ) = -(1/m)Σ[ylog(hθ) + (1-y)log(1-hθ)]
Decision Rule:  If P(y=1|x) > 0.5, predict 1; else 0
```
- **Output**: Probability between 0 and 1
- **Extension**: Multi-class via One-vs-Rest

**Evaluation Metrics (Classification):**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | Of positive predictions, how many correct |
| **Recall/Sensitivity** | TP/(TP+FN) | Of actual positives, how many detected |
| **F1-Score** | 2(Precision×Recall)/(Precision+Recall) | Harmonic mean |
| **Specificity** | TN/(TN+FP) | True negative rate |
| **AUC-ROC** | Area under PR curve | Classification quality |

**Confusion Matrix:**
```
                Predicted
               Pos    Neg
Actual Pos  [TP]   [FN]
       Neg  [FP]   [TN]
```

---

#### Decision Trees & Ensemble Methods

**Decision Trees**
- **Split Criterion**: Information Gain (Entropy-based) or Gini Impurity
- **Entropy**: H(X) = -Σ p(x)log₂(p(x))
- **Information Gain**: IG = H(parent) - Σ(|child|/|parent|)×H(child)
- **Gini Impurity**: G = 1 - Σ p(x)²
- **Pros**: Interpretable, non-linear, handles both types of data
- **Cons**: Prone to overfitting, unstable with small data changes

**Random Forest**
```
Ensemble of T trees: ŷ = (1/T)ΣT ŷₜ  [Regression]
                     ŷ = Mode(ŷ₁,...,ŷT)  [Classification]
```
- Bootstrap aggregating (Bagging): Random sample with replacement
- Random feature selection at each split
- **Pros**: Reduces overfitting, handles large datasets
- **Cons**: Less interpretable, slower prediction

**Gradient Boosting**
- Sequential tree building: Each tree corrects previous errors
- **Loss Function**: L(y, ŷ)
- **Update**: ŷₜ = ŷₜ₋₁ + αFₜ(x)
- **Popular implementations**: XGBoost, LightGBM, CatBoost

---

#### Support Vector Machines (SVM)

**Linear SVM**
```
Objective:     minimize (1/2)||w||² + C×Σξᵢ
Constraint:    yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ
```
- **w**: Weight vector
- **b**: Bias term
- **ξᵢ**: Slack variables (allow misclassification)
- **C**: Regularization parameter

**Kernel Trick**: Non-linear transformation
- **Linear**: K(x,y) = xᵀy
- **Polynomial**: K(x,y) = (xᵀy + c)^d
- **RBF**: K(x,y) = exp(-γ||x-y||²)
- **Sigmoid**: K(x,y) = tanh(κxᵀy + θ)

---

### 2️⃣ Unsupervised Learning

#### Clustering

**K-Means**
```
Algorithm:
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat 2-3 until convergence

Objective:  minimize Σᵢ₌₁ᵏ Σₓ ∈ Cᵢ ||x - μᵢ||²
```
- **Pros**: Simple, scalable, fast
- **Cons**: Requires K specification, sensitive to initialization
- **Optimal K**: Elbow method or Silhouette score

**Silhouette Score:**
```
s(i) = (b(i) - a(i))/max(a(i), b(i))
```
Where: a(i) = avg distance to cluster points, b(i) = avg distance to nearest cluster

**Hierarchical Clustering**
- **Agglomerative**: Bottom-up merging
- **Divisive**: Top-down splitting
- **Linkage Criteria**:
  - Single: min distance between clusters
  - Complete: max distance between clusters
  - Average: mean distance between clusters
  - Ward: minimizes within-cluster variance

**DBSCAN (Density-Based)**
- **Parameters**: ε (radius), MinPts (minimum density)
- **Core Point**: ≥ MinPts neighbors within ε
- **Pros**: Finds arbitrary-shaped clusters, no K needed
- **Cons**: Sensitive to parameters, varying density

---

#### Dimensionality Reduction

**Principal Component Analysis (PCA)**
```
Steps:
1. Standardize data: x' = (x - μ)/σ
2. Compute covariance matrix: Σ = (1/n)XᵀX
3. Find eigenvalues & eigenvectors
4. Sort by eigenvalues (descending)
5. Project: X_new = X × W (where W = top k eigenvectors)

Variance Explained: Vₖ = (Σᵢ₌₁ᵏ λᵢ)/(Σᵢ₌₁ⁿ λᵢ)
```
- **Pros**: Removes correlation, visualizes high-dim data, noise reduction
- **Cons**: Loss of interpretability, linear only

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Non-linear dimensionality reduction
- Preserves local structure
- Popular for visualization (2D/3D)
- **Pros**: Excellent for visualization
- **Cons**: Computationally expensive, non-deterministic

---

### 3️⃣ Semi-Supervised & Self-Supervised Learning

**Self-Training**
- Train on labeled data
- Predict on unlabeled data
- Add high-confidence predictions to training set
- Retrain

**Co-Training**
- Multiple views/models of data
- Each model teaches the other

---

## 🔬 Advanced Topics

### 1️⃣ Deep Learning Basics

**Neural Networks**
```
Layer output:   aˡ = σ(Wˡaˡ⁻¹ + bˡ)
Where:
- aˡ: Activation of layer l
- Wˡ: Weights
- bˡ: Biases
- σ: Activation function
```

**Activation Functions:**
- **ReLU**: f(x) = max(0, x) [Most common]
- **Sigmoid**: f(x) = 1/(1 + e^(-x)) [Output layer, classification]
- **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) [Hidden layers]
- **Softmax**: fᵢ(x) = e^(xᵢ)/Σ e^(xⱼ) [Multi-class output]

**Backpropagation**
```
Forward Pass:    Compute activations layer by layer
Compute Loss:    L = (1/m)Σ loss(y, ŷ)
Backward Pass:   Compute gradients ∂L/∂W, ∂L/∂b
Update:          W ← W - α∂L/∂W
```

---

### 2️⃣ Natural Language Processing (NLP)

**Text Preprocessing:**
- Tokenization: Split into words/subwords
- Lowercasing: Standardize case
- Removing punctuation & special characters
- Stemming/Lemmatization: Reduce to root form
- Stop word removal

**Vectorization:**
- **Bag-of-Words (BoW)**: Word frequency vector
- **TF-IDF**: Term Frequency-Inverse Document Frequency
  ```
  TF(t,d) = (frequency of t in d)/(total words in d)
  IDF(t) = log(total docs/docs containing t)
  TF-IDF = TF(t,d) × IDF(t)
  ```
- **Word2Vec**: Dense word embeddings (Word Skip-gram model)
- **BERT**: Contextual embeddings from transformers

---

### 3️⃣ Time Series Analysis

**Components:**
- **Trend**: Long-term direction
- **Seasonality**: Periodic patterns
- **Cyclical**: Long-term cycles (not fixed period)
- **Noise**: Random fluctuations

**Decomposition:**
```
Additive:      Yₜ = Tₜ + Sₜ + Cₜ + Nₜ
Multiplicative: Yₜ = Tₜ × Sₜ × Cₜ × Nₜ
```

**Forecasting Methods:**
- **ARIMA**: AutoRegressive Integrated Moving Average
  ```
  AR(p): Yₜ = c + Σφᵢ Yₜ₋ᵢ + εₜ
  MA(q): Yₜ = μ + εₜ + Σθᵢ εₜ₋ᵢ
  ```
- **Exponential Smoothing**: Weighted average of past observations
- **LSTM/GRU**: Recurrent neural networks for sequences

---

### 4️⃣ Model Validation & Selection

**Cross-Validation:**
```
K-Fold CV: Split data into k parts, train on k-1, test on 1
Score = (1/k)Σ scoresᵢ
```

**Hyperparameter Tuning:**
- **Grid Search**: Test all combinations
- **Random Search**: Random combinations
- **Bayesian Optimization**: Probabilistic approach

**Overfitting vs Underfitting:**
- **Overfitting**: High train accuracy, low test accuracy
  - Solution: Regularization, more data, simpler model
- **Underfitting**: Both train & test accuracies low
  - Solution: Complex model, more features, remove regularization

**Learning Curves**: Plot accuracy vs training set size

---

## 📚 Recommended Learning Path

### **Phase 1: Foundations (Weeks 1-2)**
1. Linear Algebra basics
2. Probability & Statistics fundamentals
3. Python, NumPy, Pandas
4. Data loading and basic manipulation

### **Phase 2: Data Science (Weeks 3-4)**
1. EDA techniques
2. Data cleaning & preprocessing
3. Feature engineering
4. Data visualization with Matplotlib/Seaborn

### **Phase 3: Classical ML (Weeks 5-8)**
1. Linear & Logistic Regression
2. Decision Trees & Ensemble Methods (RF, GB)
3. SVM & KNN
4. Clustering (K-Means, Hierarchical, DBSCAN)

### **Phase 4: Advanced Topics (Weeks 9-12)**
1. Dimensionality Reduction (PCA, t-SNE)
2. Time Series Analysis (ARIMA, LSTM)
3. Introduction to Deep Learning
4. NLP basics

### **Phase 5: Projects & Deployment (Weeks 13+)**
1. End-to-end ML projects
2. Model evaluation & optimization
3. Model deployment & serving
4. Monitoring & maintenance

---

## 🗂️ Repository Structure

```
📁 Data-Science/
│
├── 📁 datasets/                    # Raw & processed datasets
│   ├── raw/                        # Original data
│   └── processed/                  # Cleaned & transformed data
│
├── 📁 fundamentals/                # Mathematical & conceptual foundations
│   ├── linear_algebra/
│   ├── probability_statistics/
│   ├── calculus_optimization/
│   └── mathematics_notes.ipynb
│
├── 📁 01_eda/                      # Exploratory Data Analysis
│   ├── univariate_analysis.ipynb
│   ├── bivariate_analysis.ipynb
│   └── multivariate_analysis.ipynb
│
├── 📁 02_preprocessing/            # Data Cleaning & Feature Engineering
│   ├── missing_values.ipynb
│   ├── outlier_detection.ipynb
│   ├── feature_scaling.ipynb
│   ├── feature_encoding.ipynb
│   └── feature_selection.ipynb
│
├── 📁 03_models/                   # ML Algorithms Implementation
│   ├── 📁 regression/
│   │   ├── linear_regression.ipynb
│   │   ├── polynomial_regression.ipynb
│   │   ├── regularization.ipynb
│   │   └── advanced_regression.ipynb
│   ├── 📁 classification/
│   │   ├── logistic_regression.ipynb
│   │   ├── decision_trees.ipynb
│   │   ├── ensemble_methods.ipynb
│   │   ├── svm.ipynb
│   │   └── naive_bayes.ipynb
│   └── 📁 unsupervised/
│       ├── clustering.ipynb
│       ├── dimensionality_reduction.ipynb
│       └── anomaly_detection.ipynb
│
├── 📁 04_advanced/                 # Advanced ML Topics
│   ├── deep_learning_basics.ipynb
│   ├── nlp_basics.ipynb
│   ├── time_series.ipynb
│   └── reinforcement_learning.ipynb
│
├── 📁 05_projects/                 # End-to-End ML Projects
│   ├── 📁 project_1_house_price_prediction/
│   ├── 📁 project_2_customer_segmentation/
│   ├── 📁 project_3_fraud_detection/
│   ├── 📁 project_4_sentiment_analysis/
│   └── 📁 project_5_stock_forecasting/
│
├── 📁 06_notes/                    # Theory & Summary Notes
│   ├── ml_algorithms_summary.md
│   ├── statistical_concepts.md
│   ├── common_pitfalls.md
│   └── quick_reference.md
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- pip or conda
- Jupyter Notebook or JupyterLab
- Git

### 2. Installation

```bash
# Clone repository
git clone https://github.com/shubhmrj/Data-Science.git
cd Data-Science

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### 3. Recommended Packages

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install scipy statsmodels xgboost lightgbm
pip install tensorflow keras  # For deep learning
pip install nltk spacy       # For NLP
```

### 4. First Steps
1. Start with **fundamentals/** for math concepts
2. Move to **01_eda/** for practical data exploration
3. Follow the learning path above
4. Implement projects in **05_projects/**

---

## 📊 Key Formulas Reference

| Concept | Formula | Use |
|---------|---------|-----|
| Mean | μ = (1/n)Σxᵢ | Center of data |
| Variance | σ² = (1/n)Σ(xᵢ - μ)² | Data spread |
| Covariance | Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] | Joint variability |
| Correlation | ρ = Cov(X,Y)/(σₓσᵧ) | Standardized association |
| Z-score | z = (x - μ)/σ | Standardization |
| Entropy | H(X) = -Σp(x)log(p(x)) | Information content |
| Gini | G = 1 - Σp(x)² | Impurity measure |

---

## 🌟 Let's Connect

If you find this resource valuable, please ⭐ star the repository!
Feedback, suggestions, and collaborations are always welcome.

📧 **Email**: [shubham4312raj@gmail.com]
💼 **LinkedIn**: [linkedin.com/in/shubmraj]
🐙 **GitHub**: [github.com/shubhmrj]

---

## 📚 Additional Resources

### Online Courses
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [FastAI - Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS229 - Machine Learning](http://cs229.stanford.edu/)

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "The Hundred-Page Machine Learning Book"
- "Pattern Recognition & Machine Learning" by Bishop
- "Deep Learning" by Goodfellow, Bengio, Courville

### Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

### Communities
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Kaggle](https://www.kaggle.com/)
- [Papers With Code](https://paperswithcode.com/)

---

✨ *"Data Science is not just about algorithms—it's about transforming curiosity and questions into actionable insights."* ✨

**Happy Learning! 🚀**
