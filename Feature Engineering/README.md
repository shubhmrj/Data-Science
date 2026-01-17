# Feature Engineering - Comprehensive Guide

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Transformation](#feature-transformation)
5. [Encoding Techniques](#encoding-techniques)
6. [Feature Selection](#feature-selection)
7. [Advanced Techniques](#advanced-techniques)
8. [Project Examples](#project-examples)
9. [Best Practices](#best-practices)
10. [Performance Optimization](#performance-optimization)

---

## 🎯 Introduction

Feature Engineering is the process of using domain knowledge to select, transform, and create features from raw data to make machine learning algorithms work better. It's a crucial step that can significantly impact model performance and interpretability.

### Key Objectives:
- **Improve Model Performance**: Create features that better represent the underlying patterns
- **Enhance Interpretability**: Transform complex relationships into understandable features
- **Handle Data Limitations**: Address missing values, outliers, and data type issues
- **Reduce Complexity**: Create compact, informative representations
- **Domain Integration**: Incorporate business knowledge into feature creation

---

## 🧮 Mathematical Foundation

### 1. Statistical Transformations

**Log Transformation:**
```
y' = log(y + c)
```
Where:
- `y` = original value
- `c` = constant (usually 1) to handle zero values
- Used for right-skewed data to normalize distribution

**Mathematical Properties:**
- Variance stabilization: Var(log(Y)) ≈ constant for multiplicative relationships
- Monotonic transformation: preserves order
- Handles heteroscedasticity: Var(Y) = f(E[Y])

**Box-Cox Transformation:**
```
y(λ) = {
    (y^λ - 1) / λ,  if λ ≠ 0
    log(y),           if λ = 0
}
```

Where λ is estimated using maximum likelihood.

### 2. Information Theory

**Entropy:**
```
H(X) = -Σ p(x) log(p(x))
```
Measures the uncertainty in a random variable.

**Information Gain:**
```
IG(S, X) = H(S) - Σ |Sv| H(Sv)
```
Used in decision trees to measure feature importance.

**Mutual Information:**
```
I(X; Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
```
Measures the dependency between variables.

### 3. Distance Metrics

**Euclidean Distance:**
```
d(x, y) = √(Σ(xi - yi)²)
```

**Mahalanobis Distance:**
```
D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```
Accounts for covariance between features.

---

## 🔧 Data Preprocessing

### 1. Missing Value Treatment

**MCAR (Missing Completely At Random):**
```python
# Test for MCAR using Little's MCAR test
from scipy.stats import chi2
def little_mcar_test(data, alpha=0.05):
    # Implementation of Little's test
    chi_stat, p_value = chi2_test(data)
    return p_value > alpha
```

**Imputation Methods:**

| Method | Formula | Best For | Mathematical Properties |
|---------|---------|------------------------|
| **Mean** | x̄ = Σxi / n | Normal data, MCAR | Unbiased, preserves mean |
| **Median** | x̃ = middle value | Skewed data, outliers | Robust to outliers |
| **KNN** | x̂ = weighted average of k neighbors | Complex patterns | Non-parametric |
| **MICE** | Multiple imputation by chained equations | MAR data | Preserves relationships |
| **EM Algorithm** | θ̂ = argmaxθ P(X|θ) | Complex missingness | Maximum likelihood |

**Implementation:**
```python
# Multiple Imputation by Chained Equations (MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# MICE imputation
imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42
)
X_imputed = imputer.fit_transform(X)
```

### 2. Outlier Detection and Treatment

**Statistical Methods:**

**Z-Score:**
```
Z = (x - μ) / σ
```
Outlier if |Z| > 3 (99.7% confidence)

**Modified Z-Score:**
```
MAD = median(|xi - x̃|)
Modified Z = 0.6745 × (xi - x̃) / MAD
```

**IQR Method:**
```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

**Multivariate Outlier Detection:**

**Mahalanobis Distance:**
```python
import numpy as np
from scipy.stats import chi2

def mahalanobis_outliers(X, threshold=0.95):
    cov_matrix = np.cov(X.T)
    inv_cov = np.linalg.inv(cov_matrix)
    mean = np.mean(X, axis=0)
    
    distances = []
    for x in X:
        d = np.sqrt((x - mean).T @ inv_cov @ (x - mean))
        distances.append(d)
    
    critical_value = chi2.ppf(threshold, df=len(X[0]))
    outliers = distances > critical_value
    return outliers
```

**Isolation Forest:**
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expected outlier proportion
    random_state=42
)
outliers = iso_forest.fit_predict(X)
```

---

## 🔄 Feature Transformation

### 1. Scaling and Normalization

**Standardization (Z-score):**
```
z = (x - μ) / σ
```
Properties:
- Mean = 0, Standard Deviation = 1
- Preserves shape, removes scale effects

**Min-Max Scaling:**
```
x' = (x - min(x)) / (max(x) - min(x))
```
Properties:
- Range [0, 1]
- Sensitive to outliers

**Robust Scaling:**
```
x' = (x - median(x)) / IQR
```
Properties:
- Median centered, IQR scaled
- Robust to outliers

### 2. Non-linear Transformations

**Polynomial Features:**
```
φ(x) = [1, x, x², x³, ..., xᵈ]
```

**Interaction Terms:**
```
φ(x₁, x₂) = [x₁, x₂, x₁×x₂, x₁², x₂², x₁²×x₂, x₁×x₂²]
```

**Trigonometric Features:**
```
φ(x) = [sin(x), cos(x), tan(x), sin(2x), cos(2x)]
```
Useful for cyclical data (time, angles).

### 3. Dimensionality Reduction

**Principal Component Analysis (PCA):**
```
Covariance Matrix: Σ = (1/n) × (X - X̄)ᵀ(X - X̄)
Eigen Decomposition: Σv = λv
Principal Components: Y = XV
```

**Mathematical Properties:**
- Variance explained: λk / Σλi
- Orthogonal components: viᵀvj = 0 for i ≠ j
- Optimal reconstruction: minimizes mean squared error

**Implementation:**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA with explained variance analysis
pca = PCA()
X_pca = pca.fit_transform(X)

# Scree plot
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), 
         np.cumsum(explained_variance) * 100, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA Scree Plot')
plt.grid(True)
plt.show()
```

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
```python
from sklearn.manifold import TSNE

# t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.title('t-SNE Visualization')
plt.colorbar()
plt.show()
```

---

## 🏷️ Encoding Techniques

### 1. Nominal Encoding

**One-Hot Encoding:**
```
For categorical variable C with k categories:
Encoded = [e₁, e₂, ..., eₖ] where ei ∈ {0, 1}
Σei = 1 for each instance
```

**Mathematical Representation:**
```
X_encoded = X × W
where W is the encoding matrix
```

**Implementation:**
```python
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp

# Memory-efficient one-hot encoding
encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
X_sparse = encoder.fit_transform(X)

# Convert back to dense if needed
X_dense = X_sparse.toarray()
```

**Dummy Variable Trap:**
```python
# Avoid dummy variable trap
encoder = OneHotEncoder(drop='first', sparse=True)
X_encoded = encoder.fit_transform(X)
```

### 2. Ordinal Encoding

**Label Encoding:**
```
f: categories → {1, 2, 3, ..., k}
```

**Ordinal Encoding:**
```
f: [low, medium, high] → [0, 1, 2]
```

### 3. Target Encoding

**Mean Encoding:**
```
Encode(c) = mean(target | category = c)
```

**Smoothed Target Encoding:**
```
Encode(c) = (mean(target | category = c) × n + global_mean) / (n + global_count)
```

Where n is the smoothing parameter.

**Implementation:**
```python
import pandas as pd
import numpy as np

def target_encoding(X, y, categorical_cols, smoothing=1):
    X_encoded = X.copy()
    
    for col in categorical_cols:
        # Calculate mean target for each category
        target_mean = y.groupby(X[col]).mean()
        global_mean = y.mean()
        count = X[col].value_counts()
        
        # Apply smoothed encoding
        X_encoded[f'{col}_target_mean'] = X[col].map(target_mean)
        X_encoded[f'{col}_target_smooth'] = (
            (target_mean * count + global_mean * smoothing) / 
            (count + smoothing)
        ).map(X[col])
    
    return X_encoded

# Usage
X_encoded = target_encoding(X, y, ['category_col'])
```

### 4. Advanced Encoding

**Binary Encoding:**
```python
def binary_encoding(X, categorical_col):
    """Converts categorical to binary representation"""
    unique_values = X[categorical_col].unique()
    binary_length = np.ceil(np.log2(len(unique_values))).astype(int)
    
    encoded = []
    for value in X[categorical_col]:
        binary = format(unique_values.tolist().index(value), f'0{binary_length}b')
        encoded.append([int(bit) for bit in binary])
    
    return np.array(encoded)
```

**Hashing Trick:**
```python
from sklearn.feature_extraction import FeatureHasher

# Feature hashing for high-cardinality features
hasher = FeatureHasher(
    n_features=10,  # Number of output features
    input_type='string'
)
X_hashed = hasher.fit_transform(X[categorical_col])
```

**Embedding:**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Neural embedding for categorical variables
def create_embedding(input_dim, output_dim):
    input_layer = layers.Input(shape=(1,))
    embedding = layers.Embedding(input_dim, output_dim)(input_layer)
    return Model(inputs=input_layer, outputs=embedding)

# Usage
embedding_layer = create_embedding(num_categories, embedding_dim)
embedded_features = embedding_layer(X_categorical)
```

---

## 🎯 Feature Selection

### 1. Statistical Methods

**Correlation-based Selection:**
```python
import numpy as np
from scipy.stats import pearsonr

def correlation_selection(X, y, threshold=0.8):
    selected_features = []
    for i in range(X.shape[1]):
        corr, p_value = pearsonr(X[:, i], y)
        if abs(corr) > threshold and p_value < 0.05:
            selected_features.append(i)
    return selected_features
```

**Chi-Square Test:**
```python
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Chi-square test for categorical features
le = LabelEncoder()
y_encoded = le.fit_transform(y)
chi2_scores = chi2(X, y_encoded)
p_values = chi2_scores[1]

# Select features based on p-values
selected_features = np.where(p_values < 0.05)[0]
```

### 2. Model-based Selection

**Recursive Feature Elimination (RFE):**
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X, y)

selected_features = selector.support_
ranking = selector.ranking_
```

**LASSO (L1 Regularization):**
```python
from sklearn.linear_model import LassoCV

# LASSO for feature selection
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)

# Features with non-zero coefficients
selected_features = np.where(lasso.coef_ != 0)[0]
feature_importance = np.abs(lasso.coef_)
```

### 3. Information-theoretic Selection

**Mutual Information:**
```python
from sklearn.feature_selection import mutual_info_classif

# Mutual information for classification
mi_scores = mutual_info_classif(X, y)
selected_features = np.where(mi_scores > threshold)[0]
```

**Information Gain:**
```python
from sklearn.feature_selection import mutual_info_classif

# For decision tree-based models
info_gain = mutual_info_classif(X, y, discrete_features=True)
```

---

## 🚀 Advanced Techniques

### 1. Automated Feature Engineering

**Polynomial Features:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Automated polynomial feature generation
poly_features = PolynomialFeatures(degree=3, include_bias=False)
model = Pipeline([
    ('poly', poly_features),
    ('linear', LinearRegression())
])
model.fit(X, y)
```

**Interaction Features:**
```python
from sklearn.preprocessing import PolynomialFeatures

# Automatically create interaction terms
interaction = PolynomialFeatures(
    degree=2, 
    interaction_only=True, 
    include_bias=False
)
X_interaction = interaction.fit_transform(X)
```

### 2. Time Series Features

**Temporal Features:**
```python
import pandas as pd
import numpy as np

def create_time_features(df, datetime_col):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Extract temporal components
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['dayofweek'] = df[datetime_col].dt.dayofweek
    df['hour'] = df[datetime_col].dt.hour
    df['minute'] = df[datetime_col].dt.minute
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    return df
```

**Lag Features:**
```python
def create_lag_features(df, value_col, lags=[1, 7, 30]):
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
    return df

# Rolling window features
def create_rolling_features(df, value_col, windows=[7, 14, 30]):
    for window in windows:
        df[f'{value_col}_rolling_mean_{window}'] = (
            df[value_col].rolling(window=window).mean()
        )
        df[f'{value_col}_rolling_std_{window}'] = (
            df[value_col].rolling(window=window).std()
        )
    return df
```

### 3. Text Feature Engineering

**TF-IDF:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF for text data
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)
X_tfidf = tfidf.fit_transform(text_data)
```

**Word Embeddings:**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Topic modeling for feature extraction
vectorizer = CountVectorizer(max_features=1000)
X_counts = vectorizer.fit_transform(text_data)

lda = LatentDirichletAllocation(n_components=10, random_state=42)
X_topics = lda.fit_transform(X_counts)
```

### 4. Image Feature Engineering

**Histogram of Oriented Gradients (HOG):**
```python
from skimage.feature import hog

def extract_hog_features(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features
```

**Color Histograms:**
```python
import cv2
import numpy as np

def extract_color_histogram(image):
    # Calculate histogram for each color channel
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256], [0, 256])
    
    # Normalize and concatenate
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    
    return np.concatenate([hist_b, hist_g, hist_r])
```

---

## 📊 Project Examples

### 1. Missing Value Handling

**Dataset**: Titanic dataset with missing age and embarked values

**Techniques Applied**:
- Mean/Median imputation for numerical variables
- Mode imputation for categorical variables
- Multiple imputation for complex patterns

```python
# Comprehensive missing value analysis
import missingno as msno
import matplotlib.pyplot as plt

# Missing value patterns
msno.matrix(df)
plt.title('Missing Value Patterns')
plt.show()

# Missing value correlation
msno.heatmap(df)
plt.title('Missing Value Correlation')
plt.show()
```

### 2. Imbalanced Dataset Handling

**SMOTE (Synthetic Minority Over-sampling):**

**Mathematical Foundation:**
```
For minority sample xi:
1. Find k nearest neighbors from majority class
2. Generate synthetic sample: xsynth = xi + random(0, 1) × (xj - xi)
```

**Implementation:**
```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# Apply SMOTE
smote = SMOTE(
    sampling_strategy='auto',  # Balance all classes
    random_state=42,
    k_neighbors=5
)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Visualize before and after
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Before SMOTE
ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
ax1.set_title('Before SMOTE')

# After SMOTE
ax2.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, alpha=0.6)
ax2.set_title('After SMOTE')

plt.tight_layout()
plt.show()
```

### 3. Encoding Techniques

**Multi-level Encoding Strategy:**
```python
def advanced_encoding(df, categorical_columns):
    """
    Apply different encoding based on cardinality
    """
    encoded_df = df.copy()
    
    for col in categorical_columns:
        cardinality = df[col].nunique()
        
        if cardinality == 2:
            # Binary: Label encoding
            encoded_df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col])
        
        elif cardinality <= 10:
            # Low cardinality: One-hot encoding
            encoder = OneHotEncoder(drop='first', sparse=True)
            encoded = encoder.fit_transform(df[[col]])
            encoded_df = pd.concat([encoded_df, pd.DataFrame(encoded.toarray(), 
                columns=[f'{col}_{cat}' for cat in encoder.categories_[0][1:]])], axis=1)
        
        elif cardinality <= 100:
            # Medium cardinality: Target encoding
            target_mean = df.groupby(col)['target'].mean()
            encoded_df[f'{col}_target'] = df[col].map(target_mean)
        
        else:
            # High cardinality: Feature hashing
            hasher = FeatureHasher(n_features=10, input_type='string')
            encoded = hasher.fit_transform(df[col].astype(str))
            encoded_df = pd.DataFrame(encoded, 
                columns=[f'{col}_hash_{i}' for i in range(10)])
    
    return encoded_df
```

### 4. Outlier Detection and Treatment

**Comprehensive Outlier Analysis:**
```python
def comprehensive_outlier_analysis(df, numerical_columns):
    """
    Multiple outlier detection methods
    """
    results = {}
    
    for col in numerical_columns:
        data = df[col].dropna()
        
        # Statistical methods
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Z-score method
        z_scores = np.abs((data - data.mean()) / data.std())
        
        # Modified Z-score
        median = data.median()
        MAD = np.median(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / MAD
        
        results[col] = {
            'iqr_outliers': ((data < lower_bound) | (data > upper_bound)).sum(),
            'z_outliers': (z_scores > 3).sum(),
            'modified_z_outliers': (np.abs(modified_z) > 3.5).sum(),
            'bounds': (lower_bound, upper_bound)
        }
    
    return results

# Visualization
def plot_outlier_comparison(df, column):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original distribution
    axes[0, 0].hist(df[column], bins=50, alpha=0.7)
    axes[0, 0].set_title('Original Distribution')
    
    # Box plot
    axes[0, 1].boxplot(df[column])
    axes[0, 1].set_title('Box Plot')
    
    # After outlier removal
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_data = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    axes[1, 0].hist(cleaned_data[column], bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Cleaned Distribution')
    
    axes[1, 1].boxplot(cleaned_data[column])
    axes[1, 1].set_title('Cleaned Box Plot')
    
    plt.tight_layout()
    plt.show()
```

---

## 🎯 Best Practices

### 1. Data Leakage Prevention

**Temporal Validation:**
```python
# Correct: Time-based split for time series data
train = df[df['date'] < '2023-01-01']
test = df[df['date'] >= '2023-01-01']

# Incorrect: Random split on time series data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # WRONG!
```

**Feature Engineering Pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Prevent data leakage with proper pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

### 2. Cross-Validation Strategy

**Stratified K-Fold:**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Maintain class distribution in folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

**Time Series Cross-Validation:**
```python
from sklearn.model_selection import TimeSeriesSplit

# Account for temporal dependencies
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

### 3. Feature Importance Analysis

**Permutation Importance:**
```python
from sklearn.inspection import permutation_importance

# Model-agnostic feature importance
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

sorted_idx = result.importances_mean.argsort()[::-1]
plt.bar(range(X.shape[1]), result.importances_mean[sorted_idx])
plt.xticks(range(X.shape[1]), np.array(feature_names)[sorted_idx], rotation=45)
plt.title('Permutation Importance')
plt.show()
```

**SHAP Values:**
```python
import shap

# Explain individual predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type='bar')
```

---

## ⚡ Performance Optimization

### 1. Memory Efficiency

**Sparse Matrices:**
```python
from scipy import sparse
import numpy as np

# Memory-efficient operations
def memory_efficient_encoding(X, categorical_col):
    # Create sparse matrix instead of dense
    encoded = sparse.csr_matrix(
        (np.arange(len(X))[:, None] == X[categorical_col].values).astype(int)
    )
    return encoded

# Usage
X_sparse = memory_efficient_encoding(X, 'category')
```

**Chunking for Large Datasets:**
```python
def process_large_dataset(file_path, chunk_size=10000):
    results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results, ignore_index=True)
```

### 2. Computational Efficiency

**Vectorized Operations:**
```python
# Efficient: Vectorized operations
df['interaction'] = df['feature1'] * df['feature2']

# Inefficient: Row-wise operations
# df['interaction'] = df.apply(lambda row: row['feature1'] * row['feature2'], axis=1)
```

**Parallel Processing:**
```python
from multiprocessing import Pool
import numpy as np

def parallel_feature_engineering(X, n_processes=4):
    with Pool(n_processes) as pool:
        # Split data and process in parallel
        chunks = np.array_split(X, n_processes)
        results = pool.map(process_chunk, chunks)
    
    return np.concatenate(results)
```

### 3. GPU Acceleration

**RAPIDS (cuDF):**
```python
# GPU-accelerated operations
import cudf  # RAPIDS library

# Convert to GPU DataFrame
df_gpu = cudf.from_pandas(df)

# GPU-accelerated operations
df_gpu['feature'] = df_gpu['col1'] * df_gpu['col2']
```

---

## 📈 Project Structure

```
Feature Engineering/
├── README.md                          # This file
├── 1.0-Handling_Missing_values.ipynb # Missing value techniques
├── 2.0-Handling_Imbalance_Dataset.ipynb # Imbalanced data handling
├── 3.0-SMOTE.ipynb                  # Synthetic oversampling
├── 4.0-Handling+Outliers.ipynb         # Outlier detection
├── 5.0-Nominal+or+OHE.ipynb          # One-hot encoding
├── 7.0-Label+and+Ordinal.ipynb        # Label/ordinal encoding
└── 8.0-Target+Guided+Ordinal+Encoding.ipynb # Target encoding
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Install required packages
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy missingno
pip install imblearn shap plotly
pip install cudf-cu11  # For GPU acceleration
```

### 2. Running the Notebooks
1. **Missing Values**: `1.0-Handling_Missing_values.ipynb`
   - MCAR/MAR/MNAR analysis
   - Multiple imputation techniques
   - Missing value patterns visualization

2. **Imbalanced Data**: `2.0-Handling_Imbalance_Dataset.ipynb`
   - Upsampling and downsampling
   - SMOTE synthetic generation
   - Class balance visualization

3. **Outlier Detection**: `4.0-Handling+Outliers.ipynb`
   - Statistical and ML-based detection
   - Multivariate outlier analysis
   - Treatment strategies

4. **Encoding Techniques**: `5.0-Nominal+or+OHE.ipynb`
   - One-hot encoding implementation
   - Label and ordinal encoding
   - Memory-efficient encoding

5. **Advanced Encoding**: `8.0-Target+Guided+Ordinal+Encoding.ipynb`
   - Target-guided encoding
   - High-cardinality handling
   - Encoding optimization

### 3. Custom Feature Engineering Template
```python
class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.fitted_encoders = {}
    
    def fit(self, X, y=None):
        """Learn encoding parameters"""
        # Implement fitting logic
        pass
    
    def transform(self, X):
        """Apply learned transformations"""
        # Implement transformation logic
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.transform(self.fit(X, y))

# Usage
engineer = FeatureEngineer(config)
X_transformed = engineer.fit_transform(X_train, y_train)
X_test_transformed = engineer.transform(X_test)
```

---

## 📊 Key Insights from Projects

### Missing Value Analysis
- **Pattern Detection**: Missing values often follow MAR (Missing At Random) patterns
- **Imputation Impact**: Method choice significantly affects model performance
- **Multiple Imputation**: Superior to single imputation for complex patterns

### Imbalanced Data Handling
- **SMOTE Effectiveness**: Synthetic samples improve minority class representation
- **Class Balance**: Critical for algorithm performance (especially distance-based)
- **Evaluation Metrics**: Use F1-score, AUC-ROC for imbalanced datasets

### Encoding Strategies
- **Cardinality Matters**: Different encoding strategies for different category counts
- **Memory Efficiency**: Sparse matrices essential for high-cardinality features
- **Target Encoding**: Powerful but requires careful validation to prevent leakage

### Outlier Detection
- **Method Selection**: No single method works best for all datasets
- **Domain Knowledge**: Important for setting appropriate thresholds
- **Impact Analysis**: Outliers can significantly affect model training and predictions

---

## 🎯 Conclusion

Feature Engineering is both an art and a science that combines statistical knowledge, domain expertise, and computational techniques. The key takeaways are:

1. **Mathematical Foundation**: Understanding the underlying mathematics ensures proper application
2. **Systematic Approach**: Follow structured processes for consistent, reproducible results
3. **Validation Strategy**: Always validate feature engineering decisions with proper cross-validation
4. **Performance Awareness**: Consider computational and memory constraints in real-world applications
5. **Domain Integration**: Combine statistical techniques with business knowledge for optimal results

The projects in this repository demonstrate comprehensive feature engineering techniques across various domains and challenges. Adapt these methods to your specific datasets and requirements, always validating the impact on model performance.

---

*Last Updated: January 2026*
