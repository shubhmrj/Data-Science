# K-Nearest Neighbors (KNN) - Comprehensive Guide

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Distance Metrics](#distance-metrics)
4. [KNN Classification](#knn-classification)
5. [KNN Regression](#knn-regression)
6. [Algorithm Variants](#algorithm-variants)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Advanced Techniques](#advanced-techniques)
9. [Performance Analysis](#performance-analysis)
10. [Practical Applications](#practical-applications)

---

## 🎯 Introduction

K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm used for both classification and regression tasks. Unlike parametric methods that learn a model from training data, KNN makes predictions by finding the k most similar training instances to a query point and aggregating their outputs.

### Key Characteristics:
- **Lazy Learning**: No explicit training phase, stores all training data
- **Non-parametric**: No assumptions about data distribution
- **Instance-based**: Predictions based on local neighborhood
- **Versatile**: Works for both classification and regression
- **Interpretable**: Easy to understand and visualize

---

## 🧮 Mathematical Foundation

### 1. Basic KNN Algorithm

For a query point xₚ, the KNN prediction is:

**Classification:**
```
ŷ = mode({y₁, y₂, ..., yₖ})
```
where {y₁, y₂, ..., yₖ} are the labels of the k nearest neighbors.

**Regression:**
```
ŷ = (1/k) × Σᵢ₌₁ᵏ yᵢ
```
where yᵢ are the target values of the k nearest neighbors.

### 2. Distance-based Weighting

**Weighted KNN Classification:**
```
ŷ = argmaxⱼ Σᵢ₌₁ᵏ wᵢ × I(yᵢ = j)
```

**Weighted KNN Regression:**
```
ŷ = (Σᵢ₌₁ᵏ wᵢ × yᵢ) / (Σᵢ₌₁ᵏ wᵢ)
```

where weights wᵢ are typically:
```
wᵢ = 1 / (d(xₚ, xᵢ) + ε)
```

### 3. Probability Estimation

**Posterior Probability:**
```
P(y = j | xₚ) = (1/k) × Σᵢ₌₁ᵏ I(yᵢ = j)
```

**Weighted Probability:**
```
P(y = j | xₚ) = (Σᵢ₌₁ᵏ wᵢ × I(yᵢ = j)) / (Σᵢ₌₁ᵏ wᵢ)
```

---

## 📏 Distance Metrics

### 1. Euclidean Distance

**Formula:**
```
d(x, y) = √(Σᵢ₌₁ᵈ (xᵢ - yᵢ)²)
```

**Properties:**
- Satisfies triangle inequality
- Sensitive to feature scaling
- Most commonly used metric

**Implementation:**
```python
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Vectorized implementation
def euclidean_distance_batch(X, y):
    return np.sqrt(np.sum((X - y) ** 2, axis=1))
```

### 2. Manhattan Distance

**Formula:**
```
d(x, y) = Σᵢ₌₁ᵈ |xᵢ - yᵢ|
```

**Properties:**
- Less sensitive to outliers
- Computationally efficient
- Good for high-dimensional data

### 3. Minkowski Distance

**Formula:**
```
d(x, y) = (Σᵢ₌₁ᵈ |xᵢ - yᵢ|ᵖ)^(1/p)
```

Special cases:
- p = 1: Manhattan distance
- p = 2: Euclidean distance
- p → ∞: Chebyshev distance

### 4. Cosine Similarity

**Formula:**
```
cos(θ) = (x · y) / (||x|| × ||y||)
```

**Distance:**
```
d(x, y) = 1 - cos(θ)
```

**Properties:**
- Measures angular similarity
- Scale-invariant
- Good for text data

### 5. Mahalanobis Distance

**Formula:**
```
d(x, y) = √((x - y)ᵀ Σ⁻¹ (x - y))
```

where Σ is the covariance matrix.

**Properties:**
- Accounts for feature correlations
- Scale-invariant
- Accounts for data distribution

---

## 🎯 KNN Classification

### 1. Basic Classification Algorithm

**Pseudocode:**
```
function KNN_Classify(X_train, y_train, x_test, k):
    distances = []
    
    for each training point x_i in X_train:
        d = distance(x_test, x_i)
        distances.append((d, y_i))
    
    sort distances by d
    k_nearest = distances[:k]
    
    labels = [label for (_, label) in k_nearest]
    prediction = mode(labels)
    
    return prediction
```

### 2. Implementation

```python
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Calculate distances
            distances = self._calculate_distances(x)
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            
            # Weighted prediction
            if self.weights == 'uniform':
                prediction = Counter(k_labels).most_common(1)[0][0]
            else:  # distance-weighted
                k_distances = distances[k_indices]
                weights = 1 / (k_distances + 1e-8)
                weighted_votes = {}
                
                for label, weight in zip(k_labels, weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                
                prediction = max(weighted_votes, key=weighted_votes.get)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _calculate_distances(self, x):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.metric == 'cosine':
            dot_product = np.dot(self.X_train, x)
            norm_x = np.linalg.norm(x)
            norm_train = np.linalg.norm(self.X_train, axis=1)
            return 1 - (dot_product / (norm_x * norm_train))
```

### 3. Decision Boundary Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

def plot_decision_boundary(X, y, k=5):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Train KNN
    knn = CustomKNNClassifier(k=k)
    knn.fit(X, y)
    
    # Predict on mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
    plt.title(f'KNN Decision Boundary (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Generate sample data and plot
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)
plot_decision_boundary(X, y, k=5)
```

---

## 📈 KNN Regression

### 1. Basic Regression Algorithm

**Pseudocode:**
```
function KNN_Regress(X_train, y_train, x_test, k):
    distances = []
    
    for each training point x_i in X_train:
        d = distance(x_test, x_i)
        distances.append((d, y_i))
    
    sort distances by d
    k_nearest = distances[:k]
    
    values = [value for (_, value) in k_nearest]
    prediction = mean(values)
    
    return prediction
```

### 2. Implementation

```python
class CustomKNNRegressor(BaseEstimator):
    def __init__(self, k=5, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Calculate distances
            distances = self._calculate_distances(x)
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_values = self.y_train[k_indices]
            
            # Weighted prediction
            if self.weights == 'uniform':
                prediction = np.mean(k_values)
            else:  # distance-weighted
                k_distances = distances[k_indices]
                weights = 1 / (k_distances + 1e-8)
                prediction = np.sum(k_values * weights) / np.sum(weights)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _calculate_distances(self, x):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
```

### 3. Regression Surface Visualization

```python
def plot_regression_surface(X, y, k=5):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Train KNN
    knn = CustomKNNRegressor(k=k)
    knn.fit(X, y)
    
    # Predict on mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig = plt.figure(figsize=(15, 10))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.8)
    ax1.scatter(X[:, 0], X[:, 1], y, color='red', s=50)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Target')
    ax1.set_title(f'KNN Regression Surface (k={k})')
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(xx, yy, Z, levels=20, cmap='viridis')
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Regression Contour Plot')
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.show()

# Generate sample regression data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
plot_regression_surface(X, y, k=5)
```

---

## 🔧 Algorithm Variants

### 1. Ball Tree Algorithm

**Concept:** Hierarchical space partitioning for efficient nearest neighbor search.

**Mathematical Foundation:**
- Ball: B(x, r) = {y : d(x, y) ≤ r}
- Tree structure: Each node represents a ball containing a subset of points

**Implementation:**
```python
from sklearn.neighbors import BallTree

# Ball Tree for efficient neighbor search
tree = BallTree(X_train, leaf_size=40)
distances, indices = tree.query(X_test, k=5)
```

### 2. KD-Tree Algorithm

**Concept:** Binary space partitioning along coordinate axes.

**Mathematical Foundation:**
- Recursively split space along median of largest variance dimension
- Each node represents a hyper-rectangle region

**Implementation:**
```python
from sklearn.neighbors import KDTree

# KD Tree for low-dimensional data
tree = KDTree(X_train, leaf_size=40)
distances, indices = tree.query(X_test, k=5)
```

### 3. Approximate Nearest Neighbors

**LSH (Locality Sensitive Hashing):**
```python
# Approximate nearest neighbors for large datasets
from sklearn.neighbors import NearestNeighbors

# Use approximate algorithms
nn = NearestNeighbors(n_neighbors=5, algorithm='auto', leaf_size=30)
nn.fit(X_train)
distances, indices = nn.kneighbors(X_test)
```

---

## 🎛️ Hyperparameter Optimization

### 1. K Selection

**Cross-Validation Approach:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def find_optimal_k(X, y, max_k=31):
    cv_scores = []
    k_values = range(1, max_k + 1, 2)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Plot performance vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, 'bo-')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN Performance vs k')
    plt.grid(True)
    plt.show()
    
    optimal_k = k_values[np.argmax(cv_scores)]
    return optimal_k, max(cv_scores)

# Usage
optimal_k, best_score = find_optimal_k(X_train, y_train)
print(f"Optimal k: {optimal_k}, Best score: {best_score:.3f}")
```

### 2. Grid Search with Multiple Parameters

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': range(1, 31, 2),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
```

### 3. Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

def knn_cv_score(params):
    k, weights, metric = params
    
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        metric=metric
    )
    
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    return -scores.mean()  # Minimize negative accuracy

space = [
    Integer(1, 30, name='n_neighbors'),
    Categorical(['uniform', 'distance'], name='weights'),
    Categorical(['euclidean', 'manhattan', 'cosine'], name='metric')
]

result = gp_minimize(knn_cv_score, space, n_calls=50, random_state=42)
print(f"Best parameters: k={result.x[0]}, weights={result.x[1]}, metric={result.x[2]}")
```

---

## 🚀 Advanced Techniques

### 1. Feature Scaling and Normalization

**Standardization:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
```

**Min-Max Scaling:**
```python
from sklearn.preprocessing import MinMaxScaler

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
```

### 2. Dimensionality Reduction

**PCA + KNN:**
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Pipeline with PCA
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"PCA + KNN CV Score: {scores.mean():.3f}")
```

### 3. Ensemble Methods

**Bagging with KNN:**
```python
from sklearn.ensemble import BaggingClassifier

# Bagging KNN for variance reduction
bagging_knn = BaggingClassifier(
    base_estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)

bagging_knn.fit(X_train, y_train)
```

### 4. Adaptive KNN

**Distance-based K Selection:**
```python
class AdaptiveKNN:
    def __init__(self, base_k=5, max_k=50):
        self.base_k = base_k
        self.max_k = max_k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Adaptive k based on distance distribution
            sorted_distances = np.sort(distances)
            
            # Find elbow point in distance curve
            k = self._find_elbow_k(sorted_distances)
            k = min(k, self.max_k)
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:k]
            k_labels = self.y_train[k_indices]
            
            # Majority vote
            prediction = Counter(k_labels).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _find_elbow_k(self, distances):
        # Simple elbow detection using second derivative
        for k in range(self.base_k, len(distances)):
            if k < len(distances) - 1:
                second_derivative = distances[k+1] - 2*distances[k] + distances[k-1]
                if second_derivative > 0:  # Elbow point
                    return k
        return self.base_k
```

---

## 📊 Performance Analysis

### 1. Computational Complexity

| Operation | Brute Force | KD-Tree | Ball Tree |
|-----------|-------------|---------|-----------|
| Training | O(1) | O(n log n) | O(n log n) |
| Prediction | O(n × d) | O(log n × d) | O(log n × d) |
| Memory | O(n × d) | O(n × d) | O(n × d) |

Where:
- n = number of training samples
- d = number of dimensions

### 2. Curse of Dimensionality

**Distance Concentration:**
```
lim(d→∞) Var(d(x, y)) / E[d(x, y)]² → 0
```

**Impact on KNN:**
- In high dimensions, all points become equidistant
- Nearest neighbor concept becomes meaningless
- Dimensionality reduction becomes crucial

### 3. Bias-Variance Tradeoff

**Bias:**
- High k → High bias (underfitting)
- Low k → Low bias (overfitting)

**Variance:**
- High k → Low variance
- Low k → High variance

**Optimal k:**
```
k* = argminₖ (Bias²(k) + Variance(k))
```

### 4. Performance Metrics

**Classification:**
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_classifier(y_true, y_pred, class_names=None):
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
```

**Regression:**
```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_regressor(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()
```

---

## 🎯 Practical Applications

### 1. Image Classification

**Pixel-based KNN:**
```python
def flatten_images(images):
    """Flatten images for KNN processing"""
    return images.reshape(images.shape[0], -1)

# Example with MNIST-like data
X_flat = flatten_images(X_images)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_flat, y_labels)
```

### 2. Recommendation Systems

**Item-based KNN:**
```python
def knn_recommendations(user_item_matrix, user_id, k=10):
    """Find similar users and recommend items"""
    user_vector = user_item_matrix[user_id]
    
    # Calculate similarities
    similarities = []
    for other_user in user_item_matrix.index:
        if other_user != user_id:
            similarity = cosine_similarity(user_vector, user_item_matrix[other_user])
            similarities.append((similarity, other_user))
    
    # Get k most similar users
    similarities.sort(reverse=True)
    similar_users = similarities[:k]
    
    # Recommend items liked by similar users
    recommendations = {}
    for similarity, other_user in similar_users:
        for item, rating in user_item_matrix.loc[other_user].items():
            if rating > 0 and user_vector[item] == 0:
                recommendations[item] = recommendations.get(item, 0) + similarity * rating
    
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
```

### 3. Anomaly Detection

**Distance-based Anomaly Detection:**
```python
class KNNAnomalyDetector:
    def __init__(self, k=5, threshold_percentile=95):
        self.k = k
        self.threshold_percentile = threshold_percentile
    
    def fit(self, X):
        self.X_train = X
        # Calculate average distance to k nearest neighbors for training data
        self.avg_distances = []
        for x in X:
            distances = np.sqrt(np.sum((X - x) ** 2, axis=1))
            k_nearest = np.sort(distances)[1:self.k+1]  # Exclude self
            avg_dist = np.mean(k_nearest)
            self.avg_distances.append(avg_dist)
        
        # Set threshold
        self.threshold = np.percentile(self.avg_distances, self.threshold_percentile)
        return self
    
    def predict(self, X):
        anomalies = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_nearest = np.sort(distances)[:self.k]
            avg_dist = np.mean(k_nearest)
            
            is_anomaly = avg_dist > self.threshold
            anomalies.append(is_anomaly)
        
        return np.array(anomalies)
```

---

## 📈 Project Structure

```
K-Nearest-Neighbour/
├── README.md                          # This file
├── Classifier/
│   ├── KNNClassifier.ipynb           # Basic classification implementation
│   └── advance-knn-classifier.ipynb   # Advanced classification techniques
└── Regressor/
    └── KNNRegressor.ipynb             # Regression implementation
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn
pip install scipy plotly skopt
```

### 2. Running the Notebooks
1. **Basic Classification**: `Classifier/KNNClassifier.ipynb`
   - Implements basic KNN classification
   - Demonstrates distance calculations
   - Shows performance evaluation

2. **Advanced Classification**: `Classifier/advance-knn-classifier.ipynb`
   - Advanced KNN variants
   - Hyperparameter optimization
   - Performance analysis

3. **Regression**: `Regressor/KNNRegressor.ipynb`
   - KNN for regression tasks
   - Weighted predictions
   - Evaluation metrics

### 3. Quick Start Example
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

---

## 📊 Key Insights from Projects

### Classification Analysis
- **Optimal k**: Typically between 3-15 for most datasets
- **Distance Metrics**: Euclidean works well for normalized data
- **Feature Scaling**: Critical for distance-based algorithms
- **Curse of Dimensionality**: Performance degrades in high dimensions

### Regression Analysis
- **Smoothing Effect**: Higher k creates smoother predictions
- **Local Patterns**: KNN captures non-linear relationships well
- **Noise Sensitivity**: Outliers can significantly affect predictions
- **Computational Cost**: Prediction time scales with dataset size

### Performance Considerations
- **Training Time**: O(1) for basic KNN (lazy learning)
- **Prediction Time**: O(n × d) for brute force, O(log n × d) for tree-based
- **Memory Usage**: O(n × d) to store all training data
- **Scalability**: Tree-based methods scale better for large datasets

---

## 🎯 Conclusion

K-Nearest Neighbors is a fundamental algorithm in machine learning that exemplifies instance-based learning. Its simplicity and interpretability make it an excellent baseline algorithm and educational tool.

**Key Takeaways:**
1. **Simplicity**: Easy to understand and implement
2. **Versatility**: Works for both classification and regression
3. **Non-parametric**: No assumptions about data distribution
4. **Local Learning**: Captures local patterns in data
5. **Interpretable**: Easy to explain predictions

**Best Practices:**
- Always scale features before applying KNN
- Use cross-validation to select optimal k
- Consider dimensionality reduction for high-dimensional data
- Use tree-based algorithms for large datasets
- Apply weighted voting for imbalanced datasets

**When to Use KNN:**
- Small to medium-sized datasets
- When interpretability is important
- For baseline comparisons
- When local patterns are important
- Non-linear decision boundaries are expected

The projects in this repository provide comprehensive implementations and examples that demonstrate both the theoretical foundations and practical applications of KNN algorithms.

---

*Last Updated: January 2026*
