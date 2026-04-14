"""
=============================================================
  Fake Banking APK Detection — Complete ML Pipeline
  Dataset: Android Malware Detection (dannyrevaldo, Kaggle)
  Samples: 12,278 | Features: 328 | Binary Classification
=============================================================
"""

# ─────────────────────────────────────────
#  STEP 0 — Install Required Libraries
# ─────────────────────────────────────────
# Run this in terminal before running the script:
# pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import shap

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")


# ════════════════════════════════════════════════
#  STEP 1 — LOAD DATASET
# ════════════════════════════════════════════════

print("=" * 60)
print("  STEP 1: Loading Dataset")
print("=" * 60)

# ▶ UPDATE this path to where your CSV file is saved
df = pd.read_csv("android_malware_dataset.csv")

print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn names (last 5): {df.columns.tolist()[-5:]}")


# ════════════════════════════════════════════════
#  STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 2: Exploratory Data Analysis")
print("=" * 60)

# ── 2a. Identify target column ───────────────────
# The dataset has 327 integer feature columns + 1 label column
# Common label column names: 'Label', 'label', 'class', 'target', 'Category'
# Auto-detect the label column (non-numeric or last column)

label_col = None
for col in df.columns:
    if df[col].dtype == object or col.lower() in ['label', 'class', 'target', 'category']:
        label_col = col
        break

if label_col is None:
    label_col = df.columns[-1]  # fallback: use last column

print(f"\n🎯 Detected label column: '{label_col}'")
print(f"Unique values: {df[label_col].unique()}")
print(f"\nClass Distribution:\n{df[label_col].value_counts()}")
print(f"\nClass Balance: {df[label_col].value_counts(normalize=True).round(3) * 100}%")

# ── 2b. Check for missing values ─────────────────
missing = df.isnull().sum().sum()
print(f"\n🔍 Total missing values: {missing}")
if missing > 0:
    print(df.isnull().sum()[df.isnull().sum() > 0])

# ── 2c. Basic statistics ─────────────────────────
print(f"\n📊 Dataset Statistics:\n{df.describe().T.head(10)}")

# ── 2d. Plot class distribution ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class count bar
class_counts = df[label_col].value_counts()
axes[0].bar(class_counts.index, class_counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
axes[0].set_title("Class Distribution (Imbalanced!)", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Class (0=Benign, 1=Malicious)")
axes[0].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Class pie chart
axes[1].pie(class_counts.values, labels=['Benign', 'Malicious'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
            startangle=90, explode=(0, 0.05))
axes[1].set_title("Class Proportion", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: class_distribution.png")


# ════════════════════════════════════════════════
#  STEP 3 — PREPROCESSING
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 3: Preprocessing")
print("=" * 60)

# ── 3a. Separate features and target ─────────────
X = df.drop(columns=[label_col])
y = df[label_col]

# ── 3b. Encode label if it's string ──────────────
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"✅ Label encoded: {le.classes_} → {le.transform(le.classes_)}")
    y = pd.Series(y)

# Ensure binary: 0 = benign, 1 = malicious
print(f"Label value counts:\n{pd.Series(y).value_counts()}")

# ── 3c. Handle missing values ─────────────────────
# Fill missing with column median (safe for binary/integer feature columns)
X = X.fillna(X.median())
print(f"✅ Missing values filled with median")

# ── 3d. Remove zero-variance features ─────────────
# Features with same value in all rows = useless
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.0)
X_var = vt.fit_transform(X)
selected_features = X.columns[vt.get_support()]
X = pd.DataFrame(X_var, columns=selected_features)
print(f"✅ Features after removing zero-variance: {X.shape[1]} (was {df.shape[1]-1})")

# ── 3e. Train/Test split (stratified to preserve class ratio) ─
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y      # ← CRITICAL: keeps class ratio same in train & test
)
print(f"✅ Train size: {X_train.shape} | Test size: {X_test.shape}")

# ── 3f. Handle class imbalance with SMOTE ─────────
# Dataset is 73% benign / 27% malicious — SMOTE creates synthetic minority samples
print("\n⚖️  Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE — Train size: {X_train_balanced.shape}")
print(f"   Class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# ── 3g. Scale features (only needed for Logistic Regression) ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled  = scaler.transform(X_test)

# Save scaler for later use in Flask app
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved: scaler.pkl")


# ════════════════════════════════════════════════
#  STEP 4 — FEATURE SELECTION (Top 50 Features)
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 4: Feature Selection")
print("=" * 60)

# Use a quick Random Forest to pick top 50 most important features
# This speeds up training and reduces noise from 300+ features

print("🔍 Running feature selection with Random Forest...")
selector_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector_rf.fit(X_train_balanced, y_train_balanced)

importances = pd.Series(selector_rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(50).index.tolist()

print(f"✅ Top 50 features selected out of {X.shape[1]}")
print(f"\nTop 10 most important features:")
for i, feat in enumerate(importances.nlargest(10).index, 1):
    print(f"  {i:2}. {feat:<40} → {importances[feat]:.4f}")

# Plot top 20 features
plt.figure(figsize=(10, 8))
importances.nlargest(20).sort_values().plot(kind='barh', color='#3498db', edgecolor='black')
plt.title("Top 20 Most Important Features (Feature Selection RF)", fontsize=13, fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_selection.png", dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: feature_importance_selection.png")

# Filter datasets to top 50 features only
X_train_sel = X_train_balanced[top_features]
X_test_sel  = X_test[top_features]

# Also scale the selected features for LR
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled  = scaler.transform(X_test_sel)

# Save feature list for Flask app
joblib.dump(top_features, "top_features.pkl")
print("✅ Top features list saved: top_features.pkl")


# ════════════════════════════════════════════════
#  STEP 5 — TRAIN ALL 4 MODELS
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 5: Training All 4 Models")
print("=" * 60)

results = {}


# ── MODEL 1: Logistic Regression (Baseline) ──────
print("\n[1/4] Training Logistic Regression (Baseline)...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    solver='lbfgs',
    random_state=42
)
lr.fit(X_train_sel_scaled, y_train_balanced)

y_pred_lr = lr.predict(X_test_sel_scaled)
y_prob_lr = lr.predict_proba(X_test_sel_scaled)[:, 1]

results['Logistic Regression'] = {
    'model': lr,
    'y_pred': y_pred_lr,
    'y_prob': y_prob_lr,
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'auc': roc_auc_score(y_test, y_prob_lr)
}
print(f"  ✅ Accuracy: {results['Logistic Regression']['accuracy']:.4f} | AUC: {results['Logistic Regression']['auc']:.4f}")


# ── MODEL 2: Random Forest ────────────────────────
print("\n[2/4] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sel, y_train_balanced)

y_pred_rf = rf.predict(X_test_sel)
y_prob_rf = rf.predict_proba(X_test_sel)[:, 1]

results['Random Forest'] = {
    'model': rf,
    'y_pred': y_pred_rf,
    'y_prob': y_prob_rf,
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'auc': roc_auc_score(y_test, y_prob_rf)
}
print(f"  ✅ Accuracy: {results['Random Forest']['accuracy']:.4f} | AUC: {results['Random Forest']['auc']:.4f}")


# ── MODEL 3: XGBoost (Primary Model) ─────────────
print("\n[3/4] Training XGBoost (Primary Model)...")

# Calculate scale_pos_weight for imbalance handling
neg_count = (y_train_balanced == 0).sum()
pos_count = (y_train_balanced == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(
    X_train_sel, y_train_balanced,
    eval_set=[(X_test_sel, y_test)],
    verbose=100
)

y_pred_xgb = xgb.predict(X_test_sel)
y_prob_xgb = xgb.predict_proba(X_test_sel)[:, 1]

results['XGBoost'] = {
    'model': xgb,
    'y_pred': y_pred_xgb,
    'y_prob': y_prob_xgb,
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'auc': roc_auc_score(y_test, y_prob_xgb)
}
print(f"  ✅ Accuracy: {results['XGBoost']['accuracy']:.4f} | AUC: {results['XGBoost']['auc']:.4f}")


# ── MODEL 4: Voting Ensemble (Final Model) ────────
print("\n[4/4] Training Voting Ensemble (Final Model)...")

# Note: All models must use same input
# LR uses scaled, RF and XGB use unscaled
# We wrap LR in a Pipeline so Ensemble can handle it cleanly
from sklearn.pipeline import Pipeline

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42))
])

ensemble = VotingClassifier(
    estimators=[
        ('lr', lr_pipeline),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft',
    weights=[1, 2, 3]   # XGBoost gets highest weight
)
ensemble.fit(X_train_sel, y_train_balanced)

y_pred_ens = ensemble.predict(X_test_sel)
y_prob_ens = ensemble.predict_proba(X_test_sel)[:, 1]

results['Voting Ensemble'] = {
    'model': ensemble,
    'y_pred': y_pred_ens,
    'y_prob': y_prob_ens,
    'accuracy': accuracy_score(y_test, y_pred_ens),
    'f1': f1_score(y_test, y_pred_ens),
    'auc': roc_auc_score(y_test, y_prob_ens)
}
print(f"  ✅ Accuracy: {results['Voting Ensemble']['accuracy']:.4f} | AUC: {results['Voting Ensemble']['auc']:.4f}")


# ════════════════════════════════════════════════
#  STEP 6 — EVALUATE AND COMPARE ALL MODELS
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 6: Evaluation & Comparison")
print("=" * 60)

# ── 6a. Print classification reports ─────────────
for name, res in results.items():
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_test, res['y_pred'],
                                target_names=['Benign', 'Malicious']))

# ── 6b. Model comparison table ───────────────────
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [f"{r['accuracy']*100:.2f}%" for r in results.values()],
    'F1-Score': [f"{r['f1']:.4f}" for r in results.values()],
    'AUC-ROC':  [f"{r['auc']:.4f}" for r in results.values()],
})
print("\n📊 MODEL COMPARISON TABLE:")
print(comparison.to_string(index=False))

# ── 6c. Confusion Matrices (all 4 models) ─────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                ax=axes[idx], linewidths=0.5)
    axes[idx].set_title(f"{name}\nAcc: {res['accuracy']*100:.2f}% | AUC: {res['auc']:.4f}",
                        fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: confusion_matrices.png")

# ── 6d. ROC Curves (all 4 models) ─────────────────
plt.figure(figsize=(10, 7))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.4f})",
             color=color, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("ROC Curves — All Models Comparison", fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: roc_curves.png")

# ── 6e. Bar chart comparison ──────────────────────
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()],
    'AUC-ROC':  [r['auc'] for r in results.values()],
})
metrics_df = metrics_df.set_index('Model')

ax = metrics_df.plot(kind='bar', figsize=(12, 6),
                     color=['#3498db', '#2ecc71', '#e74c3c'],
                     edgecolor='black', width=0.7)
plt.title("Model Performance Comparison", fontsize=13, fontweight='bold')
plt.ylabel("Score")
plt.xticks(rotation=20, ha='right')
plt.ylim(0.7, 1.05)
plt.legend(loc='lower right')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: model_comparison_bar.png")


# ════════════════════════════════════════════════
#  STEP 7 — CROSS VALIDATION (XGBoost only)
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 7: Cross Validation (XGBoost)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X_train_sel, y_train_balanced,
                             cv=cv, scoring='f1', n_jobs=-1)

print(f"\n5-Fold CV F1 Scores: {cv_scores.round(4)}")
print(f"Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ════════════════════════════════════════════════
#  STEP 8 — SHAP EXPLAINABILITY (XGBoost)
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 8: SHAP Explainability")
print("=" * 60)

print("🔍 Computing SHAP values (this may take a minute)...")

explainer = shap.TreeExplainer(xgb)
# Use a sample of 500 test points for speed
X_shap_sample = X_test_sel.sample(500, random_state=42)
shap_values = explainer.shap_values(X_shap_sample)

# ── 8a. Summary Plot (global feature importance) ──
plt.figure()
shap.summary_plot(shap_values, X_shap_sample,
                  max_display=20, show=False)
plt.title("SHAP Summary Plot — Top 20 Features", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: shap_summary.png")

# ── 8b. SHAP Bar Plot (mean absolute impact) ──────
plt.figure()
shap.summary_plot(shap_values, X_shap_sample,
                  plot_type='bar', max_display=20, show=False)
plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("📸 Saved: shap_bar.png")

# ── 8c. Single APK Explanation (Force Plot) ───────
print("\n🔎 Explaining a single APK prediction:")
idx = 0
expected_val = explainer.expected_value
shap_val = explainer.shap_values(X_test_sel.iloc[[idx]])

print(f"  Actual label:     {'Malicious 🔴' if y_test.iloc[idx]==1 else 'Benign 🟢'}")
print(f"  Predicted label:  {'Malicious 🔴' if xgb.predict(X_test_sel.iloc[[idx]])[0]==1 else 'Benign 🟢'}")
print(f"  Confidence:       {xgb.predict_proba(X_test_sel.iloc[[idx]])[0][1]*100:.1f}% malicious")

# Top 5 reasons for this prediction
feature_impact = pd.Series(shap_val[0], index=X_test_sel.columns)
print(f"\n  Top 5 reasons this APK was flagged:")
for feat, val in feature_impact.abs().nlargest(5).items():
    direction = "🔴 pushes MALICIOUS" if feature_impact[feat] > 0 else "🟢 pushes BENIGN"
    print(f"    {feat:<35} {direction}  (impact: {val:.4f})")


# ════════════════════════════════════════════════
#  STEP 9 — SAVE THE BEST MODEL
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 9: Saving Best Model")
print("=" * 60)

# Find best model by AUC-ROC
best_name = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']

joblib.dump(best_model, "best_model.pkl")
joblib.dump(top_features, "top_features.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"✅ Best model: {best_name} (AUC: {results[best_name]['auc']:.4f})")
print(f"✅ Saved: best_model.pkl")
print(f"✅ Saved: top_features.pkl")
print(f"✅ Saved: scaler.pkl")


# ════════════════════════════════════════════════
#  STEP 10 — PREDICTION FUNCTION (for Flask App)
# ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 10: Prediction Function (reusable)")
print("=" * 60)

def predict_apk(feature_dict: dict) -> dict:
    """
    Takes a dictionary of APK features and returns prediction.
    
    Args:
        feature_dict: dict with feature_name → feature_value
                      e.g. {'READ_SMS': 1, 'SYSTEM_ALERT_WINDOW': 0, ...}
    
    Returns:
        dict with verdict, confidence, top_reasons
    """
    model      = joblib.load("best_model.pkl")
    features   = joblib.load("top_features.pkl")

    # Build feature vector in correct order
    row = pd.DataFrame([feature_dict])
    for feat in features:
        if feat not in row.columns:
            row[feat] = 0  # fill missing features with 0

    row = row[features]

    # Predict
    pred       = model.predict(row)[0]
    confidence = model.predict_proba(row)[0][1]

    # Explain with SHAP
    loaded_xgb = joblib.load("best_model.pkl")
    loaded_exp = shap.TreeExplainer(loaded_xgb) if hasattr(loaded_xgb, 'get_booster') else None

    top_reasons = []
    if loaded_exp:
        sv = loaded_exp.shap_values(row)
        impact = pd.Series(sv[0], index=features)
        for feat in impact.abs().nlargest(5).index:
            direction = "suspicious" if impact[feat] > 0 else "safe"
            top_reasons.append({"feature": feat, "direction": direction, "impact": round(float(impact[feat]), 4)})

    return {
        "verdict":    "FAKE / MALICIOUS" if pred == 1 else "LEGITIMATE",
        "confidence": f"{confidence * 100:.1f}%",
        "risk_score": round(float(confidence), 4),
        "top_reasons": top_reasons
    }


# ── Test the prediction function ─────────────────
print("\n🧪 Testing prediction function with a sample APK...")
sample_features = X_test_sel.iloc[0].to_dict()
result = predict_apk(sample_features)
print(f"\n  Verdict:    {result['verdict']}")
print(f"  Confidence: {result['confidence']}")
print(f"  Risk Score: {result['risk_score']}")
print(f"\n  Top Reasons:")
for r in result['top_reasons']:
    print(f"    {r['feature']:<35} → {r['direction']} (impact: {r['impact']})")


print("\n" + "=" * 60)
print("  ✅ PIPELINE COMPLETE")
print("=" * 60)
print("""
Generated Files:
  📊 class_distribution.png          — Class imbalance visualization
  📊 feature_importance_selection.png — Feature selection results  
  📊 confusion_matrices.png          — All 4 models side by side
  📊 roc_curves.png                  — ROC curve comparison
  📊 model_comparison_bar.png        — Accuracy/F1/AUC bar chart
  📊 shap_summary.png                — SHAP feature importance (beeswarm)
  📊 shap_bar.png                    — SHAP mean impact bar chart

  💾 best_model.pkl                  — Trained model (load in Flask app)
  💾 top_features.pkl                — Feature names list
  💾 scaler.pkl                      — StandardScaler

Next Step:
  → Build Flask app that loads best_model.pkl
  → User uploads APK → extract same features → call predict_apk()
  → Show result in browser
""")