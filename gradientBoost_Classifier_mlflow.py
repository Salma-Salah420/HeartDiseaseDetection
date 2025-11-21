# ================================================================
# 🚀 MLflow Version for Your New Gradient Boosting Model
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
)

from joblib import parallel_backend
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# 1️⃣ Load Processed Data
# ================================================================

X_train = pd.read_csv("X_train_final.csv")
y_train = pd.read_csv("y_train_final.csv")

X_test = pd.read_csv("X_test_final.csv")
y_test = pd.read_csv("y_test_final.csv")

print("Train:", X_train.shape, "| Test:", X_test.shape)

# ================================================================
# 2️⃣ Model + Cross Validation
# ================================================================

gb = GradientBoostingClassifier(random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

with parallel_backend('threading'):
    cv_res = cross_validate(gb, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

# ================================================================
# 3️⃣ Start MLflow Run
# ================================================================

mlflow.set_experiment("heart-disease-predict")

with mlflow.start_run() as run:

    mlflow.set_tag("model", "GradientBoostingClassifier-new-version")

    # Log CV Metrics
    for metric in scoring:
        key = "test_" + metric
        mlflow.log_metric("cv_" + metric, float(np.mean(cv_res[key])))

    # ============================================================
    # Train Final Model
    # ============================================================
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    try:
        y_proba = gb.predict_proba(X_test)[:, 1]
    except:
        y_proba = None

    # ============================================================
    # Final Metrics
    # ============================================================

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

    if y_proba is not None:
        rocauc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", rocauc)

    # ============================================================
    # Save Model to MLflow
    # ============================================================
    mlflow.sklearn.log_model(gb, artifact_path="GradientBoostingModel")

    # ============================================================
    # Confusion Matrix
    # ============================================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix - New Gradient Boosting")
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()

    # ============================================================
    # ROC Curve
    # ============================================================
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_val:.4f}")
        plt.plot([0,1], [0,1], linestyle='--')
        plt.legend()
        plt.title("ROC Curve - New Gradient Boosting")
        mlflow.log_figure(plt.gcf(), "roc_curve.png")
        plt.close()

    # ============================================================
    # Feature Importances
    # ============================================================
    feat_imp = pd.Series(gb.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feat_imp.head(20).plot(kind="bar")
    plt.title("Top Features - New Gradient Boosting")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "feature_importances.png")
    plt.close()

    feat_imp.to_csv("feature_importances.csv")
    mlflow.log_artifact("feature_importances.csv")
    os.remove("feature_importances.csv")

    # ============================================================
    # Classification Report
    # ============================================================
    cls = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(cls)

    mlflow.log_artifact("classification_report.txt")
    os.remove("classification_report.txt")

print("\n✅ MLflow Logging Completed Successfully\n")
