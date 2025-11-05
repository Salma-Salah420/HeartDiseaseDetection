# ================================================================
# 🌟 Gradient Boosting Classifier - MLflow Integration (Final)
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
)

import warnings
warnings.filterwarnings("ignore")

# ================================================================
# 1️⃣ Load Dataset (Directly from ML.csv)
# ================================================================

DATA_PATH = "ML.csv"
TARGET_COL = "HeartDisease"  # change this if your target column has another name

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# Split features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ================================================================
# 2️⃣ Setup and Train Gradient Boosting Model
# ================================================================

mlflow.set_experiment("heart-disease-predict")

with mlflow.start_run() as run:
    mlflow.set_tag("model", "GradientBoostingClassifier")

    # Default model
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        min_samples_leaf=1,
        random_state=42
    )

    # Train model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # ================================================================
    # 3️⃣ Evaluation Metrics
    # ================================================================
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=4)

    mlflow.log_params({
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 1.0,
        "min_samples_leaf": 1
    })

    mlflow.log_metrics({
        "accuracy": acc,
        "f1_score": f1
    })

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", cls_report)

    # ================================================================
    # 4️⃣ Log Model to MLflow
    # ================================================================
    mlflow.sklearn.log_model(clf, artifact_path="GradientBoostingModel")

    # ================================================================
    # 5️⃣ Confusion Matrix
    # ================================================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix - Gradient Boosting (Heart Disease)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.show()
    plt.close()

    # ================================================================
    # 6️⃣ ROC Curve
    # ================================================================
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Gradient Boosting (Heart Disease)")
        plt.legend(loc="lower right")
        mlflow.log_figure(plt.gcf(), "roc_curve.png")
        plt.show()
        plt.close()

        mlflow.log_metric("roc_auc", roc_auc)

    # ================================================================
    # 7️⃣ Feature Importances
    # ================================================================
    try:
        feat_imps = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feat_imps.head(20).plot(kind="bar")
        plt.title("Top 20 Feature Importances - Gradient Boosting (Heart Disease)")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "feature_importances.png")
        plt.show()
        plt.close()

        feat_imps.to_csv("feature_importances.csv", header=True)
        mlflow.log_artifact("feature_importances.csv")
        os.remove("feature_importances.csv")
    except Exception:
        pass

    # ================================================================
    # 8️⃣ Classification Report (as text file)
    # ================================================================
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Accuracy: {:.6f}\n".format(acc))
        f.write("F1-score: {:.6f}\n\n".format(f1))
        f.write("Classification Report:\n")
        f.write(cls_report)

    mlflow.log_artifact("classification_report.txt")
    os.remove("classification_report.txt")
