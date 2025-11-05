import os
import argparse
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


def train_and_log_gb(data_path: str,
                     target_col: str,
                     n_estimators: int,
                     learning_rate: float,
                     max_depth: int,
                     subsample: float,
                     min_samples_leaf: int):
    """
    Train GradientBoostingClassifier on ML.csv directly and log everything in MLflow.
    """

    # ✅ Set experiment name
    mlflow.set_experiment("heart-disease-predict")

    # ---------------- Load Dataset ---------------- #
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )

    # ---------------- Train Model ---------------- #
    with mlflow.start_run() as run:
        mlflow.set_tag("clf", "GradientBoostingClassifier")

        # Model definition
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # Train model
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred, digits=4)

        # Log params and metrics
        mlflow.log_params({
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "min_samples_leaf": min_samples_leaf
        })

        mlflow.log_metrics({
            "accuracy": float(acc),
            "f1_score": float(f1)
        })

        # Log model
        mlflow.sklearn.log_model(clf, artifact_path=f"GradientBoostingModel")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap="Blues")
        plt.title("Confusion Matrix - Gradient Boosting (Heart Disease)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        conf_fig = plt.gcf()
        mlflow.log_figure(conf_fig, artifact_file="confusion_matrix.png")
        plt.close()

        # ROC curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Gradient Boosting (Heart Disease)')
            plt.legend(loc="lower right")
            roc_fig = plt.gcf()
            mlflow.log_figure(roc_fig, artifact_file="roc_curve.png")
            plt.close()

            mlflow.log_metric("roc_auc", float(roc_auc_val))

        # Feature importances
        try:
            feat_imps = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            feat_imps.head(20).plot(kind="bar")
            plt.title("Top 20 Feature Importances - Heart Disease Prediction")
            plt.tight_layout()
            fi_fig = plt.gcf()
            mlflow.log_figure(fi_fig, artifact_file="feature_importances.png")
            plt.close()

            fi_path = "feature_importances.csv"
            feat_imps.to_csv(fi_path, header=True)
            mlflow.log_artifact(fi_path)
            os.remove(fi_path)
        except Exception:
            pass

        # Classification report text file
        report_path = "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Accuracy: {:.6f}\n".format(acc))
            f.write("F1-score: {:.6f}\n\n".format(f1))
            f.write("Classification Report:\n")
            f.write(cls_report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        print(f"\n✅ Accuracy: {acc:.4f}")
        print(f"✅ F1 Score: {f1:.4f}")
        print("\nClassification Report:\n", cls_report)


def main(data_path, target_col, n_estimators, learning_rate, max_depth, subsample, min_samples_leaf):
    train_and_log_gb(
        data_path=data_path,
        target_col=target_col,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-f", type=str, default="ML.csv", help="Path to dataset file")
    parser.add_argument("--target_col", "-t", type=str, default="HeartDisease", help="Target column name")
    parser.add_argument("--n_estimators", "-n", type=int, default=100)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.1)
    parser.add_argument("--max_depth", "-d", type=int, default=3)
    parser.add_argument("--subsample", "-s", type=float, default=1.0)
    parser.add_argument("--min_samples_leaf", "-l", type=int, default=1)

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        target_col=args.target_col,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_samples_leaf=args.min_samples_leaf
    )
