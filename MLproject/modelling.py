"""
modelling.py — Baseline ML Models with MLflow Tracking
=======================================================
Proyek: SMSML_Ihya_Abdillah
Dicoding: Membangun Sistem Machine Learning

Trains baseline models (Logistic Regression, Random Forest, Gradient Boosting)
and logs parameters, metrics, model artifacts, confusion matrix, and
classification report to MLflow.
"""

import os
import sys
import json
import logging
import warnings

# Reconfigure stdout/stderr to UTF-8 to avoid encoding crashes on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ──────────────────────────── Logging ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ──────────────────────────── Constants ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "processed")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Churn_Baseline_Experiments"

# ──────────────────────────── Data Loading ─────────────────────────────
def load_processed_data() -> tuple:
    """Load preprocessed train/test splits."""
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv")).squeeze()
    logger.info(f"Loaded data — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────── Artifact Helpers ─────────────────────────
def save_confusion_matrix(y_true, y_pred, filepath: str) -> str:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {filepath}")
    return filepath


def save_classification_report(y_true, y_pred, filepath: str) -> str:
    """Generate and save classification report as JSON."""
    report = classification_report(
        y_true, y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
    )
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Classification report saved → {filepath}")
    return filepath


# ──────────────────────────── Training & Logging ───────────────────────
def train_and_log_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """Train a model and log everything using MLflow autolog."""
    is_nested = "MLFLOW_RUN_ID" in os.environ
    with mlflow.start_run(run_name=model_name, nested=is_nested):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*50}")
        
        # Fit the model. Because mlflow.sklearn.autolog() is active, 
        # this will automatically log parameters, training score, and the model artifact.
        model.fit(X_train, y_train)

        # ── Predict ──
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # ── Calculate Metrics ──
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

        # ── Log Test Set Metrics ──
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        # ── Custom Artifact 1: Confusion Matrix PNG ──
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
        os.remove(cm_path)  # Clean up local temp file

        # ── Custom Artifact 2: Classification Report JSON ──
        cr_path = f"classification_report_{model_name.replace(' ', '_').lower()}.json"
        save_classification_report(y_test, y_pred, cr_path)
        mlflow.log_artifact(cr_path, artifact_path="evaluation")
        os.remove(cr_path)  # Clean up local temp file

        logger.info(f"[OK] {model_name} -- Run logged successfully via autolog\n")

    return metrics


# ──────────────────────────── Main Execution ───────────────────────────
def main():
    """Run baseline experiments for all models using local MLflow autologging."""
    # Set MLflow tracking only if not executed within a parent mlflow run
    if "MLFLOW_RUN_ID" not in os.environ:
        import requests
        try:
            requests.get(MLFLOW_TRACKING_URI, timeout=1)
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        except Exception:
            local_db = "sqlite:///mlflow.db"
            mlflow.set_tracking_uri(local_db)
            logger.info(f"MLflow tracking server not running. Using local database: {local_db}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"Experiment set to: {EXPERIMENT_NAME}")
    else:
        logger.info(f"Running inside MLflow run (ID: {os.environ['MLFLOW_RUN_ID']}). Keeping current tracking URI: {mlflow.get_tracking_uri()}")

    # Enable autologging
    mlflow.sklearn.autolog()
    logger.info("MLflow scikit-learn autologging enabled")

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Define baseline models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42, learning_rate=0.1
        ),
    }

    # Train and log each model
    results = {}
    for name, model in models.items():
        metrics = train_and_log_model(
            model, name, X_train, X_test, y_train, y_test
        )
        results[name] = metrics

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 60)
    results_df = pd.DataFrame(results).T
    logger.info(f"\n{results_df.to_string()}")

    best_model = results_df["f1_score"].idxmax()
    logger.info(f"\n[BEST] Best Model (F1): {best_model} -> {results_df.loc[best_model, 'f1_score']:.4f}")


if __name__ == "__main__":
    main()
