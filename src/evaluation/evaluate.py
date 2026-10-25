"""
MODULE 9: Model Evaluation
Evaluates all baseline models and the Deep Neural Network model.
Generates Confusion Matrix, ROC-AUC, Precision-Recall curves, and benchmark tables.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              average_precision_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation suite comparing Baseline ML vs Deep Learning models.
    """
    def __init__(self, figures_dir: str = "reports/figures", models_dir: str = "models/saved"):
        self.figures_dir = figures_dir
        self.models_dir = models_dir
        os.makedirs(figures_dir, exist_ok=True)
        self.all_results = {}
        logger.info(f"ModelEvaluator initialized. Reports dir: {figures_dir}")
        
    def evaluate_model(self, y_true, y_pred, y_prob, model_name: str) -> dict:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0
        
        metrics = {
            'model_name': model_name,
            'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
            'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            'auc_roc': round(float(roc_auc_score(y_true, y_prob)), 4),
            'avg_precision': round(float(average_precision_score(y_true, y_prob)), 4),
            'specificity': specificity,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
        self.all_results[model_name] = metrics
        return metrics

    def evaluate_all_models(self, X_test, y_test) -> dict:
        logger.info("=" * 60)
        logger.info("MODULE 9: Model Evaluation")
        logger.info("=" * 60)
        
        # 1. Evaluate baseline models
        baseline_models = {
            'Random Forest': 'baseline_random_forest.joblib',
            'Gradient Boosting': 'baseline_gradient_boosting.joblib',
            'Logistic Regression': 'baseline_logistic_regression.joblib',
            'SVM': 'baseline_svm.joblib'
        }
        
        for name, filename in baseline_models.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                model = joblib.load(path)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                self.evaluate_model(y_test, y_pred, y_prob, name)
                
        # 2. Evaluate DNN model
        dnn_candidates = ['best_optimized_dnn.h5', 'best_dnn_model.h5', 'phishing_dnn_final.h5']
        for dnn_file in dnn_candidates:
            dnn_path = os.path.join(self.models_dir, dnn_file)
            if os.path.exists(dnn_path):
                import tensorflow as tf
                dnn_model = tf.keras.models.load_model(dnn_path)
                y_prob_dnn = dnn_model.predict(X_test, verbose=0).flatten()
                y_pred_dnn = (y_prob_dnn > 0.5).astype(int)
                self.evaluate_model(y_test, y_pred_dnn, y_prob_dnn, 'Deep Neural Network')
                break
                
        self._print_results_table()
        self.plot_confusion_matrices()
        self.plot_roc_curves(X_test, y_test)
        self.plot_metrics_comparison()
        self._save_results()
        
        return self.all_results

    def _print_results_table(self):
        print("\n" + "=" * 88)
        print("COMPREHENSIVE MODEL EVALUATION BENCHMARK")
        print("=" * 88)
        print(f"{'Model':<24} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8} {'Specificity':>12}")
        print("-" * 88)
        
        best_f1 = -1
        best_model = ""
        for name, m in self.all_results.items():
            print(f"{name:<24} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1_score']:>8.4f} {m['auc_roc']:>8.4f} {m['specificity']:>12.4f}")
            if m['f1_score'] > best_f1:
                best_f1 = m['f1_score']
                best_model = name
        print("-" * 88)
        print(f"Top Performing Model: {best_model} (F1 Score: {best_f1:.4f})")

    def plot_confusion_matrices(self):
        n = len(self.all_results)
        if n == 0:
            return
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, (name, metrics) in enumerate(self.all_results.items()):
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Legitimate', 'Phishing'],
                        yticklabels=['Legitimate', 'Phishing'],
                        ax=axes[i], cbar=False)
            axes[i].set_title(f'{name}\nAcc: {metrics["accuracy"]*100:.1f}% | F1: {metrics["f1_score"]:.3f}', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Predicted Class')
            axes[i].set_ylabel('Ground Truth')
            
        for j in range(len(self.all_results), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved reports/figures/confusion_matrices.png")

    def plot_roc_curves(self, X_test, y_test):
        plt.figure(figsize=(9, 7))
        palette = sns.color_palette("tab10", len(self.all_results))
        
        for (name, metrics), color in zip(self.all_results.items(), palette):
            model = self._load_model_instance(name)
            if model is not None:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.predict(X_test, verbose=0).flatten()
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['auc_roc']:.4f})", lw=2, color=color)
                
        plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Guessing (AUC = 0.5000)')
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
        plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved reports/figures/roc_curves.png")

    def plot_metrics_comparison(self):
        if not self.all_results:
            return
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        models = list(self.all_results.keys())
        
        df_plot = []
        for m in models:
            for met in metrics:
                df_plot.append({'Model': m, 'Metric': met.replace('_', ' ').upper(), 'Score': self.all_results[m][met]})
        df_plot = pd.DataFrame(df_plot)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_plot, x='Model', y='Score', hue='Metric', palette='viridis')
        plt.title('Comparative Performance Metrics Across Models', fontsize=13, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=15)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved reports/figures/metrics_comparison.png")

    def _load_model_instance(self, name: str):
        mapping = {
            'Random Forest': 'baseline_random_forest.joblib',
            'Gradient Boosting': 'baseline_gradient_boosting.joblib',
            'Logistic Regression': 'baseline_logistic_regression.joblib',
            'SVM': 'baseline_svm.joblib'
        }
        if name in mapping:
            p = os.path.join(self.models_dir, mapping[name])
            return joblib.load(p) if os.path.exists(p) else None
        elif name == 'Deep Neural Network':
            for d in ['best_optimized_dnn.h5', 'best_dnn_model.h5', 'phishing_dnn_final.h5']:
                p = os.path.join(self.models_dir, d)
                if os.path.exists(p):
                    import tensorflow as tf
                    return tf.keras.models.load_model(p)
        return None

    def _save_results(self):
        out_path = os.path.join(self.models_dir, "evaluation_results.json")
        save_copy = {k: {sk: sv for sk, sv in v.items() if sk != 'classification_report'}
                     for k, v in self.all_results.items()}
        with open(out_path, 'w') as f:
            json.dump(save_copy, f, indent=2)
        logger.info(f"Saved evaluation results to {out_path}")


def main():
    try:
        X_test = np.load("data/processed/X_test.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("Data files not found. Run preprocessing first.")
        sys.exit(1)
        
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(X_test, y_test)


if __name__ == "__main__":
    main()
