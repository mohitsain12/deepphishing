"""
MODULE 6: ML Baseline Models
Trains and evaluates traditional ML models as baselines.
Models: Random Forest, Logistic Regression, Gradient Boosting, Support Vector Machine.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModels:
    """
    Trains and evaluates traditional ML baseline models for comparison
    against the deep learning approach.
    """
    
    def __init__(self, models_dir: str = "models/saved"):
        """
        Initialize baseline model trainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models = {}
        self.results = {}
        logger.info(f"BaselineModels trainer initialized. Target dir: {models_dir}")
    
    def _get_models(self) -> dict:
        """
        Define baseline models with their configurations.
        
        Returns:
            Dictionary of model name -> model instance
        """
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                solver='lbfgs'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                C=1.0
            )
        }
        return models
    
    def _evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'auc_roc': round(float(roc_auc_score(y_test, y_prob)), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test,
                           X_val=None, y_val=None) -> dict:
        """
        Train all baseline models and evaluate them.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dictionary of model results
        """
        logger.info("=" * 60)
        logger.info("MODULE 6: ML Baseline Models Training")
        logger.info("=" * 60)
        
        models = self._get_models()
        
        print("\n" + "=" * 60)
        print("BASELINE MODEL TRAINING & EVALUATION")
        print("=" * 60)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            print(f"\n{'-' * 40}")
            print(f"Model: {name}")
            print(f"{'-' * 40}")
            
            # Train
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Evaluate on test set
            metrics = self._evaluate_model(model, X_test, y_test, name)
            self.results[name] = metrics
            
            # Print metrics
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            
            # Evaluate on validation set too if provided
            if X_val is not None and y_val is not None:
                val_metrics = self._evaluate_model(model, X_val, y_val, name)
                print(f"  Val F1:    {val_metrics['f1_score']:.4f}")
                self.results[name]['val_metrics'] = val_metrics
            
            # Save model
            model_path = os.path.join(self.models_dir, f"baseline_{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"  Model saved to {model_path}")
        
        # Save results summary
        results_path = os.path.join(self.models_dir, "baseline_results.json")
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {
                k: v for k, v in metrics.items()
                if k != 'classification_report'
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self._print_comparison_table()
        return self.results
    
    def _print_comparison_table(self):
        """Print a formatted comparison table of all models."""
        print("\n" + "=" * 70)
        print("BASELINE MODEL COMPARISON TABLE")
        print("=" * 70)
        print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
        print("-" * 70)
        
        best_f1 = 0
        best_model = ""
        
        for name, metrics in self.results.items():
            print(f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {metrics['auc_roc']:>10.4f}")
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model = name
        
        print("-" * 70)
        print(f"Best Baseline Model: {best_model} (F1: {best_f1:.4f})")
    
    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        """
        if model_name not in self.models:
            model_path = os.path.join(self.models_dir, f"baseline_{model_name.lower().replace(' ', '_')}.joblib")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
            else:
                logger.warning(f"Model '{model_name}' not found")
                return None
        
        model = self.models[model_name]
        fn_path = os.path.join(self.models_dir, "feature_names.joblib")
        if hasattr(model, 'feature_importances_') and os.path.exists(fn_path):
            feature_names = joblib.load(fn_path)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None


def main():
    """Main function to train baseline models."""
    try:
        X_train = np.load("data/processed/X_train.npy")
        X_val = np.load("data/processed/X_val.npy")
        X_test = np.load("data/processed/X_test.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_val = np.load("data/processed/y_val.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("Preprocessed data not found. Running data acquisition and preprocessing first...")
        from src.data.download_dataset import PhishingDatasetDownloader
        from src.features.feature_engineering import DataPreprocessor
        
        downloader = PhishingDatasetDownloader()
        df = downloader.prepare_dataset()
        preprocessor = DataPreprocessor()
        splits = preprocessor.preprocess(df)
        X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
        y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    baseline = BaselineModels()
    results = baseline.train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val)
    
    importance = baseline.get_feature_importance('Random Forest')
    if importance is not None:
        print("\nTop 15 Important Features (Random Forest):")
        print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
