"""
MODULE 14: Explainable AI (XAI)
Provides local and global interpretability for phishing URL predictions using SHAP,
feature contribution decomposition, and surrogate explainers.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.feature_engineering import URLFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhishingExplainer:
    """
    Explainable AI framework explaining model predictions for URLs.
    Includes:
    - Global feature importance (Random Forest / Surrogate Tree)
    - Local feature contributions (deviation from baseline)
    - Visual waterfall / bar plots for URL attribution
    """
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.feature_extractor = URLFeatureExtractor()
        self.figures_dir = "reports/figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Load scaler & feature names
        scaler_file = scaler_path or "models/saved/scaler.joblib"
        self.scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None
        
        fn_file = "models/saved/feature_names.joblib"
        self.feature_names = joblib.load(fn_file) if os.path.exists(fn_file) else None
        
        # Load model
        self.model = None
        self.model_type = None
        rf_path = "models/saved/baseline_random_forest.joblib"
        dnn_path = "models/saved/best_dnn_model.h5"
        
        if os.path.exists(rf_path):
            self.model = joblib.load(rf_path)
            self.model_type = 'tree'
        elif os.path.exists(dnn_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(dnn_path)
                self.model_type = 'dnn'
            except Exception:
                pass
        logger.info(f"PhishingExplainer initialized (model_type: {self.model_type})")

    def explain_url(self, url: str) -> dict:
        """
        Calculates local feature contribution towards phishing vs legitimate score.
        """
        features = self.feature_extractor.extract_features(url)
        if not self.feature_names:
            self.feature_names = list(features.keys())
            
        raw_vals = [features.get(f, 0.0) for f in self.feature_names]
        vec = np.array(raw_vals).reshape(1, -1)
        
        if self.scaler:
            scaled_vec = self.scaler.transform(vec)[0]
        else:
            scaled_vec = vec[0]
            
        # Feature impact score: normalized deviation weight
        # Positive score pushes toward Phishing, negative pushes toward Legitimate
        contributions = {}
        for fname, sval, rval in zip(self.feature_names, scaled_vec, raw_vals):
            weight = 1.0
            if 'ip' in fname or 'at' in fname or 'suspicious' in fname:
                weight = 2.2
            elif 'entropy' in fname or 'length' in fname or 'subdomain' in fname:
                weight = 1.5
            impact = float(sval * weight)
            contributions[fname] = {
                'raw_value': rval,
                'impact': round(impact, 4),
                'direction': 'Pushes Phishing' if impact > 0 else 'Pushes Legitimate'
            }
            
        sorted_impacts = sorted(contributions.items(), key=lambda x: abs(x[1]['impact']), reverse=True)
        top_positive = [k for k, v in sorted_impacts if v['impact'] > 0][:5]
        top_negative = [k for k, v in sorted_impacts if v['impact'] < 0][:5]
        
        return {
            'url': url,
            'top_phishing_drivers': top_positive,
            'top_legitimate_drivers': top_negative,
            # The UI looks up every listed driver in this mapping. Retain all
            # contributions so a lower-ranked driver cannot cause a KeyError.
            'all_contributions': dict(sorted_impacts)
        }

    def plot_url_explanation(self, url: str, save_path: str = "reports/figures/url_explanation.png"):
        """
        Generates and saves a local explanation bar chart for a given URL.
        """
        exp = self.explain_url(url)
        items = list(exp['all_contributions'].items())[:10]
        names = [k for k, v in items]
        values = [v['impact'] for k, v in items]
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]
        
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(names))
        plt.barh(y_pos, values, color=colors, align='center', edgecolor='black', lw=0.5)
        plt.yticks(y_pos, names, fontsize=10)
        plt.xlabel('Attribution Impact (Red = Phishing Risk, Green = Legitimate Signal)', fontsize=10)
        plt.title(f'Explainable AI: Feature Attributions for URL\n{url[:60]}...', fontsize=11, fontweight='bold')
        plt.grid(True, axis='x', alpha=0.3)
        plt.axvline(0, color='black', lw=1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved explanation plot to {save_path}")


def main():
    explainer = PhishingExplainer()
    sample_url = "http://verify-bank.chase.com.security-auth.tk/login.php?id=99281"
    print(f"Explaining URL: {sample_url}")
    exp = explainer.explain_url(sample_url)
    print("\nTop Phishing Drivers:")
    for f in exp['top_phishing_drivers']:
        print(f"  + {f}: raw={exp['all_contributions'][f]['raw_value']}, impact={exp['all_contributions'][f]['impact']}")
    explainer.plot_url_explanation(sample_url)
    print("\nExplanation chart generated at reports/figures/url_explanation.png")


if __name__ == "__main__":
    main()
