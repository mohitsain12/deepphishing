"""
MODULE 11: URL Prediction Pipeline
End-to-end inference engine for classifying arbitrary input URLs as Phishing or Legitimate.
Supports single URL, batch URL scoring, risk level categorization, and explainable indicator heuristics.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.feature_engineering import URLFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhishingPredictor:
    """
    End-to-end inference engine for phishing URL classification.
    """
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.feature_extractor = URLFeatureExtractor()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = None
        
        self._load_model(model_path)
        self._load_scaler(scaler_path)
        logger.info("PhishingPredictor pipeline initialized")
        
    def _load_model(self, model_path: str = None):
        models_dir = "models/saved"
        candidates = [
            ('best_optimized_dnn.h5', 'dnn'),
            ('best_dnn_model.h5', 'dnn'),
            ('phishing_dnn_final.h5', 'dnn'),
            ('baseline_random_forest.joblib', 'sklearn'),
            ('baseline_gradient_boosting.joblib', 'sklearn')
        ]
        if model_path:
            m_type = 'dnn' if model_path.endswith('.h5') else 'sklearn'
            candidates.insert(0, (model_path, m_type))
            
        for path_name, m_type in candidates:
            p = path_name if os.path.isabs(path_name) else os.path.join(models_dir, path_name)
            if os.path.exists(p):
                if m_type == 'dnn':
                    try:
                        import tensorflow as tf
                        self.model = tf.keras.models.load_model(p)
                        self.model_type = 'dnn'
                        logger.info(f"Loaded DNN model from {p}")
                        return
                    except Exception as e:
                        logger.warning(f"Error loading {p}: {e}")
                else:
                    try:
                        self.model = joblib.load(p)
                        self.model_type = 'sklearn'
                        logger.info(f"Loaded Scikit-Learn model from {p}")
                        return
                    except Exception as e:
                        logger.warning(f"Error loading {p}: {e}")
        logger.warning("No saved model found! Using heuristic fallback classifier until model is trained.")
        self.model_type = 'heuristic'

    def _load_scaler(self, scaler_path: str = None):
        p = scaler_path or "models/saved/scaler.joblib"
        if os.path.exists(p):
            self.scaler = joblib.load(p)
        fn_p = "models/saved/feature_names.joblib"
        if os.path.exists(fn_p):
            self.feature_names = joblib.load(fn_p)

    def predict_url(self, url: str) -> dict:
        """
        Takes raw URL string, parses features, runs normalization, model forward pass,
        and returns enriched risk metrics and threat indicators.
        """
        url = str(url).strip()
        features = self.feature_extractor.extract_features(url)
        
        if self.model_type == 'heuristic' or self.model is None:
            # High precision rule-based heuristic fallback if weights not yet saved
            phish_score = 0.0
            if features.get('has_ip'): phish_score += 0.35
            if features.get('has_at_symbol'): phish_score += 0.30
            if features.get('has_suspicious_tld'): phish_score += 0.25
            if features.get('has_shortening_service'): phish_score += 0.20
            if features.get('url_length', 0) > 80: phish_score += 0.15
            if not features.get('has_https'): phish_score += 0.15
            if features.get('has_login_keyword'): phish_score += 0.20
            if features.get('num_subdomains', 0) >= 3: phish_score += 0.15
            probability = min(1.0, phish_score)
        else:
            if self.feature_names:
                vec = np.array([features.get(name, 0.0) for name in self.feature_names])
            else:
                vec = np.array(list(features.values()))
            vec = vec.reshape(1, -1)
            if self.scaler is not None:
                vec = self.scaler.transform(vec)
            vec = np.nan_to_num(vec, nan=0.0, posinf=1e5, neginf=-1e5)
            
            if self.model_type == 'dnn':
                probability = float(self.model.predict(vec, verbose=0)[0][0])
            else:
                probability = float(self.model.predict_proba(vec)[0][1])
                
        is_phishing = int(probability >= 0.5)
        confidence = probability if is_phishing else (1.0 - probability)
        
        return {
            'url': url,
            'prediction': is_phishing,
            'label': 'PHISHING' if is_phishing == 1 else 'LEGITIMATE',
            'probability': round(probability, 4),
            'confidence': round(confidence, 4),
            'risk_level': self._get_risk_level(probability),
            'model_type': self.model_type,
            'key_indicators': self._get_key_indicators(features, is_phishing),
            'features': features
        }

    def predict_batch(self, urls: list) -> list:
        return [self.predict_url(u) for u in urls]

    @staticmethod
    def _get_risk_level(prob: float) -> str:
        if prob >= 0.85: return "CRITICAL"
        if prob >= 0.65: return "HIGH"
        if prob >= 0.45: return "MODERATE"
        if prob >= 0.25: return "LOW"
        return "SAFE"

    @staticmethod
    def _get_key_indicators(features: dict, is_phish: int) -> list:
        indicators = []
        if is_phish:
            if features.get('has_ip'): indicators.append("Direct IP address used instead of hostname")
            if features.get('has_at_symbol'): indicators.append("Embedded '@' credential obfuscation character")
            if features.get('has_suspicious_tld'): indicators.append("Uncommon or abuse-prone Top-Level Domain (TLD)")
            if features.get('has_shortening_service'): indicators.append("Obfuscated via URL shortening service")
            if features.get('url_length', 0) > 75: indicators.append(f"Excessive URL character length ({features['url_length']} chars)")
            if features.get('num_subdomains', 0) >= 3: indicators.append(f"Multiple subdomain levels ({features['num_subdomains']})")
            if not features.get('has_https'): indicators.append("Insecure HTTP transport (Missing SSL/TLS)")
            if features.get('domain_entropy', 0) > 3.8: indicators.append("Abnormally high lexical entropy in domain name")
        else:
            if features.get('has_https'): indicators.append("Valid HTTPS protocol")
            if features.get('url_length', 0) <= 60: indicators.append("Concise standard URL structure")
            if features.get('num_subdomains', 0) <= 1: indicators.append("Clean domain hierarchy")
        return indicators if indicators else ["Standard URL structural profile"]


def main():
    predictor = PhishingPredictor()
    test_urls = [
        "https://www.google.com/search?q=cybersecurity",
        "http://192.168.1.1/paypal/login/verify.html",
        "https://github.com/torvalds/linux",
        "http://secure-login.paypal.com.account-update.tk/verify?auth=token"
    ]
    for u in test_urls:
        res = predictor.predict_url(u)
        print(f"[{res['label']}] ({res['probability']*100:.1f}% prob | Risk: {res['risk_level']}) -> {u}")


if __name__ == "__main__":
    main()
