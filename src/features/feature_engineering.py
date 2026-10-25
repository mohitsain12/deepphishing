"""
MODULE 5: Data Preprocessing & Feature Engineering
Extracts comprehensive features from URLs for phishing detection.
Handles data cleaning, transformation, and feature generation.
"""

import os
import sys
import re
import math
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class URLFeatureExtractor:
    """
    Extracts a comprehensive set of features from URLs for phishing detection.
    
    Feature Categories:
    1. Length-based features
    2. Count-based features (special characters)
    3. Binary features (presence/absence)
    4. Domain-based features
    5. Statistical features (entropy, ratios)
    6. Structural features
    """
    
    # Known URL shortening services
    SHORTENING_SERVICES = {
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
        'is.gd', 'buff.ly', 'adf.ly', 'tiny.cc', 'lnkd.in',
        'db.tt', 'qr.ae', 'cur.lv', 'ity.im', 'q.gs',
        'po.st', 'bc.vc', 'twit.ac', 'su.pr', 'tr.im',
        'cli.gs', 'v.gd', 'mcaf.ee', 'short.to', 'rb.gy'
    }
    
    # Suspicious TLDs commonly used in phishing
    SUSPICIOUS_TLDS = {
        '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top',
        '.work', '.click', '.link', '.info', '.online',
        '.site', '.club', '.buzz', '.country', '.stream',
        '.download', '.racing', '.review', '.win', '.bid',
        '.accountant', '.science', '.party', '.faith', '.pw'
    }
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
        logger.info("URLFeatureExtractor initialized")
    
    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """
        Calculate Shannon entropy of a string.
        Higher entropy suggests more randomness (common in phishing URLs).
        
        Args:
            text: Input string
            
        Returns:
            Shannon entropy value
        """
        if not text:
            return 0.0
        
        counter = Counter(text)
        length = len(text)
        entropy = -sum(
            (count / length) * math.log2(count / length)
            for count in counter.values()
        )
        return round(entropy, 4)
    
    @staticmethod
    def _parse_url_safely(url: str) -> dict:
        """
        Safely parse a URL into its components.
        
        Args:
            url: URL string to parse
            
        Returns:
            Dictionary with URL components
        """
        url = str(url).strip()
        
        # Add scheme if missing
        if not url.startswith(('http://', 'https://', 'ftp://')):
            url = 'http://' + url
        
        try:
            parsed = urlparse(url)
            return {
                'scheme': parsed.scheme or '',
                'netloc': parsed.netloc or '',
                'path': parsed.path or '',
                'query': parsed.query or '',
                'fragment': parsed.fragment or '',
                'hostname': parsed.hostname or '',
                'port': parsed.port,
                'full_url': url
            }
        except Exception:
            return {
                'scheme': '', 'netloc': '', 'path': '',
                'query': '', 'fragment': '', 'hostname': '',
                'port': None, 'full_url': url
            }
    
    def extract_features(self, url: str) -> dict:
        """
        Extract all features from a single URL.
        
        Args:
            url: URL string to extract features from
            
        Returns:
            Dictionary of feature name -> value pairs
        """
        url = str(url).strip()
        parsed = self._parse_url_safely(url)
        
        features = {}
        
        # ==========================================
        # 1. LENGTH-BASED FEATURES
        # ==========================================
        features['url_length'] = len(url)
        features['domain_length'] = len(parsed['hostname'])
        features['path_length'] = len(parsed['path'])
        features['query_length'] = len(parsed['query'])
        features['fragment_length'] = len(parsed['fragment'])
        
        # ==========================================
        # 2. COUNT-BASED FEATURES
        # ==========================================
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_exclamation'] = url.count('!')
        features['num_space'] = url.count(' ') + url.count('%20')
        features['num_tilde'] = url.count('~')
        features['num_comma'] = url.count(',')
        features['num_plus'] = url.count('+')
        features['num_star'] = url.count('*')
        features['num_hash'] = url.count('#')
        features['num_dollar'] = url.count('$')
        features['num_percent'] = url.count('%')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        features['num_params'] = len(parse_qs(parsed['query']))
        
        # ==========================================
        # 3. BINARY FEATURES
        # ==========================================
        features['has_ip'] = int(bool(re.search(
            r'(\d{1,3}\.){3}\d{1,3}', url
        )))
        features['has_https'] = int(url.startswith('https'))
        features['has_http'] = int(url.startswith('http://'))
        features['has_www'] = int('www.' in url.lower())
        features['has_port'] = int(parsed['port'] is not None)
        features['has_at_symbol'] = int('@' in url)
        features['has_double_slash'] = int('//' in url[8:])  # After protocol
        features['has_shortening_service'] = int(
            any(service in url.lower() for service in self.SHORTENING_SERVICES)
        )
        features['is_encoded'] = int('%' in url)
        features['has_suspicious_tld'] = int(
            any(url.lower().endswith(tld) or f"{tld}/" in url.lower()
                for tld in self.SUSPICIOUS_TLDS)
        )
        features['has_login_keyword'] = int(
            any(kw in url.lower() for kw in ['login', 'signin', 'verify', 'secure', 'account', 'confirm', 'update', 'banking'])
        )
        features['has_redirect'] = int(
            any(r in url.lower() for r in ['redirect', 'redir', 'url=', 'link=', 'goto=', 'return='])
        )
        
        # ==========================================
        # 4. DOMAIN-BASED FEATURES
        # ==========================================
        hostname = parsed['hostname']
        domain_parts = hostname.split('.') if hostname else []
        
        features['num_subdomains'] = max(0, len(domain_parts) - 2)
        features['domain_token_count'] = len(domain_parts)
        features['longest_domain_token'] = max((len(p) for p in domain_parts), default=0)
        features['avg_domain_token_length'] = (
            np.mean([len(p) for p in domain_parts]) if domain_parts else 0
        )
        
        # ==========================================
        # 5. STATISTICAL FEATURES
        # ==========================================
        total_chars = len(url) or 1
        features['letter_ratio'] = round(features['num_letters'] / total_chars, 4)
        features['digit_ratio'] = round(features['num_digits'] / total_chars, 4)
        features['special_char_ratio'] = round(
            1 - features['letter_ratio'] - features['digit_ratio'], 4
        )
        features['domain_entropy'] = self._calculate_entropy(hostname)
        features['path_entropy'] = self._calculate_entropy(parsed['path'])
        features['url_entropy'] = self._calculate_entropy(url)
        features['query_entropy'] = self._calculate_entropy(parsed['query'])
        
        # ==========================================
        # 6. STRUCTURAL FEATURES
        # ==========================================
        features['url_depth'] = parsed['path'].count('/')
        features['has_subdirectory'] = int(features['url_depth'] > 1)
        features['has_query_string'] = int(len(parsed['query']) > 0)
        features['tld_length'] = len(domain_parts[-1]) if domain_parts else 0
        
        # Vowel-consonant ratio in domain (phishing domains often have unusual ratios)
        vowels = sum(1 for c in hostname.lower() if c in 'aeiou')
        consonants = sum(1 for c in hostname.lower() if c.isalpha() and c not in 'aeiou')
        features['vowel_consonant_ratio'] = round(
            vowels / max(consonants, 1), 4
        )
        
        # Consecutive character patterns
        features['max_consecutive_consonants'] = max(
            (len(m.group()) for m in re.finditer(r'[bcdfghjklmnpqrstvwxyz]+', hostname.lower())),
            default=0
        )
        
        return features
    
    def extract_features_batch(self, urls: pd.Series) -> pd.DataFrame:
        """
        Extract features from a batch of URLs.
        
        Args:
            urls: Series of URL strings
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Extracting features from {len(urls)} URLs...")
        
        features_list = []
        for i, url in enumerate(urls):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i+1}/{len(urls)} URLs...")
            features_list.append(self.extract_features(url))
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = list(features_df.columns)
        
        logger.info(f"Extracted {len(self.feature_names)} features")
        return features_df


class DataPreprocessor:
    """
    Handles data preprocessing including cleaning, scaling, and splitting.
    
    Pipeline:
    1. Feature extraction
    2. Handle missing values
    3. Remove duplicates
    4. Scale features
    5. Split into train/val/test sets
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        }
        self.feature_extractor = URLFeatureExtractor()
        self.scaler = None
        self.feature_names = None
        logger.info("DataPreprocessor initialized")
    
    def preprocess(self, df: pd.DataFrame) -> dict:
        """
        Run the full preprocessing pipeline.
        
        Args:
            df: Raw DataFrame with 'url' and 'status' columns
            
        Returns:
            Dictionary with processed data splits and metadata
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        logger.info("=" * 60)
        logger.info("MODULE 5: Data Preprocessing")
        logger.info("=" * 60)
        
        # Step 1: Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['url']).reset_index(drop=True)
        logger.info(f"Removed {initial_len - len(df)} duplicate URLs")
        
        # Step 2: Extract features
        if 'url' in df.columns:
            features_df = self.feature_extractor.extract_features_batch(df['url'])
            self.feature_names = list(features_df.columns)
        else:
            # If URL column doesn't exist, assume features are already extracted
            exclude_cols = ['status', 'url', 'label', 'target']
            self.feature_names = [c for c in df.columns if c not in exclude_cols]
            features_df = df[self.feature_names]
        
        # Step 3: Handle target variable
        if 'status' in df.columns:
            y = df['status'].values
        elif 'label' in df.columns:
            y = df['label'].values
        else:
            raise ValueError("No target column found ('status' or 'label')")
        
        # Ensure binary encoding (0 = legitimate, 1 = phishing)
        unique_vals = np.unique(y)
        if set(unique_vals) == {'legitimate', 'phishing'} or set(unique_vals) == {'good', 'bad'}:
            y = (y == unique_vals[1]).astype(int)
        y = y.astype(int)
        
        X = features_df.values
        
        # Step 4: Handle missing/infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Step 5: Split data (train / val / test)
        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('val_size', 0.1)
        random_state = self.config.get('random_state', 42)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Step 6: Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 7: Save processed data and artifacts
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models/saved", exist_ok=True)
        
        # Save as numpy arrays
        np.save("data/processed/X_train.npy", X_train_scaled)
        np.save("data/processed/X_val.npy", X_val_scaled)
        np.save("data/processed/X_test.npy", X_test_scaled)
        np.save("data/processed/y_train.npy", y_train)
        np.save("data/processed/y_val.npy", y_val)
        np.save("data/processed/y_test.npy", y_test)
        
        # Save scaler and feature names
        joblib.dump(self.scaler, "models/saved/scaler.joblib")
        joblib.dump(self.feature_names, "models/saved/feature_names.joblib")
        
        # Save feature-engineered DataFrame
        features_df['status'] = y
        features_df.to_csv("data/processed/features_dataset.csv", index=False)
        
        logger.info("Processed data saved to data/processed/")
        logger.info("Scaler saved to models/saved/scaler.joblib")
        
        result = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'n_features': X_train_scaled.shape[1]
        }
        
        print("\n" + "=" * 60)
        print("Data Preprocessing Complete!")
        print("=" * 60)
        print(f"Number of features: {result['n_features']}")
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Test samples: {len(y_test)}")
        print(f"Feature names: {self.feature_names[:10]}...")
        
        return result


def main():
    """Main function to run feature engineering pipeline."""
    # Load raw data
    data_path = "data/raw/phishing_dataset.csv"
    
    if not os.path.exists(data_path):
        print("Raw dataset not found. Please run data acquisition first.")
        print("  python -m src.data.download_dataset")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Run preprocessing
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess(df)
    
    print(f"\nFeature extraction and preprocessing complete!")
    print(f"Files saved in data/processed/ and models/saved/")


if __name__ == "__main__":
    main()
