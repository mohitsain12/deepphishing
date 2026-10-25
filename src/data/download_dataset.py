"""
MODULE 3: Dataset Acquisition
Downloads and prepares the phishing URL dataset for the project.
Supports multiple data sources and creates a standardized dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhishingDatasetDownloader:
    """
    Downloads and prepares phishing URL datasets from multiple sources.
    Creates a unified dataset for training and evaluation.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize the dataset downloader.
        
        Args:
            output_dir: Directory to save raw datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Dataset downloader initialized. Output: {output_dir}")
    
    def generate_synthetic_dataset(self, n_samples: int = 11000) -> pd.DataFrame:
        """
        Generate a synthetic phishing URL dataset for demonstration.
        Creates realistic URL patterns for both legitimate and phishing URLs.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame with 'url' and 'status' columns
        """
        logger.info(f"Generating synthetic dataset with {n_samples} samples...")
        np.random.seed(42)
        
        n_legit = n_samples // 2
        n_phish = n_samples - n_legit
        
        # Legitimate URL patterns
        legit_domains = [
            "google.com", "facebook.com", "amazon.com", "microsoft.com",
            "apple.com", "github.com", "stackoverflow.com", "wikipedia.org",
            "linkedin.com", "twitter.com", "youtube.com", "reddit.com",
            "netflix.com", "dropbox.com", "salesforce.com", "adobe.com",
            "oracle.com", "ibm.com", "intel.com", "cisco.com",
            "paypal.com", "ebay.com", "walmart.com", "target.com",
            "bestbuy.com", "homedepot.com", "costco.com", "nike.com",
            "bbc.co.uk", "cnn.com", "nytimes.com", "washingtonpost.com",
            "medium.com", "wordpress.com", "blogger.com", "tumblr.com",
            "spotify.com", "soundcloud.com", "twitch.tv", "discord.com",
            "zoom.us", "slack.com", "trello.com", "notion.so",
            "figma.com", "canva.com", "shutterstock.com", "unsplash.com",
            "python.org", "numpy.org", "pytorch.org", "tensorflow.org"
        ]
        
        legit_paths = [
            "/", "/about", "/contact", "/help", "/login", "/signup",
            "/products", "/services", "/blog", "/news", "/pricing",
            "/docs", "/api", "/support", "/faq", "/terms", "/privacy",
            "/settings", "/profile", "/dashboard", "/search", "/download",
            "/features", "/enterprise", "/solutions", "/partners",
            "/careers", "/press", "/investors", "/security",
            "/community", "/resources", "/tutorials", "/guides",
        ]
        
        legit_urls = []
        for _ in range(n_legit):
            domain = np.random.choice(legit_domains)
            path = np.random.choice(legit_paths)
            protocol = "https" if np.random.random() > 0.1 else "http"
            www = "www." if np.random.random() > 0.4 else ""
            
            # Some URLs have query parameters
            query = ""
            if np.random.random() > 0.7:
                params = ["id", "page", "ref", "source", "lang", "q", "category"]
                param = np.random.choice(params)
                val = np.random.choice(["1", "2", "en", "us", "home", "main", "true"])
                query = f"?{param}={val}"
            
            url = f"{protocol}://{www}{domain}{path}{query}"
            legit_urls.append(url)
        
        # Phishing URL patterns
        phish_techniques = {
            'misspelling': [
                "g00gle.com", "faceb00k.com", "arnazon.com", "micr0soft.com",
                "app1e.com", "paypai.com", "amaz0n.com", "go0gle.com",
                "linkedln.com", "twltter.com", "netfllx.com", "dropb0x.com",
                "faceboook.com", "gooogle.com", "amazom.com", "mircosoft.com",
                "lnstagram.com", "whatsaap.com", "yah00.com", "0utlook.com"
            ],
            'subdomain_abuse': [
                "login.google.com.evil.xyz", "secure.paypal.com.phish.tk",
                "account.amazon.com.verify.ml", "signin.microsoft.com.auth.ga",
                "www.apple.com.secure.phish.tk", "update.netflix.com.verify.ml",
                "login.facebook.com.security.ga", "auth.dropbox.com.verify.cf",
                "secure.bank.com.login.tk", "verify.paypal.com.auth.ml",
                "signin.apple.com.verify.ga", "login.ebay.com.secure.tk",
                "account.chase.com.verify.ml", "auth.wellsfargo.com.login.ga",
                "secure.citibank.com.auth.cf", "verify.bankofamerica.com.login.tk"
            ],
            'ip_based': [
                "http://192.168.1.1/login", "http://10.0.0.1/paypal/verify",
                "http://172.16.0.1/signin", "http://203.0.113.1/account",
                "http://198.51.100.1/secure/login", "http://192.0.2.1/banking",
                "http://100.24.56.78/verify-account", "http://54.123.45.67/signin",
                "http://13.234.56.78/login/verify", "http://52.66.78.90/secure",
                "http://35.154.67.89/account/verify", "http://3.6.78.90/banking/login"
            ],
            'long_url': [
                "http://this-is-a-very-long-phishing-url-designed-to-trick-users.com/login/secure/verify/account/update/confirm",
                "http://secure-login-verification-account-update-required-immediately.com/signin",
                "http://your-account-has-been-compromised-click-here-to-verify-now.com/verify",
                "http://important-security-update-required-for-your-account-protection.com/update",
                "http://verify-your-identity-to-prevent-account-suspension-now.com/auth",
                "http://urgent-action-required-suspicious-activity-detected-on-account.com/secure",
                "http://immediate-verification-required-to-restore-account-access-now.com/login",
                "http://account-security-alert-unusual-signin-activity-detected-verify.com/confirm"
            ],
            'suspicious_tld': [
                "secure-login.tk/paypal", "verify-account.ml/google",
                "account-update.ga/amazon", "secure-signin.cf/microsoft",
                "login-verify.xyz/apple", "auth-secure.top/facebook",
                "account-confirm.buzz/netflix", "signin-secure.club/dropbox",
                "verify-login.work/linkedin", "secure-auth.site/twitter",
                "update-account.online/instagram", "confirm-signin.info/whatsapp",
                "login-secure.biz/yahoo", "auth-verify.pw/outlook",
                "signin-confirm.cc/chase", "verify-auth.ws/wellsfargo"
            ],
            'encoded': [
                "http://phishing.com/%70%61%79%70%61%6C/login",
                "http://evil.com/%67%6F%6F%67%6C%65/signin",
                "http://fake.com/%61%6D%61%7A%6F%6E/verify",
                "http://scam.com/%6D%69%63%72%6F%73%6F%66%74/auth",
                "http://trap.com/%61%70%70%6C%65/secure",
                "http://bait.com/%66%61%63%65%62%6F%6F%6B/login",
                "http://hook.com/%6E%65%74%66%6C%69%78/update",
                "http://lure.com/%64%72%6F%70%62%6F%78/verify"
            ],
            'at_symbol': [
                "http://google.com@evil.com/login",
                "http://paypal.com@phishing.com/signin",
                "http://amazon.com@scam.com/verify",
                "http://microsoft.com@fake.com/auth",
                "http://apple.com@trap.com/secure",
                "http://facebook.com@bait.com/login",
                "http://netflix.com@hook.com/update",
                "http://dropbox.com@lure.com/verify"
            ]
        }
        
        phish_urls = []
        techniques_list = list(phish_techniques.keys())
        
        for _ in range(n_phish):
            technique = np.random.choice(techniques_list)
            base_urls = phish_techniques[technique]
            base = np.random.choice(base_urls)
            
            # Add random variations
            if np.random.random() > 0.5:
                suffixes = ["/index.html", "/default.asp", "/home.php",
                           "/verify.html", "/confirm.php", "/update.asp",
                           "/secure/login.html", "/auth/verify.php"]
                base += np.random.choice(suffixes)
            
            if np.random.random() > 0.6:
                params = ["token", "session", "auth", "verify", "uid", "ref"]
                vals = ["abc123", "xyz789", "true", "1", "confirm", "urgent"]
                param = np.random.choice(params)
                val = np.random.choice(vals)
                separator = "&" if "?" in base else "?"
                base += f"{separator}{param}={val}"
            
            phish_urls.append(base)
        
        # Create DataFrame
        urls = legit_urls + phish_urls
        labels = [0] * n_legit + [1] * n_phish  # 0 = legitimate, 1 = phishing
        
        df = pd.DataFrame({'url': urls, 'status': labels})
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated dataset shape: {df.shape}")
        logger.info(f"Class distribution:\n{df['status'].value_counts()}")
        
        return df
    
    def download_from_url(self, url: str, filename: str = "dataset.csv") -> pd.DataFrame:
        """
        Download dataset from a URL.
        
        Args:
            url: URL to download the dataset from
            filename: Name to save the file as
            
        Returns:
            pd.DataFrame with the downloaded data
        """
        logger.info(f"Attempting download from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"Dataset downloaded and saved to {filepath}")
            logger.info(f"Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            logger.info("Falling back to synthetic dataset generation...")
            return None
    
    def prepare_dataset(self) -> pd.DataFrame:
        """
        Main method to prepare the dataset.
        Tries to download from URL first, falls back to synthetic data.
        
        Returns:
            pd.DataFrame with the prepared dataset
        """
        logger.info("=" * 60)
        logger.info("MODULE 3: Dataset Acquisition")
        logger.info("=" * 60)
        
        # Try downloading first
        dataset_url = "https://raw.githubusercontent.com/GregaVrbancic/PhishingDataset/master/dataset_full.csv"
        df = self.download_from_url(dataset_url)
        
        if df is None:
            # Generate synthetic dataset
            df = self.generate_synthetic_dataset(n_samples=11000)
        
        # Ensure consistent column names
        if 'url' not in df.columns:
            # Try to identify URL column
            url_cols = [c for c in df.columns if 'url' in c.lower()]
            if url_cols:
                df = df.rename(columns={url_cols[0]: 'url'})
        
        if 'status' not in df.columns:
            # Try to identify target column
            target_cols = [c for c in df.columns if any(t in c.lower() for t in ['status', 'label', 'target', 'phishing', 'class'])]
            if target_cols:
                df = df.rename(columns={target_cols[0]: 'status'})
        
        # Save the dataset
        output_path = os.path.join(self.output_dir, "phishing_dataset.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df


def main():
    """Main function to run dataset acquisition."""
    downloader = PhishingDatasetDownloader()
    df = downloader.prepare_dataset()
    
    print("\n" + "=" * 60)
    print("Dataset Acquisition Complete!")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns)}")
    if 'status' in df.columns:
        print(f"\nClass Distribution:")
        print(df['status'].value_counts())
    print(f"\nSaved to: data/raw/phishing_dataset.csv")


if __name__ == "__main__":
    main()
