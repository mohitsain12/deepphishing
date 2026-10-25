"""
MODULE 13: Testing - Feature Extraction Unit Tests
Verifies feature extraction functions across standard, edge-case, and adversarial URLs.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.feature_engineering import URLFeatureExtractor


@pytest.fixture
def extractor():
    return URLFeatureExtractor()


def test_extractor_initialization(extractor):
    assert extractor is not None


def test_legitimate_url_features(extractor):
    url = "https://www.google.com/search?q=machine+learning"
    feats = extractor.extract_features(url)
    
    assert isinstance(feats, dict)
    assert feats['has_https'] == 1
    assert feats['has_http'] == 0
    assert feats['has_ip'] == 0
    assert feats['has_at_symbol'] == 0
    assert feats['url_length'] > 0
    assert feats['domain_length'] > 0
    assert feats['num_dots'] >= 2


def test_phishing_ip_url(extractor):
    url = "http://192.168.1.1/secure/login"
    feats = extractor.extract_features(url)
    
    assert feats['has_ip'] == 1
    assert feats['has_https'] == 0
    assert feats['has_http'] == 1


def test_at_symbol_obfuscation(extractor):
    url = "http://paypal.com@evil-attacker.com/signin"
    feats = extractor.extract_features(url)
    
    assert feats['has_at_symbol'] == 1
    assert feats['num_at'] == 1


def test_suspicious_tld_detection(extractor):
    url = "http://verify-bank.tk/update"
    feats = extractor.extract_features(url)
    
    assert feats['has_suspicious_tld'] == 1


def test_shortening_service_detection(extractor):
    url = "https://bit.ly/3xyzAbc"
    feats = extractor.extract_features(url)
    
    assert feats['has_shortening_service'] == 1


def test_entropy_computation(extractor):
    random_str = "x7k9p2m4q1w8z"
    repetitive_str = "aaaaaaaaaaaa"
    
    ent_random = extractor._calculate_entropy(random_str)
    ent_rep = extractor._calculate_entropy(repetitive_str)
    
    assert ent_random > ent_rep
    assert ent_rep == 0.0


def test_empty_or_malformed_url(extractor):
    feats = extractor.extract_features("")
    assert isinstance(feats, dict)
    assert 'url_length' in feats
