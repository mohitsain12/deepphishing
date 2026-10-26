"""
MODULE 13: Testing - Prediction Pipeline Unit Tests
Verifies prediction pipeline handling of individual URLs, edge cases, and response contracts.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.predict_pipeline import PhishingPredictor


@pytest.fixture
def predictor():
    return PhishingPredictor()


def test_predictor_initialization(predictor):
    assert predictor is not None
    assert predictor.feature_extractor is not None


def test_predict_single_url_contract(predictor):
    url = "https://www.wikipedia.org"
    res = predictor.predict_url(url)
    
    assert isinstance(res, dict)
    assert 'url' in res
    assert 'prediction' in res
    assert res['prediction'] in [0, 1]
    assert 'probability' in res
    assert 0.0 <= res['probability'] <= 1.0
    assert 'risk_level' in res
    assert 'key_indicators' in res
    assert isinstance(res['key_indicators'], list)


def test_predict_phishing_url(predictor):
    url = "http://192.168.0.1/paypal.com/verify-account.php?token=123"
    res = predictor.predict_url(url)
    
    assert res['prediction'] == 1 or res['probability'] > 0.4
    assert res['risk_level'] in ['CRITICAL', 'HIGH', 'MODERATE']
