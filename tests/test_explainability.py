"""Regression tests for local URL feature attributions."""

from src.explainability.explainer import PhishingExplainer


def test_listed_xai_drivers_have_contribution_details():
    explainer = PhishingExplainer()
    explanation = explainer.explain_url(
        "http://login.google.com.security-verify.tk/auth?id=9928"
    )

    listed_drivers = (
        explanation["top_phishing_drivers"]
        + explanation["top_legitimate_drivers"]
    )
    assert all(driver in explanation["all_contributions"] for driver in listed_drivers)
