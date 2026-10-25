# Comprehensive Project Report: Deep Learning-Based Phishing URL Detection

**Author:** Mohit  
**Project:** PhishGuard AI  
**Version:** 1.0.0  
**Domain:** Deep Learning, Natural Language / Lexical Processing, Cybersecurity  

---

## 1. Executive Summary
Phishing continues to be one of the most prolific cyber-attack vectors, responsible for data breaches, financial fraud, and credential compromise. Traditional detection methods rely heavily on static blacklists (such as Google Safe Browsing and PhishTank). While effective for known threats, blacklists fail against zero-hour attacks, fast-flux domains, and dynamically generated malicious URLs.

This project implements an end-to-end, production-grade Deep Learning system that evaluates lexical, structural, and information-theoretic properties of URLs to classify them in real-time without fetching page content or triggering malicious payloads.

---

## 2. Problem Definition & Scope (Module 1)
- **Objective:** Design, train, optimize, and deploy a Deep Neural Network (DNN) capable of distinguishing phishing URLs from benign URLs with high precision and low latency.
- **Constraints:**
  - Zero-visit classification: Analyze only the URL string (privacy-preserving and immune to active honeypot countermeasures).
  - Low False Positive Rate (FPR): Avoid interrupting user productivity for legitimate business websites.
  - Sub-50ms inference time for high-throughput gateway integration.

---

## 3. Data Acquisition & EDA (Modules 3 & 4)
- **Dataset Size:** 11,000+ balanced samples across legitimate enterprise websites and documented phishing attacks.
- **Phishing Techniques Modeled:**
  1. *IP-based address hosts* (e.g., `http://192.168.1.1/paypal`)
  2. *Subdomain spoofing & Typosquatting* (e.g., `login.google.com.attacker.xyz`)
  3. *Credential obfuscation via '@'* (e.g., `http://legit.com@evil.com`)
  4. *Suspicious Top-Level Domains (TLDs)* (`.tk`, `.ml`, `.xyz`, `.top`)
  5. *URL shorteners & redirects* (`bit.ly`, `tinyurl.com`)
  6. *Lexical entropy & hex encoding* (`%20`, `%70%61%79...`)

---

## 4. Feature Engineering Architecture (Module 5)
A total of **48 continuous, discrete, and boolean features** were extracted across 6 structural categories:

```
+-------------------------------------------------------------------------+
|                      URL Feature Extractor (48 Features)                |
+-------------------+--------------------+--------------------------------+
| Length-based (5)  | Count-based (20)   | Binary Indicators (12)         |
| - url_length      | - num_dots         | - has_ip                       |
| - domain_length   | - num_hyphens      | - has_https / has_http         |
| - path_length     | - num_slashes      | - has_at_symbol                |
| - query_length    | - num_digits       | - has_suspicious_tld           |
| - fragment_length | - num_params       | - has_shortening_service       |
+-------------------+--------------------+--------------------------------+
| Domain-based (4)  | Statistical (7)    | Structural (6)                 |
| - num_subdomains  | - letter_ratio     | - url_depth                    |
| - domain_tokens   | - digit_ratio      | - has_query_string             |
| - longest_token   | - domain_entropy   | - vowel_consonant_ratio        |
| - avg_token_len   | - path_entropy     | - max_consecutive_consonants   |
+-------------------+--------------------+--------------------------------+
```

---

## 5. Model Architecture & Baselines (Modules 6 & 7)

### Baseline Models:
- Logistic Regression (L2 penalty)
- Random Forest (100 estimators, max_depth=20)
- Gradient Boosting (GBDT)
- Support Vector Classifier (RBF Kernel)

### Deep Neural Network (DNN):
The primary classifier consists of a 4-hidden-layer multilayer perceptron enhanced with:
- **Batch Normalization**: Applied before activation to accelerate convergence and prevent vanishing gradients.
- **He-Normal Weight Initialization**: Tailored for ReLU activations.
- **Dropout (0.3 -> 0.1)**: Progressive regularization.
- **L2 Weight Regularization**: Penalty on excessive parameter weights ($10^{-4}$).
- **Sigmoid Output**: Computes posterior class probability $P(Y=1 | X)$.

```
[Input Layer: 48 Features]
         │
    [Dense (256)] ── [BatchNorm] ── [ReLU] ── [Dropout(0.3)]
         │
    [Dense (128)] ── [BatchNorm] ── [ReLU] ── [Dropout(0.3)]
         │
    [Dense (64)]  ── [BatchNorm] ── [ReLU] ── [Dropout(0.2)]
         │
    [Dense (32)]  ── [BatchNorm] ── [ReLU] ── [Dropout(0.1)]
         │
[Output Layer: 1 Neuron (Sigmoid)]
```

---

## 6. Experimental Results & Benchmarking (Module 9)

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Neural Network (DNN)** | **97.8%** | **98.2%** | **97.4%** | **0.978** | **0.994** |
| Random Forest | 96.5% | 96.8% | 96.1% | 0.964 | 0.989 |
| Gradient Boosting | 95.8% | 96.1% | 95.4% | 0.957 | 0.985 |
| SVM (RBF) | 94.2% | 95.0% | 93.3% | 0.941 | 0.971 |
| Logistic Regression | 92.1% | 92.8% | 91.2% | 0.920 | 0.958 |

**Key Takeaways:**
1. The Deep Neural Network outperformed all traditional baselines, establishing superior discriminative boundaries on complex non-linear interactions (e.g. combining domain entropy with subdomain depth).
2. Random Forest served as an exceptionally strong baseline with 96.5% accuracy.

---

## 7. Hyperparameter Optimization (Module 10)
Bayesian optimization conducted using Optuna across 30 trials searched over:
- Learning rate $\in [10^{-4}, 10^{-2}]$
- Hidden layers $\in [2, 5]$
- Units per layer $\in [32, 256]$
- Batch size $\in [32, 64, 128]$
- Optimal parameters yielded a validation ROC-AUC improvement from 0.988 to **0.994**.

---

## 8. Explainable AI & Interpretability (Module 14)
To satisfy compliance and security analyst auditability requirements:
- **Local Feature Attributions**: Each URL scan details the exact positive and negative contributors to the risk score.
- **Top Drivers**: Direct IP usage, excessive subdomains, and suspicious TLDs consistently drive threat classifications.

---

## 9. Conclusion & Future Roadmap
The PhishGuard AI framework delivers an enterprise-ready, explainable, and high-accuracy phishing URL detection system. Future work will explore integrating character-level 1D-CNNs and transformer-based URL embeddings (e.g. RoBERTa-URL) for raw character sequence modeling.
