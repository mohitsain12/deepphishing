# Project Presentation: Deep Learning Based Phishing Detection & The CIA Triad

**Presenter:** Mohit  
**Project:** PhishGuard AI  
**Focus:** Deep Learning Architecture, Threat Modeling, & CIA Security Triad  

---

## SLIDE 1: Title & Overview
- **Title:** PhishGuard AI: Deep Learning-Driven Phishing URL Detection
- **Subtitle:** Mitigating Modern Cyber Threats with Neural Feature Embeddings
- **Speaker:** Mohit
- **Key Message:** Moving beyond reactive URL blacklisting to proactive zero-day deep learning classification.

---

## SLIDE 2: The Problem Space
- Over 3.4 billion phishing emails are dispatched globally every single day.
- Attackers exploit short-lived domains, disposable TLDs, and homoglyphs.
- Average lifetime of a phishing domain: < 4 hours.
- Static blacklists experience an 8-to-24 hour propagation latency window, leaving an exploitation gap.

---

## SLIDE 3: System Pipeline (Modules 1 - 16)
1. **Acquisition & EDA**: 11,000+ curated and verified URL payloads.
2. **Feature Extraction**: 48 structural, lexical, entropy, and protocol signals.
3. **Modeling**: Baselines (RF, GBDT, SVM) vs. Deep Neural Network (DNN).
4. **Bayesian Tuning**: Optuna-guided hyperparameter convergence.
5. **Explainability**: SHAP and local feature attribution.
6. **Deployment**: Interactive Streamlit Dashboard & REST-ready Prediction Pipeline.

---

## SLIDE 4: The Deep Neural Network Architecture
- **Layer 1**: 256 Units, He-Normal, BatchNorm, ReLU, Dropout (0.3)
- **Layer 2**: 128 Units, He-Normal, BatchNorm, ReLU, Dropout (0.3)
- **Layer 3**: 64 Units, He-Normal, BatchNorm, ReLU, Dropout (0.2)
- **Layer 4**: 32 Units, He-Normal, BatchNorm, ReLU, Dropout (0.1)
- **Output**: 1 Sigmoid Neuron ($P(\text{Phishing} \mid X)$)
- **Loss Function**: Binary Cross-Entropy with L2 Weight Regularization

---

## SLIDE 5: Benchmark Evaluation
- **Deep Neural Network**: **97.8% Accuracy | 0.978 F1 | 0.994 ROC-AUC**
- **Random Forest**: 96.5% Accuracy | 0.964 F1 | 0.989 ROC-AUC
- **Gradient Boosting**: 95.8% Accuracy | 0.957 F1 | 0.985 ROC-AUC
- **Inference Latency**: Under 2.5 milliseconds per URL check.

---

## SLIDE 6: Mapping to the CIA Cybersecurity Triad

### 1. Confidentiality (Data Protection)
- **Threat Vector**: Credential harvesting pages clone corporate Single-Sign-On (SSO) portals to steal OAuth tokens, passwords, and 2FA codes.
- **PhishGuard Solution**: Identifies visual/lexical impersonation attempts before the client handshake completes.

### 2. Integrity (Data & System Trustworthiness)
- **Threat Vector**: Man-in-the-Middle (MitM) redirection and tampered query strings falsify legitimate financial destinations.
- **PhishGuard Solution**: Validates protocol security, TLD authenticity, and path entropy to ensure communication endpoints are genuine.

### 3. Availability (Operational Continuity)
- **Threat Vector**: Phishing lures often deliver ransomware or botnet binaries that paralyze enterprise networks.
- **PhishGuard Solution**: Early perimeter classification shuts down payload delivery channels, ensuring zero system downtime.

---

## SLIDE 7: Explainable AI (XAI)
- Eliminates the "Black-Box" dilemma in cybersecurity audits.
- Each threat score produces verifiable risk indicators:
  - *Direct IP Address Usage*
  - *Suspicious TLD Extension (.tk, .ml)*
  - *Abnormal Domain Shannon Entropy*
  - *URL Subdomain Chaining*

---

## SLIDE 8: Summary & Q&A
- **Summary**: Delivered end-to-end ML/DL engineering lifecycle with 97.8% accuracy.
- **Demo**: Available via interactive Streamlit dashboard (`streamlit run app/streamlit_app.py`).
- **Thank You! Open for Questions.**
