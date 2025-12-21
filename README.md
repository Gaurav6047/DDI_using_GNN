# Reliability-Aware Drug–Drug Interaction Prediction

## Overview

This repository presents a **research-grade framework for multi-class Drug–Drug Interaction (DDI) prediction** with a strong emphasis on **generalization, uncertainty modeling, and reliability auditing**.

The system is trained on the **DrugBank dataset** and predicts **76 distinct interaction types** using a **Siamese Graph Neural Network (GNN)** architecture.  
Evaluation is conducted under a **strict pair-level Murcko scaffold split**, ensuring **zero chemical scaffold leakage** between training, validation, and test sets.

Beyond standard predictive modeling, the framework integrates:
- epistemic uncertainty estimation via Monte Carlo Dropout,
- an independent external auditor model,
- and selective prediction based on a composite reliability score.

The goal of this project is not merely high accuracy, but **trustworthy, deployment-aware decision making** for safety-critical biomedical applications.

---

## Motivation

Many existing DDI prediction systems report strong performance metrics but suffer from fundamental methodological weaknesses:

- Chemical scaffold leakage between training and evaluation data
- Inflated performance due to unrealistic random splits
- No explicit modeling of prediction uncertainty
- No independent mechanism to audit or verify predictions
- Over-reliance on softmax confidence as a proxy for trustworthiness

In pharmacological and biomedical settings, such limitations can lead to **unsafe or misleading conclusions**.

This work prioritizes **scientific rigor, reliability, and real-world deployment realism** over headline accuracy.

---

## Key Contributions

### 1. Leakage-Free Evaluation Protocol

- Drug–drug interactions are split using a **pair-level Murcko scaffold strategy**
- Both drugs jointly define the scaffold identity of an interaction
- Training and test sets share **zero overlapping scaffolds**
- Guarantees evaluation on **genuinely unseen chemical structures**

This protocol closely reflects real-world drug discovery and pharmacovigilance scenarios.

---

### 2. Graph-Based Molecular Representation

Each drug is represented as a molecular graph constructed from SMILES strings using RDKit.

**Atom features include:**
- Atomic number
- Total degree
- Hybridization state
- Aromaticity
- Formal charge
- Ring membership
- Radical electrons
- Chirality tag

**Bond features include:**
- Bond type
- Conjugation
- Ring participation

This representation preserves detailed chemical structure and topology.

---

### 3. Siamese Graph Neural Network Architecture

The predictive model follows a **Siamese GNN design with shared weights**:

- Two identical GATv2-based encoders (one per drug)
- Edge-aware attention convolutions
- Layer normalization and dropout for stability
- Global mean and max pooling
- Fully connected classifier for interaction prediction

This architecture learns **interaction-specific patterns**, not just individual molecular properties.

---

### 4. Imbalance-Aware Training Strategy

The dataset exhibits strong class imbalance across interaction types.

This is addressed through:
- Class-balanced weighting
- Square-root reweighting
- Focal loss for hard and minority classes
- Gradient clipping for training stability
- Mixed-precision training (AMP) for computational efficiency

---

### 5. Generalization Performance

All reported metrics are obtained from a **scaffold-held-out test set**.

- Weighted F1-score: approximately **0.84–0.85**
- Macro F1-score: approximately **0.82–0.83**
- Number of interaction classes: **76**

These results demonstrate strong generalization under chemically realistic evaluation conditions.

---

### 6. Uncertainty Estimation via Monte Carlo Dropout

Epistemic uncertainty is modeled using **Monte Carlo Dropout**:

- Dropout layers remain active at inference time
- Multiple stochastic forward passes are performed
- Mean predictive probabilities and entropy are computed

This allows quantification of **model uncertainty per prediction**, rather than relying solely on softmax confidence.

---

### 7. External Reliability Auditor

In addition to uncertainty modeling, the framework employs an **independent external auditor**:

- Auditor model: Random Forest classifier
- Input: concatenated Morgan fingerprints of drug pairs
- Evaluation: stratified cross-validation
- Provides an external consistency check against GNN predictions

Disagreement between the GNN and the auditor reduces the final reliability score.

---

### 8. Reliability Scoring Mechanism

For each test prediction, a reliability score is computed by combining:

- Model confidence (softmax probability)
- MC Dropout uncertainty (entropy-based penalty)
- Class rarity penalty (based on training distribution)
- Auditor agreement or disagreement

This score reflects **how trustworthy a prediction is**, not merely how confident the model appears.

---

### 9. Selective Prediction and Risk Control

Using reliability scores, the system supports **selective prediction**:

- Low-reliability predictions can be rejected or deferred
- Accuracy increases as coverage decreases

Observed behavior on the test set:

- Full coverage accuracy ≈ 0.85
- ~80% coverage → accuracy ≈ 0.92
- ~60% coverage → accuracy ≈ 0.96
- ~40% coverage → accuracy ≈ 0.98

This behavior is essential for **safety-critical biomedical applications**.

---

## Research Artifacts

The framework produces and stores the following reproducible artifacts:

- Fine-tuned Siamese GNN model
- Trained Random Forest auditor
- Confusion matrix (CSV)
- Calibration curve
- Selective prediction results
- Full experiment metadata (JSON)

All artifacts are reproducible and export-ready.

---

## Intended Applications

- Drug–drug interaction screening
- Pharmacovigilance research
- Reliability-aware biomedical machine learning
- Decision-support systems for early-stage drug discovery

This project is intended strictly for **research and educational use**.

---

## Strengths

- Zero scaffold leakage evaluation
- Strong generalization to unseen chemical structures
- Explicit uncertainty modeling via MC Dropout
- Independent reliability auditing
- Selective prediction for risk-aware decision making
- Comprehensive artifact logging and reproducibility

---

## Limitations and Future Work

- No full baseline comparison under identical scaffold splits
- Evaluation limited to a single dataset (DrugBank)
- Training performed with a single random seed
- Auditor limited to classical fingerprint-based models

Future work may explore:
- Multi-dataset validation (e.g., TWOSIDES)
- Seed-averaged statistical analysis
- Neural or ensemble-based auditor models
- Joint training of predictor and auditor components

---

## Ethical Considerations

This framework is intended **solely for defensive biomedical research and decision support**.  
It must not be used for clinical decision making without expert validation and regulatory approval.

---

## Author

**Gaurav**  
Machine Learning Engineer  
Focus areas: Graph Neural Networks, Biomedical AI, Reliable Deep Learning

---

## Summary

This work demonstrates that **accurate DDI prediction alone is insufficient for real-world use**.

By combining leakage-free evaluation, graph-based modeling, uncertainty estimation, and independent reliability auditing, the framework provides a **more trustworthy and deployment-aware approach** to drug–drug interaction prediction.
