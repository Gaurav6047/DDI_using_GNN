<div style="
  max-width: 1000px;
  margin: 40px auto;
  padding: 36px 40px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.7;
  color: #1a1a1a;
  background: #ffffff;
">

<h1 style="text-align:center; color:#0b3c5d; font-size:2.4em; margin-bottom:6px;">
Reliability-Aware Drug–Drug Interaction Prediction
</h1>

<p style="text-align:center; color:#555; font-size:1.05em;">
A research-grade framework for leakage-free, uncertainty-aware, and reliability-calibrated DDI prediction
</p>

<hr style="margin:32px 0; border:0; border-top:1px solid #e6e6e6;">

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Overview</h2>

<p>
This repository presents a <strong>research-grade framework for multi-class Drug–Drug Interaction (DDI) prediction</strong>
with a strong emphasis on <strong>generalization, uncertainty modeling, and reliability auditing</strong>.
</p>

<p>
The system is trained on the <strong>DrugBank dataset</strong> and predicts
<strong>76 distinct interaction types</strong> using a <strong>Siamese Graph Neural Network (GNN)</strong> architecture.
Evaluation is conducted under a <strong>strict pair-level Murcko scaffold split</strong>,
ensuring <strong>zero chemical scaffold leakage</strong> between training, validation, and test sets.
</p>

<p>
Beyond standard predictive modeling, the framework integrates:
</p>

<ul>
  <li>Epistemic uncertainty estimation via Monte Carlo Dropout</li>
  <li>An independent external auditor model</li>
  <li>Selective prediction based on a composite reliability score</li>
</ul>

<p>
The goal of this project is not merely high accuracy, but
<strong>trustworthy, deployment-aware decision making</strong> for safety-critical biomedical applications.
</p>

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Motivation</h2>

<p>
Many existing DDI prediction systems report strong performance metrics but suffer from
fundamental methodological weaknesses:
</p>

<ul>
  <li>Chemical scaffold leakage between training and evaluation data</li>
  <li>Inflated performance due to unrealistic random splits</li>
  <li>No explicit modeling of prediction uncertainty</li>
  <li>No independent mechanism to audit or verify predictions</li>
  <li>Over-reliance on softmax confidence as a proxy for trustworthiness</li>
</ul>

<p>
In pharmacological and biomedical settings, such limitations can lead to
<strong>unsafe or misleading conclusions</strong>.
</p>

<p>
This work prioritizes <strong>scientific rigor, reliability, and real-world deployment realism</strong>
over headline accuracy.
</p>

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Key Contributions</h2>

<div style="background:#f8f9fa; border-left:4px solid #0b3c5d; padding:14px 18px; margin:18px 0;">
<strong>1. Leakage-Free Evaluation Protocol</strong><br><br>
<ul>
  <li>Drug–drug interactions are split using a <strong>pair-level Murcko scaffold strategy</strong></li>
  <li>Both drugs jointly define the scaffold identity of an interaction</li>
  <li>Training and test sets share <strong>zero overlapping scaffolds</strong></li>
  <li>Guarantees evaluation on <strong>genuinely unseen chemical structures</strong></li>
</ul>
This protocol closely reflects real-world drug discovery and pharmacovigilance scenarios.
</div>

<p><strong>2. Graph-Based Molecular Representation</strong></p>

<p>
Each drug is represented as a molecular graph constructed from SMILES strings using RDKit.
</p>

<p><em>Atom features include:</em></p>
<ul>
  <li>Atomic number</li>
  <li>Total degree</li>
  <li>Hybridization state</li>
  <li>Aromaticity</li>
  <li>Formal charge</li>
  <li>Ring membership</li>
  <li>Radical electrons</li>
  <li>Chirality tag</li>
</ul>

<p><em>Bond features include:</em></p>
<ul>
  <li>Bond type</li>
  <li>Conjugation</li>
  <li>Ring participation</li>
</ul>

<p>
This representation preserves detailed chemical structure and topology.
</p>

<p><strong>3. Siamese Graph Neural Network Architecture</strong></p>

<ul>
  <li>Two identical GATv2-based encoders (one per drug)</li>
  <li>Edge-aware attention convolutions</li>
  <li>Layer normalization and dropout for stability</li>
  <li>Global mean and max pooling</li>
  <li>Fully connected classifier for interaction prediction</li>
</ul>

<p>
This architecture learns <strong>interaction-specific patterns</strong>,
not just individual molecular properties.
</p>

<p><strong>4. Imbalance-Aware Training Strategy</strong></p>

<ul>
  <li>Class-balanced weighting</li>
  <li>Square-root reweighting</li>
  <li>Focal loss for hard and minority classes</li>
  <li>Gradient clipping for training stability</li>
  <li>Mixed-precision training (AMP) for efficiency</li>
</ul>

<p><strong>5. Generalization Performance</strong></p>

<ul>
  <li>Weighted F1-score: <strong>≈ 0.84–0.85</strong></li>
  <li>Macro F1-score: <strong>≈ 0.82–0.83</strong></li>
  <li>Number of interaction classes: <strong>76</strong></li>
</ul>

<p>
All metrics are reported on a <strong>scaffold-held-out test set</strong>,
demonstrating strong generalization.
</p>

<p><strong>6. Uncertainty Estimation (MC Dropout)</strong></p>

<ul>
  <li>Dropout active at inference time</li>
  <li>Multiple stochastic forward passes</li>
  <li>Entropy-based uncertainty estimation</li>
</ul>

<p><strong>7. External Reliability Auditor</strong></p>

<ul>
  <li>Independent Random Forest auditor</li>
  <li>Morgan fingerprint-based drug pair representation</li>
  <li>Cross-validated consistency checking</li>
</ul>

<p><strong>8. Reliability Scoring & Selective Prediction</strong></p>

<ul>
  <li>Confidence-based scoring</li>
  <li>Uncertainty penalties</li>
  <li>Class rarity penalties</li>
  <li>Auditor agreement weighting</li>
</ul>

<p>
Accuracy improves as coverage decreases:
</p>

<ul>
  <li>100% coverage → ~0.85 accuracy</li>
  <li>80% coverage → ~0.92 accuracy</li>
  <li>60% coverage → ~0.96 accuracy</li>
  <li>40% coverage → ~0.98 accuracy</li>
</ul>

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Intended Applications</h2>

<ul>
  <li>Drug–drug interaction screening</li>
  <li>Pharmacovigilance research</li>
  <li>Reliability-aware biomedical machine learning</li>
  <li>Decision-support systems for early-stage drug discovery</li>
</ul>

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Limitations & Future Work</h2>

<ul>
  <li>Single-dataset evaluation (DrugBank)</li>
  <li>Single-seed training</li>
  <li>Auditor limited to classical ML models</li>
</ul>

<p>
Future work includes multi-dataset validation, seed-averaged analysis,
and neural or ensemble-based auditors.
</p>

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Ethical Considerations</h2>

<p>
This framework is intended <strong>strictly for research and educational use</strong>.
It must not be used for clinical decision-making without expert validation
and regulatory approval.
</p>

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Author</h2>

<p>
<strong>Gaurav</strong><br>
Machine Learning Engineer<br>
Focus areas: Graph Neural Networks, Biomedical AI, Reliable Deep Learning
</p>

<h2 style="color:#0b3c5d; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Summary</h2>

<p>
This work demonstrates that <strong>accurate DDI prediction alone is insufficient</strong>.
By combining leakage-free evaluation, graph-based modeling, uncertainty estimation,
and independent reliability auditing, the framework provides a
<strong>trustworthy and deployment-aware solution</strong> for drug–drug interaction prediction.
</p>

</div>
