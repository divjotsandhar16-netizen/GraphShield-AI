GraphShield AI investigates whether explicitly modeling network communication relationships as graphs improves intrusion detection and generalization to attack types absent from training, compared with conventional flow-based machine-learning models
“The project investigates payload-independent flow-level detection that can operate without relying on decrypted application payloads; CIC-IDS2017 is used as a flow-level benchmark rather than as an exclusively encrypted-traffic dataset.”
B. Literature grounding

Your initial literature survey identified:

Zhong et al. (2024) — A Survey on Graph Neural Networks for Intrusion Detection Systems: Methods, Trends and Challenges

The paper explicitly discusses graph construction, network design, deployment, and challenges in GNN-based IDS.

Alshehri et al. (2025) — Systematic Review of Graph Neural Network for Malicious Attack Detection

The review analyzed 28 studies from 2020–2025 and identifies generalization and practical application as areas needing further work.

Your presentation uses these papers to frame the research gap.

But our literature survey should eventually grow

For the final research paper, two review papers aren't enough.
B. Literature grounding

Your initial literature survey identified:

Zhong et al. (2024) — A Survey on Graph Neural Networks for Intrusion Detection Systems: Methods, Trends and Challenges

The paper explicitly discusses graph construction, network design, deployment, and challenges in GNN-based IDS.

Alshehri et al. (2025) — Systematic Review of Graph Neural Network for Malicious Attack Detection

The review analyzed 28 studies from 2020–2025 and identifies generalization and practical application as areas needing further work.

Your presentation uses these papers to frame the research gap.

But our literature survey should eventually grow

For the final research paper, two review papers aren't enough.
RQ2 — Unseen attack generalization

How well do flow-based machine-learning and graph-based GNN models generalize to attack families that are absent from the training data?

This is probably your most important secondary question.

It connects directly to your project's zero-day motivation without claiming literal zero-day detection.

RQ3 — Does topology actually add information?

Does network topology provide predictive information beyond the original flow-level features?

Q4 — How does graph construction affect detection?

Your literature review specifically identifies graph construction as an important research issue.

So:

How do different graph construction choices—such as time-window size, node definition, and edge representation—affect intrusion-detection performance and computational cost?

RQ5 — What happens under temporal distribution shift?

How does model performance change when the evaluation traffic comes from a later capture period than the training traffic?

This is extremely useful because your random-split experiment already demonstrated that evaluation protocol matters.

RQ6 — What is the computational trade-off?

Your presentation already includes practical deployment/scaling as a research gap.

So we can ask:

What computational and scalability trade-offs are introduced by graph construction and GNN inference compared with conventional flow-based intrusion detection?Training time
Inference time
Memory usage
Number of nodes
Number of edges
Graph construction time
For the paper, I would keep the official set to five:

RQ1

Does explicit network topology improve intrusion-detection performance compared with conventional flow-based machine learning?

RQ2

Can graph-based models generalize to attack families that are absent from the training data?

RQ3

Does network topology provide information beyond flow-level features, as determined through graph-feature and model ablation studies?

RQ4

How do temporal graph construction choices affect detection performance and robustness under distribution shift?

RQ5

What computational trade-offs arise from graph construction and GNN inference compared with conventional tabular intrusion detection?
GraphShield AI is a research-oriented intrusion-detection framework that represents network flow interactions as communication graphs and investigates Graph Neural Networks for detecting anomalous and previously unseen attack patterns. The study compares graph-based learning with conventional flow-level machine-learning approaches under both conventional and temporally separated evaluation protocols. Rather than relying on decrypted application payloads, the framework operates on flow-level metadata and statistical traffic characteristics, allowing the study to examine the value of relational network information for intrusion detection. A controlled unseen-attack experiment is used as a proxy for zero-day generalization, while ablation studies investigate whether improvements arise from graph topology itself. The project additionally evaluates the computational trade-offs of graph construction and GNN inference and provides a visualization layer for experimental monitoring.


| Evaluation                       |         F1 |     Recall |    ROC-AUC |     PR-AUC |
| -------------------------------- | ---------: | ---------: | ---------: | ---------: |
| **Random stratified split**      | **0.9973** | **0.9985** | **1.0000** | **1.0000** |
| **Temporal validation — July 6** |     0.0037 |     0.0021 |     0.9408 |     0.0649 |
| **Temporal test — July 7**       |     0.4403 |     0.2824 |     0.9811 |     0.9656 |
