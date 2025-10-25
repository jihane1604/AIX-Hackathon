- Centralizes every knob that controls **risk labeling**, **readiness** **scoring**, **gap prioritization**, and **explanations**.

- Separates **policy** (this file) from **code** so you can change weights, thresholds, and templates without redeploying.

- Supports both static domains (explicitly listed) and dynamic domains (induced from regulator text). Unknown domains get the dynamic_domains.fallback_weight with optional size bias.

- Keeps scoring deterministic and auditable: model predicts evidence and risk signals; this file defines how to convert them into a final score and narrative.

- Needs to be updated after all domains and weights are determined