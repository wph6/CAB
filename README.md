# CAB: Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge
## 🚕 Overview
CAB is a data-efficient framework for transferring attention knowledge from Transformer teachers to state-space student models such as Mamba.
It introduces a lightweight MLP-based bridge that aligns Transformer’s attention projections (Q/K) with Mamba’s dynamic projections (B/C), enabling fine-grained, token-level supervision.
CAB further adopts a hierarchical layer alignment strategy to handle architectural heterogeneity.
Across both vision and language tasks, CAB achieves superior performance and efficiency, demonstrating that attention-based inductive biases can be effectively transferred to recurrent models.
## 🔍 Key Features
- **Attention Bridge** – A lightweight MLP module that aligns Transformer attention (Q/K) with Mamba’s dynamic projections (B/C). This enables fine-grained, token-level supervision and allows effective transfer of attention structures into recurrent state-space models.
- **Dual Efficiency** – CAB achieves both computational and data efficiency: it avoids the heavy quadratic cost of dense attention alignment and remains effective in low-data regimes, making it a scalable solution for cross-architecture knowledge transfer.

## Quick Start
