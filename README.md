# DualChannelHNN

A PyTorch implementation of a dual-channel cis/trans eQTL model. The model estimates cis additive effects (`beta`) and low-rank cis/trans interaction effects (`gamma`) from preprocessed genotype arrays. It combines a bidirectional GRU for cis SNPs, a graph neural network for trans SNPs, bidirectional cross-attention, post-normalized residual feed-forward blocks, and a low-rank interaction decoder.

## Intended data-processing context

In the intended analysis workflow, TensorQTL is used upstream to obtain and align genotype and phenotype arrays. Population structure and PEER-derived phenotype covariance are handled before the HNN: the PEER covariance component is projected out in the spatial domain, and the model is supplied the remaining orthogonal-complement/effect components. Consequently, DualChannelHNN is intended to model only the cis- and trans-SNP effect components; it is not itself a TensorQTL association engine, population-stratification correction, or PEER-projection implementation.

The included synthetic harness does not execute TensorQTL or PEER preprocessing; it generates synthetic genotypes and phenotypes to exercise the architecture.

## Prior structure and architecture

- **Cis channel:** a bidirectional (forward/reverse) GRU encodes ordered cis SNP inputs. Its recurrent representation is intended to encode prior information and capture possible LD structure among cis SNPs.
- **Trans channel:** trans genotypes are combined with signed genomic positions and module labels, embedded, and propagated over a fixed normalized adjacency with graph convolution. In the intended applied workflow, the trans graph prior is derived from RNA-seq co-expression/network analysis with pyWGCNA and organized to reflect the tree-like trans-SNP topology around the cis center. The model accepts this adjacency as an input.
- **Synthetic-demo distinction:** the included simulation harness constructs a radial adjacency from trans-SNP distances; it does not call pyWGCNA. Replace the supplied adjacency with the pyWGCNA-derived graph for the intended applied workflow.
- **Cross-channel fusion:** each channel attends to the other, then applies residual LayerNorm and a feed-forward block with another residual LayerNorm.
- **Effect decoding:** separate heads estimate cis `beta`, cis factors `z`, and trans factors `w`; `gamma` is formed as `z @ w.T` per sample and used in phenotype prediction.

![DualChannelHNN architecture and training flow](dual_channel_hnn_architecture.svg)

The diagram describes the representation flow, interaction/prediction equations, and the staged training procedure. See [`dual_channel_hnn.py`](dual_channel_hnn.py) for the reusable PyTorch model.

## Model usage

```python
import torch
from dual_channel_hnn import DualChannelHNN

# x_cis: (batch, n_cis), x_trans: (batch, n_trans)
# A_norm: (n_trans, n_trans); positions and module_ids: (n_trans,)
model = DualChannelHNN(
    hidden_dim=32,
    rank=2,
    adjacency=A_norm,
    trans_positions=trans_positions,
    module_ids=module_ids,
    target_gamma_std=target_gamma_std,
    beta_initial_std=beta_initial_std,
    attention_heads=4,
    n_gnn_layers=2,
)
model.calibrate_factor_gains(x_cis[:256], x_trans[:256])
y_hat, beta, z, w, gamma = model(x_cis, x_trans)
```

`gamma` has shape `(batch, n_cis, n_trans)`. For each sample it is the sum of rank-one factor products. Prediction is `mu + sum(beta * X_cis) + sum(gamma * X_cis * X_trans)`.

## Synthetic training results

The following results are from the revised post-normalized model, trained for 300 epochs on 1,200 synthetic samples per case with an 80/20 train/test split. Cases vary cis/trans SNP counts, the number of cis LD blocks, or the number and distance/pattern of trans clusters. The numeric data are in [`model_sweep_summary.csv`](model_sweep_summary.csv); [`gamma_recovery_summary.svg`](gamma_recovery_summary.svg) visualizes gamma correlation and scale recovery across cases.

| Case | Cis / trans SNPs | Gamma correlation | Gamma NRMSE | Gamma scale ratio | Radial Spearman | Held-out R² | Cis beta correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 15 / 50 | 0.9774 | 0.2126 | 0.9766 | -0.9834 | 0.9877 | 0.9824 |
| More cis LD blocks | 30 / 60 | 0.9879 | 0.1598 | 0.9580 | -0.9837 | 0.9849 | -0.9676 |
| More trans clusters | 20 / 80 | 0.9760 | 0.2184 | 0.9602 | -0.9999 | 0.9923 | 0.8391 |
| Asymmetric distance bands | 18 / 72 | 0.9894 | 0.1472 | 0.9841 | -1.0000 | 0.9934 | 0.7931 |

The gamma recovery metrics are strong in these deterministic synthetic tests. Interpret cis beta recovery cautiously: in the 30-cis-SNP case its correlation is negative despite strong held-out prediction and gamma metrics, indicating that prediction quality alone does not guarantee recovery of individual additive effects when predictors are correlated. Results are demonstrations on synthetic data, not independent empirical validation.

## Training design

The simulation harness uses mini-batch Adam optimization and staged regularization. Early epochs emphasize MSE and output-scale anchors; shape, radial amplitude/ranking, and factor-independence terms are added later. Held-out evaluation includes phenotype R², gamma correlation/scale, radial rank correlation, and cis-effect recovery.

## Requirements and data

- Python 3.10+
- PyTorch
- NumPy and Matplotlib for the synthetic simulation/plotting harness

This repository contains synthetic model code and summary metrics only; it includes no real genotype or RNA-seq data.