This repository contains autoregressive next-token prediction (`LM`) and reinforcement-learning (`RL`) experiments for the paper.

## Computation Axes

The main computation axes in RL and LM are:

| Name | CLI argument | Meaning |
|---|---|---|
| Within-step depth | `--depth` | Number of sequential expert transformations per environment step. |
| Number of experts | `--num_experts` | Number of experts evaluated in parallel at each depth level. |
| Expert width | `--expert_hidden_dim` | Hidden dimension of each expert. |


