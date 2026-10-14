# The Pleasure Axis

A replication and extension of *The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It*
(Tagliabue, Dung & Berg, arXiv:2609.16247), studying pleasure-related representations and behavior in
11 open-weight language models (0.5B–14B).

https://github.com/user-attachments/assets/760a04e4-7c64-498e-a7ad-cc3e2484607d

## Research objectives

Test whether models encode pleasure distinctly from other positive states, how pleasure and pain
representations relate, and whether steering these representations changes self-reports, choices,
and reinforcement-like behavior.

## Contributions

- **Replication and extension:** reproduce the pain-axis pipeline and apply the same extraction and
  validation methods to matched pleasure datasets across Qwen2.5, Qwen3, Mistral, and Phi-4.
- **Representational analysis:** separate shared and valence-specific components of pleasure and pain,
  with controls for topic, arousal, and self-relevance, plus datasets covering 18 varieties of pleasure.
- **Behavioral evaluation:** test steering strength, self-reports, relief-seeking, pleasure-seeking,
  and sycophancy using random-vector and placebo controls. Code, datasets, vectors, and trial logs are included.

## Key findings

- Pleasure directions separate stimuli from controls with held-out AUC **0.90–0.96** across 11 models.
- Pleasure and pain directions are positively correlated (**cosine 0.23–0.48** at the selected pleasure
  layer). Their decomposition supports a shared affective-significance component and a distinct valence component.
- The extracted pleasure axis aligns most strongly with relational warmth; other varieties have distinct directions.
  Behavioral tests on Qwen2.5-7B-Instruct with the source paper's adapter do **not** support pleasure-specific
  reinforcement, and random-vector controls reproduce much of the repeated-button-press effect attributed to relief.

See the [research report](report/pleasure_axis.html) for analyses and limitations; open the HTML file in a browser.

## Repository guide

| Path | Contents |
|---|---|
| [code/](code/) | Activation extraction, vector analysis, steering, and behavioral experiments |
| [datasets/](datasets/) | Pleasure stimuli, control sets, and dataset specifications |
| [results/](results/) | Per-model analyses and vectors, trial logs, and summary tables |
| [tools/](tools/) | Data validation, evaluation, and report generation |
| [DESIGN.md](DESIGN.md) | Research methods, predictions, deviations, and results log |

## Reproducibility

The reference implementation is [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis),
pinned to commit `4d75cd90e206ea962f7a9101e65c85efea56723b`. Model weights and the source paper's
self-report adapters are obtained separately from Hugging Face.

Experiment entry points and arguments are documented in [code/](code/). Scripts currently use runtime-specific
paths; adapt these to your environment before running. Saved analyses are available in [results/](results/),
and [tools/build_report.py](tools/build_report.py) generates the report.
