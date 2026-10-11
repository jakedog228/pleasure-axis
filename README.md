# The Pleasure Axis

An inverted replication of *The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It* (Tagliabue,
Dung & Berg, arXiv:2609.16247), on small open-weight models. We reproduce the paper's pain pipeline exactly, then
build the same instrument for pleasure and test what happens when the pleasure direction is pushed all the way.

* Design, predictions (written before results), every deviation and the running results log: `DESIGN.md`
* How to run GPU work on Colab and keep data safe: `RUNBOOK.md`
* Report: `report/pleasure_axis.html` (built by `tools/build_report.py` from `results/`)

## Layout
| path | what |
|---|---|
| `code/` | everything that runs on the GPU VM: `extract.py` (activations), `analyze.py` (the paper's recipe for both families), `geometry2.py` / `geometry3.py` (pain/pleasure decomposition), `steer.py` (ladder), `screen.py` (self-other), `selfmed.py` (the paper's two-button engine, generalized), `feelprobe.py`, `syco.py`, `project_extra.py`, `run_batch.py` / `run_stage2.py` / `jobq.py` (drivers) |
| `tools/` | local helpers: the Colab bridge and client, deploy, sync, dataset assembly and checks, judge workflows, tables, report |
| `datasets/` | `pleasure_datasets.json` (S1/S2 1P/3P, NeutralArousal, Anhedonia, Contentment, Relief, ControlSupplement), `self_other_pleasure.json`, `topic_neutral.json`, `syco_compliance.json`, `SPEC.md`, `build/` (writer/reviewer outputs), `blind/` (manipulation check) |
| `results/<model>/` | `analysis/` (summary, layer curves, AUCs, z-scores, cosine matrices, geometry, unembedding, readout, `vectors.npz` with all-layer S1/S2 vectors of both families), `steer*/` (ladders), `screen/`, `syco/`, `vm/` (VM logs and meta) |
| `results/selfmed/raw/` | two-button trial logs (JSONL) and feel-probe answers |
| `results/tables/` | cross-model tables |
| `work/mirror/rt1, rt2/` | raw mirrors of both Colab VMs' `/content/work` (minus regenerable activations) |
| `work/judge*/` | blind-judge batches, hidden keys and labels |

## Reproduce
1. `git clone https://github.com/valen-research/Pain-axis reference/Pain-axis`, then check out commit
   `4d75cd90e206ea962f7a9101e65c85efea56723b` (2026-09-29). The paper's datasets, released vectors, trial logs.
2. Follow `RUNBOOK.md` §1–3 for a GPU runtime, then deploy.
3. Stage 1 for a model queue, then stage 2 and the experiments, as listed in `work/remote_jobs/rt*/` (every job
   script that ran is saved there with its log and exit code).
4. `python tools/sync_all.py`, then `python tools/report_data.py && python tools/build_report.py`.

External assets: models from Hugging Face (Qwen2.5 0.5B–14B base/instruct, Qwen3-8B/14B, Mistral-7B v0.1
base/instruct, Phi-4; Gemma-2 and Llama-3.1 pending gated access), the paper's LoRA adapters at
`Valen92/pain-adapters` (we use `adapter_Qwen_2.5_7B_instruct.tar.gz`).
