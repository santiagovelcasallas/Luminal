# Benchmark results — Qwen3-0.6B, HANDOVER protocol

Raw outputs of the quantization benchmark (`../qbench`, `../run_phase*.py`). Dense fp32 reference
reproduced the protocol exactly: wikitext-2 PPL 27.15, Pile PPL 9.38. All runs use 2 rotation seeds
(where rotation applies); metrics are PPL (wt2/Pile), KL vs dense, top-1 agreement, lm-eval 0-shot
accuracy, flips (Dutta et al. 2024) and a golden-set proxy.

## Contents

- `phase1/` — V1 / V2 / V3 / control (Hadamard + mixed precision at 2.04–4.25 bpw).
  - `summary.json` — per-model metrics, both seeds.
  - `tables.md` — aggregated tables (mean ± std).
- `phase2/` — rivals with their official pipelines, same calibration.
  - `phase2_llamacpp.json` — llama.cpp b11000 + imatrix, 9 formats. Reproduces the HANDOVER
    reference figures within ~1% (protocol control).
  - `phase2_gptq.json` — GPTQModel 4/3/2-bit g128.
- `phase3/` — ablations: uniform log (g128/64/32), fixed public recipe, HAWQ-style proxy, no rotation.
  - `tables.md` — aggregated tables for all four ablations vs llama.cpp.

## Headline (Pile / wikitext PPL, lower is better)

| bpw  | Luminal best | llama.cpp + imatrix |
|------|--------------|---------------------|
| ~2.0 | 23.8 / 84.7  | 109 / 453 (IQ2_XXS) |
| ~2.3 | 17.2 / 55.6  | 53.5 / 210.6 (IQ2_XS) |
| ~2.75| 13.0 / 41.4  | 20.6 / 82.0 (IQ2_M) |
| ~3.0 | 12.0 / 37.1  | 15.4 / 49.6 (IQ3_XXS) |

Below ~3 bpw Luminal leads on every metric, not only perplexity. See the phase tables for KL,
accuracy and flips.

## Provenance & gaps

These files were produced on Kaggle (T4 GPUs) by running the notebooks in `../notebooks`. Present
here: phase-1 `summary.json`, phase-2 rival JSON, phase-3 aggregated tables. **Not yet archived**:
the raw `phase1_seed{0,1}.json`, `phase3_seed{0,1}.json`, and the AWQ rows (AWQ was run separately in
an earlier benchmark the author holds). Re-running any notebook regenerates the full set.
