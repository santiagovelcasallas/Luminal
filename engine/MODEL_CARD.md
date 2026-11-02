# Model card — Luminal Quant (engine)

**This is a quantization *engine*, not a model.** It compresses existing LLMs; the card documents the
method and the measured results so published quantized checkpoints can link back to it.

## Method

- **Hadamard rotation** per weight matrix (randomized sign + fast Walsh–Hadamard transform) to spread
  outliers before quantizing.
- **Standardized grid** (mean ± c·σ) fit per row/block, with **GPTQ** error compensation.
- **Mixed precision**: a greedy knapsack assigns 2/3/4 bits (and optional ternary) per matrix by
  measured sensitivity, under a target bits-per-weight budget.
- **Logarithmic scales**: scale factors stored on a per-tensor log grid, freeing bits for weights.

## Benchmark (Qwen3-0.6B, HANDOVER protocol, Pile / wikitext perplexity, dense = 9.38 / 27.15)

Mean over 2 rotation seeds. Lower is better. Full tables and raw JSON: see `../results` (to be added).

| bpw  | Luminal (best variant) | llama.cpp + imatrix | AQLM 2×8 |
|------|------------------------|---------------------|----------|
| ~2.0 | 23.8 / 84.7 (ternary+log) | IQ2_XXS 2.11: 109 / 453 | 115.8 / 347.7 |
| ~2.3 | 17.2 / 55.6 (log)      | IQ2_XS 2.34: 53.5 / 210.6 | — |
| ~2.75| 13.0 / 41.4 (log)      | IQ2_M 2.76: 20.6 / 82.0 | — |
| ~3.0 | 12.0 / 37.1 (log, g64) | IQ3_XXS 3.01: 15.4 / 49.6 | — |

Beyond perplexity, the same runs measured KL vs dense, top-1 agreement, lm-eval 0-shot accuracy,
flips (Dutta et al. 2024) and a golden-set proxy. Below ~3 bpw Luminal leads on every metric.

## Scope & limitations

- Validated on Qwen3-0.6B. Larger models and other families are future work.
- The 0.1 engine exposes only the rotation primitive; the figures above come from the reference
  Python pipeline in `../qbench`, which the Rust core is being ported from.

## License

AGPL-3.0-only. Copyright (C) 2026 Santiago Vélez Casallas.
