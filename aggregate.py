#!/usr/bin/env python
"""Builds the HANDOVER §7 tables from phase1_seed*.json and phase2_*.json (any folder layout below --inp).
Output: tables.md, tables.csv, summary.json. Seeds are aggregated as mean ± std."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]
COLS = ["bpw", "ppl_wt2", "ppl_pile", "kl_wt2", "kl_pile", "klp99_wt2", "klp99_pile", "top1_wt2", "top1_pile"] + \
       [f"acc_{t}" for t in TASKS] + ["acc_mean", "flip_c2i", "flip_i2c", "gold_exact", "gold_rougeL"]


def flat(entry):
    m = entry["metrics"]
    r = {"bpw": entry.get("bpw")}
    for s in ("wt2", "pile"):
        r[f"ppl_{s}"] = m[s]["ppl"]
        r[f"kl_{s}"] = m[s].get("kl_mean", 0.0)
        r[f"klp99_{s}"] = m[s].get("kl_p99", 0.0)
        r[f"top1_{s}"] = m[s].get("top1", 100.0)
    if "lmeval" in m:
        accs = [m["lmeval"][t]["acc"] for t in TASKS if t in m["lmeval"]]
        for t in TASKS:
            if t in m["lmeval"]:
                r[f"acc_{t}"] = m["lmeval"][t]["acc"]
        r["acc_mean"] = float(np.mean(accs))
    if "flips" in m:
        r["flip_c2i"], r["flip_i2c"] = m["flips"]["all"]["c2i"], m["flips"]["all"]["i2c"]
    if "golden" in m:
        r["gold_exact"], r["gold_rougeL"] = m["golden"]["exact"], m["golden"]["rougeL"]
    return r


def dense_row(d):
    r = {"bpw": 16.0, "kl_wt2": 0.0, "kl_pile": 0.0, "klp99_wt2": 0.0, "klp99_pile": 0.0, "top1_wt2": 100.0,
         "top1_pile": 100.0, "ppl_wt2": d["wt2"]["ppl"], "ppl_pile": d["pile"]["ppl"], "flip_c2i": 0.0,
         "flip_i2c": 0.0, "gold_exact": 100.0, "gold_rougeL": 1.0}
    if "lmeval" in d:
        for t in TASKS:
            if t in d["lmeval"]:
                r[f"acc_{t}"] = d["lmeval"][t]["acc"]
        r["acc_mean"] = float(np.mean([d["lmeval"][t]["acc"] for t in TASKS if t in d["lmeval"]]))
    return r


def fmt(vals, col):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    nd = 2 if col.startswith(("ppl", "top1", "flip", "gold_exact")) else (4 if col.startswith(("kl", "bpw")) else 3)
    if col.startswith("acc"):
        vals = [100 * v for v in vals]
        nd = 1
    mu = np.mean(vals)
    return f"{mu:.{nd}f}" + (f" ± {np.std(vals, ddof=1):.{nd}f}" if len(vals) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", default=["."])
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted({f for d in a.inp for f in glob.glob(os.path.join(d, "**", "phase*.json"), recursive=True)})
    groups, order, dense, infeasible = defaultdict(list), [], None, []
    for f in files:
        D = json.load(open(f))
        if dense is None and "dense" in D:
            dense = D["dense"]
        for key, e in D.get("models", {}).items():
            if e.get("infeasible"):
                infeasible.append(f"{e['method']} @ {e['target_bpw']} bpw (min {e['min_bpw']:.3f})")
                continue
            if not e.get("metrics"):
                continue
            g = key.split("#s")[0]
            if g not in groups:
                order.append(g)
            groups[g].append(flat(e))
    rows = [("dense (fp32)", [dense_row(dense)])] if dense else []
    rows += [(g, groups[g]) for g in sorted(order, key=lambda g: (np.mean([r["bpw"] for r in groups[g]]), g))]

    def table(cols, title):
        out = [f"### {title}", "", "| method | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for name, rs in rows:
            out.append(f"| {name} | {len(rs)} | " + " | ".join(fmt([r.get(c) for r in rs], c) for c in cols) + " |")
        return "\n".join(out) + "\n"

    md = ["# Qwen3-0.6B — quantization benchmark (HANDOVER protocol)", "",
          "Mean ± std over rotation seeds (n = number of seeds/runs). acc in %, KL in nats/token, top-1 in %, "
          "flips in % of questions vs dense. Rows sorted by effective bpw (196 block linears only).", "",
          table(["bpw", "ppl_wt2", "ppl_pile"], "Perplexity"),
          table(["bpw", "kl_wt2", "klp99_wt2", "top1_wt2", "kl_pile", "klp99_pile", "top1_pile"], "KL vs dense"),
          table(["bpw"] + [f"acc_{t}" for t in TASKS] + ["acc_mean"], "lm-eval 0-shot accuracy"),
          table(["bpw", "flip_c2i", "flip_i2c", "acc_mean"], "Flips (Dutta et al. 2024)"),
          table(["bpw", "gold_exact", "gold_rougeL"], "Golden set (proxy: greedy 128 tok vs dense)")]
    if infeasible:
        md += ["### Infeasible budgets", ""] + [f"- {x}" for x in infeasible]
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md))
    with open(os.path.join(a.out, "tables.csv"), "w") as f:
        f.write("method,n," + ",".join(COLS) + "\n")
        for name, rs in rows:
            f.write(f"\"{name}\",{len(rs)}," + ",".join(
                str(np.mean([r[c] for r in rs if r.get(c) is not None])) if any(r.get(c) is not None for r in rs)
                else "" for c in COLS) + "\n")
    json.dump({"files": files, "rows": {n: rs for n, rs in rows}, "infeasible": infeasible},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
