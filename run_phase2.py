#!/usr/bin/env python
"""Phase 2: rivals with their official pipelines and the shared calibration, evaluated on the same bench.
  --part llamacpp   llama.cpp imatrix formats (needs a built llama.cpp at --llamacpp)
  --part gptq       GPTQModel 4/3/2 bits g128; desc_act and sym chosen on calibration data (not on eval data)
  --part awq        AutoAWQ 4 and 3 bits g128 (isolated site at --awq_site)
"""
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_phase1 as P1
from qbench import bench as B
from qbench import protocol as PR
from qbench.llamacpp import FORMATS, HANDOVER_REF, LlamaCpp
from qbench.results import Results, env_info, make_logger

HERE = os.path.dirname(os.path.abspath(__file__))


def extra_args(ap):
    ap.add_argument("--part", required=True, choices=["llamacpp", "gptq", "awq"])
    ap.add_argument("--llamacpp", default="/tmp/llama.cpp")
    ap.add_argument("--awq_site", default="/tmp/site_awq")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--gptq_bits", default="4,3,2")
    ap.add_argument("--awq_bits", default="4,3")
    ap.add_argument("--work", default="/tmp/qwork")


def evaluate(res, key, entry, W, bench, P, dense_ref, prompts, cfg, log):
    bench.load(W)
    log(f"{key}: bpw {entry['bpw']:.4f}")
    entry["metrics"] = B.full_eval(bench, P, dense_ref, prompts, cfg, log)
    res.put_model(key, entry)


def worker(script, args, gpu, extra_env=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    env.update(extra_env or {})
    cmd = [sys.executable, os.path.join(HERE, "qbench", script)] + args
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed ({r.returncode})")


def main():
    ap = P1.build_parser()
    extra_args(ap)
    args = ap.parse_args()
    log = make_logger(f"[{args.part}]")
    gpu = int(args.device.split(":")[1]) if ":" in args.device else 0
    tok, dense, names, P, bench, prompts, cfg = P1.setup(args, log)
    res = Results(os.path.join(args.out, f"phase2_{args.part}.json"), args.resume_from)
    res.data["meta"] = {"part": args.part, "env": env_info(), "protocol": PR.fingerprint(P), "cfg": cfg}
    res.save()
    dense_ref = P1.get_dense(res, bench, P, prompts, cfg, args, log)
    mods = dict(dense.named_modules())
    shapes = {n: tuple(mods[n].weight.shape) for n in names}
    N = sum(r * c for r, c in shapes.values())
    snap = PR.model_snapshot(args.model)
    os.makedirs(args.work, exist_ok=True)
    calib_pt = os.path.join(args.work, "calib.pt")
    torch.save(P["calib"].clone(), calib_pt)
    names_json = os.path.join(args.work, "names.json")
    json.dump(names, open(names_json, "w"))

    if args.part == "llamacpp":
        lc = LlamaCpp(args.llamacpp, os.path.join(args.work, "gguf"), log)
        bf16 = lc.convert(snap, dense, names)
        imat = lc.imatrix(bf16, P["calib_text"])
        res.data["llamacpp_commit"] = subprocess.run(["git", "-C", args.llamacpp, "describe", "--tags"],
                                                     capture_output=True, text=True).stdout.strip()
        for fmt in args.formats.split(","):
            key = f"llama.cpp {fmt}"
            if res.done(key):
                continue
            W, nbytes, types = lc.quantize(bf16, imat, fmt, names, shapes)
            entry = {"method": "llama.cpp", "format": fmt, "bpw": nbytes * 8 / N, "tensor_types": types,
                     "handover_ref": HANDOVER_REF.get(fmt)}
            evaluate(res, key, entry, W, bench, P, dense_ref, prompts, cfg, log)
            ref = HANDOVER_REF.get(fmt)
            if ref:
                m = res.data["models"][key]["metrics"]
                log(f"    control vs HANDOVER: bpw {entry['bpw']:.2f}/{ref[0]} pile {m['pile']['ppl']:.2f}/{ref[1]} "
                    f"wt2 {m['wt2']['ppl']:.2f}/{ref[2]}")

    elif args.part == "gptq":
        search = res.data.setdefault("gptq_search", {})
        for b in map(int, args.gptq_bits.split(",")):
            key = f"GPTQ {b}bit g128"
            if res.done(key):
                continue
            best = None
            for desc in (0, 1):
                for sym in (1, 0):
                    tag = f"{b}bit desc_act={desc} sym={sym}"
                    out = os.path.join(args.work, "gptq_w.pt")
                    t0 = time.time()
                    worker("gptq_worker.py", ["--model", snap, "--calib", calib_pt, "--names", names_json,
                                              "--bits", str(b), "--sym", str(sym), "--desc_act", str(desc),
                                              "--out", out, "--tmp", os.path.join(args.work, "gptq_tmp")], gpu)
                    d = torch.load(out)
                    bench.load(d["weights"])
                    lp = bench.logppl(P["sens_ids"])
                    search[tag] = {"calib_logppl": lp, "bpw": d["bytes"] * 8 / N, "seconds": time.time() - t0}
                    res.save()
                    log(f"  {tag}: calib log-ppl {lp:.5f} bpw {d['bytes'] * 8 / N:.4f}")
                    if best is None or lp < best[0]:
                        best = (lp, tag, d)
                    else:
                        del d
            lp, tag, d = best
            entry = {"method": "GPTQModel", "bits": b, "config": tag, "bpw": d["bytes"] * 8 / N,
                     "selected_on": "calibration log-ppl (sens_ids)"}
            evaluate(res, key, entry, d["weights"], bench, P, dense_ref, prompts, cfg, log)
            del best, d

    elif args.part == "awq":
        for b in map(int, args.awq_bits.split(",")):
            key = f"AWQ {b}bit g128"
            if res.done(key):
                continue
            out = os.path.join(args.work, "awq_w.pt")
            worker("awq_worker.py", ["--model", snap, "--calib", calib_pt, "--names", names_json, "--bits", str(b),
                                     "--out", out], gpu, {"PYTHONPATH": args.awq_site})
            d = torch.load(out)
            entry = {"method": "AutoAWQ", "bits": b, "bpw": d["bpw"], "folded_norms": len(d["folded_norms"])}
            evaluate(res, key, entry, d["weights"], bench, P, dense_ref, prompts, cfg, log)
    log(f"PHASE 2 {args.part} DONE")


if __name__ == "__main__":
    main()
