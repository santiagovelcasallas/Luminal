#!/usr/bin/env python
"""Phase 1: control, V1 (log scales, g128 and g64), V2 (ternary), V3 (V1 + ternary) for one rotation seed.

  --screen_only   picks (k, bz) for the V1 log scales on calibration data (seed-0 rotations) and exits
  --preflight     fast end-to-end check of every code path (minutes) before committing GPU hours
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qbench import bench as B
from qbench import hessian, mixed
from qbench import protocol as PR
from qbench import quant as Q
from qbench.hadamard import rotation_signs
from qbench.results import Results, env_info, make_logger

POINTS = [2.04, 2.3, 2.75, 3.0, 3.5, 3.9, 4.25]
SCREEN = [(6, 4), (6, 6), (8, 4), (8, 6)]


def get_args(argv=None):
    return build_parser().parse_args(argv)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PR.MP)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--points", default=",".join(map(str, POINTS)))
    ap.add_argument("--variants", default="control,V2,V1,V3,V1g64")
    ap.add_argument("--v1_kbz", default="auto", help="'auto' (read screening file) or e.g. '8,6'")
    ap.add_argument("--screen_only", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--no_lmeval", action="store_true")
    ap.add_argument("--no_golden", action="store_true")
    ap.add_argument("--lmeval_limit", type=int, default=1000)
    ap.add_argument("--lmeval_bs", type=int, default=16)
    ap.add_argument("--lmeval_tasks", default=",".join(B.LMEVAL_TASKS))
    ap.add_argument("--lmeval_include", default=None)
    ap.add_argument("--fake_data", action="store_true", help="synthetic text instead of Pile/wikitext/dolly (tests)")
    ap.add_argument("--max_layers", type=int, default=0, help="tests only: quantize the first N layers")
    return ap


# ---------------------------------------------------------------- setup shared by every phase
def setup(args, log):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    tok, dense = PR.load_model(args.model, args.device)
    names = PR.linear_names(dense.config.num_hidden_layers)
    if args.max_layers:
        names = names[:7 * args.max_layers]
    raw = None
    if args.fake_data:
        from tests.fake import fake_corpus
        raw = fake_corpus()
    cache = os.path.join(args.cache, "protocol_fake.pt" if args.fake_data else "protocol.pt")
    P = PR.load_protocol(tok, cache_path=cache, raw=raw)
    bench = B.Bench(dense, tok, names, args.device, lmeval_tasks=args.lmeval_tasks.split(","),
                    lmeval_include=args.lmeval_include)
    if args.fake_data:
        from tests.fake import fake_prompts
        prompts = fake_prompts(tok)
    elif args.no_golden:
        prompts = []
    else:
        prompts = B.golden_prompts(tok)
    cfg = {"lmeval": not args.no_lmeval, "golden": not args.no_golden, "lmeval_limit": args.lmeval_limit,
           "lmeval_bs": args.lmeval_bs}
    log(f"setup done: {len(names)} linears, device {args.device}, protocol {PR.fingerprint(P)}")
    return tok, dense, names, P, bench, prompts, cfg


def check_dense(d, fake, log):
    dev = {k: abs(d[k]["ppl"] - PR.DENSE_REF[k]) / PR.DENSE_REF[k] for k in PR.DENSE_REF}
    log(f"dense vs HANDOVER reference (wt2 27.15 / Pile 9.38): rel. dev {dev}")
    if fake:
        return dev
    if max(dev.values()) > 0.03:
        raise SystemExit(f"PROTOCOL MISMATCH: dense ppl {d['wt2']['ppl']:.3f}/{d['pile']['ppl']:.3f} "
                         f"vs 27.15/9.38. Stop and check versions/data before spending GPU time.")
    if max(dev.values()) > 0.005:
        log("WARNING: dense ppl deviates >0.5% from the HANDOVER reference")
    return dev


def get_dense(res, bench, P, prompts, cfg, args, log):
    if "dense" not in res.data:
        res.data["dense"] = B.dense_eval(bench, P, prompts, cfg, log)
        res.data["dense_check"] = check_dense(res.data["dense"], args.fake_data, log)
        res.save()
    return res.data["dense"]


def get_hessians(dense, P, names, args, log):
    t0 = time.time()
    H = hessian.collect_hessians(dense, P["calib"], names, args.device)
    log(f"hessians: {len(names)} ({time.time() - t0:.0f}s)")
    return H


def quantize_all(dense, H, names, opts, seed, device, log):
    mods = dict(dense.named_modules())
    store, t0 = {m: {} for m in names}, time.time()
    for j, m in enumerate(names):
        W = mods[m].weight.data.to(device)
        signs = rotation_signs(W.shape[1], seed, j)
        Wr, Hr = Q.rotate(W, H[m], signs)
        Hinv, dead = Q.prep_hinv(Hr)
        for opt in opts:
            store[m][opt.name] = Q.quantize_matrix(Wr, Hr, Hinv, dead, opt, signs)
        del Wr, Hr, Hinv
        if (j + 1) % 28 == 0 or j + 1 == len(names):
            log(f"  quantized {j + 1}/{len(names)} matrices x {len(opts)} options ({time.time() - t0:.0f}s)")
    return store


def bits_table(dense, names, opts):
    mods = dict(dense.named_modules())
    bits, N = {}, 0
    for m in names:
        r, n = mods[m].weight.shape
        N += r * n
        bits[m] = {o.name: Q.opt_bits(o, r, n) for o in opts}
    return bits, N


def weights_for(store, choice, device):
    return {m: store[m][o].weff(device) for m, o in choice.items()}


# ---------------------------------------------------------------- screening of (k, bz) for V1
def screen(args, log):
    tok, dense, names, P, bench, prompts, cfg = setup(args, log)
    H = get_hessians(dense, P, names, args, log)
    opts = [Q.Opt("q", b) for b in (2, 3)] + [Q.Opt("q", b, "log", k, bz, 128) for (k, bz) in SCREEN for b in (2, 3)]
    store = quantize_all(dense, H, names, opts, 0, args.device, log)
    bits, N = bits_table(dense, names, opts)
    L, bpw = {}, {}
    for o in opts:
        bench.load(weights_for(store, {m: o.name for m in names}, args.device))
        L[o.name] = bench.logppl(P["sens_ids"])
        bpw[o.name] = sum(bits[m][o.name] for m in names) / N
        log(f"  screen {o.name:22s} bpw {bpw[o.name]:.4f} calib log-ppl {L[o.name]:.5f}")
    lam = (L["q2"] - L["q3"]) / (bpw["q3"] - bpw["q2"])
    score = {}
    for key in [None] + SCREEN:
        tot = 0.0
        for b in (2, 3):
            name = f"q{b}" if key is None else Q.Opt("q", b, "log", key[0], key[1], 128).name
            tot += L[name] + lam * (bpw[name] - bpw[f"q{b}"])
        score["fp16" if key is None else f"k{key[0]}z{key[1]}"] = tot
    best = min(SCREEN, key=lambda kb: score[f"k{kb[0]}z{kb[1]}"])
    out = {"choice": list(best), "lambda_logppl_per_bpw": lam, "score_at_equal_bpw": score,
           "calib_logppl": L, "bpw": bpw, "env": env_info(), "protocol": PR.fingerprint(P)}
    with open(os.path.join(args.out, "v1_screening.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"screening: chose k={best[0]} bz={best[1]}; scores {score}")


# ---------------------------------------------------------------- main run for one seed
def run(args, log):
    if args.v1_kbz == "auto":
        with open(os.path.join(args.out, "v1_screening.json")) as f:
            k, bz = json.load(f)["choice"]
    else:
        k, bz = map(int, args.v1_kbz.split(","))
    points = [float(p) for p in args.points.split(",")]
    variants = args.variants.split(",")
    tok, dense, names, P, bench, prompts, cfg = setup(args, log)
    res = Results(os.path.join(args.out, f"phase1_seed{args.seed}.json"), args.resume_from)
    res.data["meta"] = {"seed": args.seed, "device": args.device, "v1_kbz": [k, bz], "points": points,
                        "variants": variants, "env": env_info(), "protocol": PR.fingerprint(P), "cfg": cfg,
                        "n_linears": len(names)}
    res.save()
    dense_ref = get_dense(res, bench, P, prompts, cfg, args, log)
    H = get_hessians(dense, P, names, args, log)

    F = [Q.Opt("q", 2), Q.Opt("q", 3), Q.Opt("q", 4), Q.Opt("tern")]
    L = [Q.Opt("q", b, "log", k, bz, 128) for b in (2, 3, 4)]
    L64 = [Q.Opt("q", b, "log", k, bz, 64) for b in (2, 3, 4)]
    TL = [Q.Opt("tern", 0, "log", k, bz, 128)]
    opts = list(F)
    if {"V1", "V3"} & set(variants):
        opts += L
    if "V1g64" in variants:
        opts += L64
    if "V3" in variants:
        opts += TL
    log(f"quantizing seed {args.seed}: options {[o.name for o in opts]}")
    store = quantize_all(dense, H, names, opts, args.seed, args.device, log)
    del H
    gc.collect()
    bits, N = bits_table(dense, names, opts)
    res.data["n_weights"] = N

    fam = {"V2": ([o.name for o in F], "q4"), "V1": ([o.name for o in L], L[2].name),
           "V1g64": ([o.name for o in L64], L64[2].name), "V3": ([o.name for o in TL + L], L[2].name)}
    sens = res.data.setdefault("sensitivity", {})
    for v in ("V2", "V1", "V1g64", "V3"):
        if v not in variants or v in sens:
            continue
        o_names, ref = fam[v]
        t0 = time.time()
        if v == "V3" and "V1" in sens:
            base, D = mixed.sensitivity(bench, store, names, [TL[0].name, ref], ref, P["sens_ids"], log)
            for m in names:
                D[m].update(sens["V1"]["delta"][m])
        else:
            base, D = mixed.sensitivity(bench, store, names, o_names, ref, P["sens_ids"], log)
        sens[v] = {"ref": ref, "base_logppl": base, "delta": D, "seconds": time.time() - t0}
        res.save()
        log(f"sensitivity {v}: {time.time() - t0:.0f}s")

    jobs = []
    if "control" in variants:
        for o in ("q2", "q3", "q4"):
            jobs.append(("control", o, {m: o for m in names}))
    for v in ("V2", "V1", "V3", "V1g64"):
        if v not in variants:
            continue
        o_names, _ = fam[v]
        for p in points:
            choice, tot = mixed.knapsack(names, o_names, bits, sens[v]["delta"], p * N)
            key = f"{v}@{p:.2f}#s{args.seed}"
            if choice is None:
                res.put_model(key, {"method": v, "target_bpw": p, "seed": args.seed, "infeasible": True,
                                    "min_bpw": tot / N, "metrics": None})
                log(f"{key}: infeasible (min {tot / N:.4f} bpw)")
                continue
            jobs.append((v, p, choice))

    t_start = time.time()
    for i, (v, p, choice) in enumerate(jobs):
        key = f"{v}@{p}#s{args.seed}" if v == "control" else f"{v}@{p:.2f}#s{args.seed}"
        if res.done(key):
            continue
        tot = sum(bits[m][choice[m]] for m in names)
        alloc = {}
        for o in choice.values():
            alloc[o] = alloc.get(o, 0) + 1
        log(f"[{i + 1}/{len(jobs)}] {key}: bpw {tot / N:.4f} alloc {alloc}")
        bench.load(weights_for(store, choice, args.device))
        metrics = B.full_eval(bench, P, dense_ref, prompts, cfg, log)
        res.put_model(key, {"method": v, "target_bpw": p, "seed": args.seed, "bpw": tot / N, "alloc": alloc,
                            "choice": choice, "metrics": metrics})
        el = time.time() - t_start
        log(f"    done; elapsed {el / 60:.1f} min, ~{el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min left")
    log("PHASE 1 DONE")


def preflight(args, log):
    """Every code path on a tiny budget: dense check, lm-eval/golden on few items, 2 layers quantized, all options,
    sensitivity + knapsack + one full eval."""
    t0 = time.time()
    tok, dense, names, P, bench, prompts, cfg = setup(args, log)
    cfg = dict(cfg, lmeval_limit=min(5, args.lmeval_limit))
    prompts = prompts[:4]
    d = B.dense_eval(bench, P, prompts, cfg, log)
    check_dense(d, args.fake_data, log)
    sub = names[:14]
    Hs = hessian.collect_hessians(dense, P["calib"][:2], sub, args.device)
    opts = [Q.Opt("q", 2), Q.Opt("q", 4), Q.Opt("tern"), Q.Opt("q", 2, "log", 8, 6, 128),
            Q.Opt("q", 4, "log", 8, 6, 128), Q.Opt("q", 2, "log", 8, 6, 64), Q.Opt("tern", 0, "log", 8, 6, 128)]
    store = quantize_all(dense, Hs, sub, opts, 0, args.device, log)
    bits, N = bits_table(dense, sub, opts)
    bench.reset()
    bench_sub = B.Bench(dense, tok, sub, args.device, lmeval_tasks=bench.lmeval_tasks, lmeval_include=bench.lmeval_include)
    del bench
    base, D = mixed.sensitivity(bench_sub, store, sub, ["q2", "q4", "tern"], "q4", P["sens_ids"], log)
    choice, tot = mixed.knapsack(sub, ["tern", "q2", "q4"], bits, D, 3.0 * N)
    log(f"knapsack ok: {tot / N:.3f} bpw")
    bench_sub.load(weights_for(store, choice, args.device))
    m = B.full_eval(bench_sub, P, d, prompts, cfg, log)
    assert m["wt2"]["ppl"] > 0 and m["wt2"]["kl_mean"] > 0
    log(f"PREFLIGHT OK ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = get_args()
    lg = make_logger(f"[s{a.seed}]")
    if a.preflight:
        preflight(a, lg)
    elif a.screen_only:
        screen(a, lg)
    else:
        run(a, lg)
