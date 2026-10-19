#!/usr/bin/env python
"""Phase 3 ablations (same protocol, log scales with the phase-1 (k, bz)):
  uniform  Hadamard + log scales, every matrix q2/q3/q4, groups 128/64/32
  recipe   Hadamard + log scales, fixed public recipe: base b, +1 bit attn_v/ffn_down, +1 bit first/last 2 layers
  proxy    Hadamard + log scales, knapsack on a HAWQ-style proxy (tr(H)/n · ||W_eff − W||²), no ppl measured
  norot    no rotation, V1 pipeline (measured sensitivity + knapsack); seed 0 only (deterministic)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_phase1 as P1
from qbench import bench as B
from qbench import mixed
from qbench import protocol as PR
from qbench import quant as Q
from qbench.hadamard import rotation_signs
from qbench.results import Results, env_info, make_logger

POINTS = [2.3, 2.75, 3.0, 3.5]


def prep(dense, H, names, seed, device, rot):
    mods = dict(dense.named_modules())
    for j, m in enumerate(names):
        W = mods[m].weight.data.to(device)
        if rot:
            s = rotation_signs(W.shape[1], seed, j)
            Wr, Hr = Q.rotate(W, H[m], s)
        else:
            s, Wr, Hr = None, W.double(), H[m].to(device).double()
        Hinv, dead = Q.prep_hinv(Hr)
        yield m, Wr, Hr, Hinv, dead, s


def bits_of(opt, shape, rot):
    r, n = shape
    return Q.opt_bits(opt, r, n) - (0 if rot else n)


def main():
    ap = P1.build_parser()
    ap.add_argument("--parts", default="uniform,recipe,proxy,norot")
    args = ap.parse_args()
    log = make_logger(f"[p3 s{args.seed}]")
    if args.v1_kbz == "auto":
        k, bz = json.load(open(os.path.join(args.out, "v1_screening.json")))["choice"]
    else:
        k, bz = map(int, args.v1_kbz.split(","))
    tok, dense, names, P, bench, prompts, cfg = P1.setup(args, log)
    res = Results(os.path.join(args.out, f"phase3_seed{args.seed}.json"), args.resume_from)
    res.data["meta"] = {"seed": args.seed, "v1_kbz": [k, bz], "env": env_info(), "protocol": PR.fingerprint(P)}
    res.save()
    dref = P1.get_dense(res, bench, P, prompts, cfg, args, log)
    H = P1.get_hessians(dense, P, names, args, log)
    mods = dict(dense.named_modules())
    shape = {m: tuple(mods[m].weight.shape) for m in names}
    N = sum(r * c for r, c in shape.values())
    nl = dense.config.num_hidden_layers
    parts = args.parts.split(",")
    s = args.seed

    def run_choice(key, choice, rot, extra):
        if res.done(key):
            return
        t0 = time.time()
        for m, Wr, Hr, Hinv, dead, sg in prep(dense, H, names, s, args.device, rot):
            bench.set_one(m, Q.quantize_matrix(Wr, Hr, Hinv, dead, choice[m], sg).weff(args.device))
        bpw = sum(bits_of(choice[m], shape[m], rot) for m in names) / N
        log(f"{key}: bpw {bpw:.4f} (quantized in {time.time() - t0:.0f}s)")
        res.put_model(key, dict(extra, seed=s, bpw=bpw, metrics=B.full_eval(bench, P, dref, prompts, cfg, log)))

    def run_store(key, store, choice, rot, extra):
        if res.done(key):
            return
        bench.load({m: store[m][o].weff(args.device) for m, o in choice.items()})
        bpw = sum(bits_of(store[m][o].opt, shape[m], rot) for m, o in choice.items()) / N
        alloc = {}
        for o in choice.values():
            alloc[o] = alloc.get(o, 0) + 1
        log(f"{key}: bpw {bpw:.4f} alloc {alloc}")
        res.put_model(key, dict(extra, seed=s, bpw=bpw, alloc=alloc, choice=choice,
                                metrics=B.full_eval(bench, P, dref, prompts, cfg, log)))

    if "uniform" in parts:
        for g in (128, 64, 32):
            for b in (2, 3, 4):
                o = Q.Opt("q", b, "log", k, bz, g)
                run_choice(f"U-log q{b} g{g}#s{s}", {m: o for m in names}, True, {"method": "uniform-log"})

    if "recipe" in parts:
        for b in (2, 3):
            for g in (128, 64, 32):
                ch = {}
                for m in names:
                    i = int(m.split(".")[2])
                    e = int(m.endswith(("v_proj", "down_proj"))) + int(i < 2 or i >= nl - 2)
                    ch[m] = Q.Opt("q", b + e, "log", k, bz, g)
                run_choice(f"Recipe b{b} g{g}#s{s}", ch, True, {"method": "fixed-recipe"})

    L = [Q.Opt("q", b, "log", k, bz, 128) for b in (2, 3, 4)]
    on = [o.name for o in L]

    def mixed_runs(tag, store, D, rot):
        bits = {m: {o.name: bits_of(o, shape[m], rot) for o in L} for m in names}
        for p in POINTS:
            choice, tot = mixed.knapsack(names, on, bits, D, p * N)
            key = f"{tag}@{p:.2f}#s{s}"
            if choice is None:
                res.put_model(key, {"method": tag, "infeasible": True, "target_bpw": p, "min_bpw": tot / N,
                                    "metrics": None})
                continue
            run_store(key, store, choice, rot, {"method": tag, "target_bpw": p})

    if "proxy" in parts:
        store = {m: {o.name: Q.quantize_matrix(Wr, Hr, Hinv, dead, o, sg) for o in L}
                 for m, Wr, Hr, Hinv, dead, sg in prep(dense, H, names, s, args.device, True)}
        D = {}
        for m in names:
            W = mods[m].weight.data.to(args.device).float()
            tr = float(torch.trace(H[m])) / H[m].shape[0]
            D[m] = {o: tr * float(((store[m][o].weff(args.device) - W) ** 2).sum()) for o in on}
        res.data.setdefault("proxy_delta", D)
        mixed_runs("Proxy", store, D, True)
        del store

    if "norot" in parts and s == 0:
        store = {m: {o.name: Q.quantize_matrix(Wr, Hr, Hinv, dead, o, sg) for o in L}
                 for m, Wr, Hr, Hinv, dead, sg in prep(dense, H, names, s, args.device, False)}
        if "norot_sens" not in res.data:
            _, D = mixed.sensitivity(bench, store, names, on, on[2], P["sens_ids"], log)
            res.data["norot_sens"] = D
            res.save()
        mixed_runs("NoRot", store, res.data["norot_sens"], False)
    log("PHASE 3 DONE")


if __name__ == "__main__":
    main()
