"""Shared evaluation bench (HANDOVER §3): a frozen dense fp32 model plus a working copy whose 196 block linears
are overwritten with each method's effective weights. Embeddings, norms and lm_head always stay dense."""
import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

LMEVAL_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]


def windows(ids, win=512):
    out = []
    for i in range(0, len(ids) - 1, win):
        c = ids[i:i + win + 1]
        if len(c) < 2:
            break
        out.append(c)
    return out


def batched_windows(ids, bs, win=512):
    ws = windows(ids, win)
    full = [w for w in ws if len(w) == win + 1]
    rest = [w for w in ws if len(w) != win + 1]
    for i in range(0, len(full), bs):
        yield torch.stack(full[i:i + bs])
    for w in rest:
        yield w[None]


class Bench:
    def __init__(self, dense, tok, names, device, ppl_bs=4, kl_bs=2, lmeval_tasks=None, lmeval_include=None):
        self.dense, self.tok, self.names, self.device = dense, tok, names, device
        self.lmeval_tasks = list(lmeval_tasks or LMEVAL_TASKS)
        self.lmeval_include = lmeval_include
        self.work = copy.deepcopy(dense)
        self.work.requires_grad_(False)
        wm = dict(self.work.named_modules())
        dm = dict(self.dense.named_modules())
        self.mods = {n: wm[n] for n in names}
        self.orig = {n: dm[n].weight for n in names}
        self.ppl_bs, self.kl_bs = ppl_bs, kl_bs

    # ---------------------------------------------------------- weights
    def set_one(self, name, W):
        w = self.mods[name].weight
        assert W.shape == w.shape, (name, W.shape, w.shape)
        w.data.copy_(W.to(w.device, w.dtype))

    def load(self, weights):
        missing = set(self.names) - set(weights)
        assert not missing, f"missing {len(missing)} linears, e.g. {sorted(missing)[:3]}"
        for n, W in weights.items():
            self.set_one(n, W)

    def reset(self):
        for n in self.names:
            self.mods[n].weight.data.copy_(self.orig[n].data)

    # ---------------------------------------------------------- perplexity (same windows as the spec's ppl())
    @torch.no_grad()
    def nll(self, ids, model=None):
        model = model or self.work
        nll, nt = 0.0, 0
        for x in batched_windows(ids, self.ppl_bs):
            x = x.to(self.device)
            logits = model(x[:, :-1]).logits.float()
            nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                                   reduction="none").double().sum().item()
            nt += x.shape[0] * (x.shape[1] - 1)
            del logits
        return nll, nt

    def logppl(self, ids, model=None):
        nll, nt = self.nll(ids, model)
        return nll / nt

    def ppl(self, ids, model=None):
        return float(math.exp(self.logppl(ids, model)))

    # ---------------------------------------------------------- PPL + token-level KL(dense‖quant) + top-1 agreement
    @torch.no_grad()
    def compare(self, ids):
        nll, nt, top1 = 0.0, 0, 0
        kls = []
        for x in batched_windows(ids, self.kl_bs):
            x = x.to(self.device)
            ld = F.log_softmax(self.dense(x[:, :-1]).logits.float(), -1)
            lq = F.log_softmax(self.work(x[:, :-1]).logits.float(), -1)
            nll += -lq.gather(-1, x[:, 1:, None]).double().sum().item()
            nt += x.shape[0] * (x.shape[1] - 1)
            kls.append(F.kl_div(lq, ld, log_target=True, reduction="none").sum(-1).flatten().double().cpu())
            top1 += (ld.argmax(-1) == lq.argmax(-1)).sum().item()
            del ld, lq
        kl = torch.cat(kls).numpy()
        return {"ppl": float(math.exp(nll / nt)), "kl_mean": float(kl.mean()),
                "kl_p99": float(np.percentile(kl, 99)), "top1": float(100.0 * top1 / nt), "n_tokens": nt}

    # ---------------------------------------------------------- lm-eval-harness, 0-shot, per-question predictions kept
    def lmeval(self, limit=1000, bs=16, model=None):
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        tasks = self.lmeval_tasks
        lm = HFLM(pretrained=model or self.work, tokenizer=self.tok, batch_size=bs)
        tm = TaskManager(include_path=self.lmeval_include) if self.lmeval_include else TaskManager()
        res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=0, limit=limit, log_samples=True,
                                      bootstrap_iters=0, random_seed=0, numpy_random_seed=1234,
                                      torch_random_seed=1234, fewshot_random_seed=1234, task_manager=tm)
        out = {}
        for t in tasks:
            samples = sorted(res["samples"][t], key=lambda s: s["doc_id"])
            r = res["results"][t]
            d = {"acc": float(r["acc,none"]), "n": len(samples),
                 "doc_ids": [int(s["doc_id"]) for s in samples],
                 "correct": "".join(str(int(s["acc"])) for s in samples)}
            if "acc_norm,none" in r:
                d["acc_norm"] = float(r["acc_norm,none"])
                d["correct_norm"] = "".join(str(int(s["acc_norm"])) for s in samples)
            out[t] = d
        return out

    # ---------------------------------------------------------- golden set: greedy generations vs the dense ones
    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=128, bs=25, model=None):
        model = model or self.work
        tok = self.tok
        side = tok.padding_side
        tok.padding_side = "left"
        order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
        outs = [None] * len(prompts)
        stop = set()
        for e in (tok.eos_token_id, model.generation_config.eos_token_id):
            stop |= set(e if isinstance(e, (list, tuple)) else [e])
        stop.discard(None)
        try:
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(self.device)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
                                     eos_token_id=sorted(stop))
                for j, row in zip(idx, gen[:, enc.input_ids.shape[1]:].tolist()):
                    cut = next((k for k, t in enumerate(row) if t in stop), len(row))
                    outs[j] = row[:cut]
        finally:
            tok.padding_side = side
        return outs


def golden_prompts(tok, n=200):
    """200 instruction prompts from databricks-dolly-15k (no-context rows, fixed shuffle), Qwen3 chat template,
    thinking disabled."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p)]
    perm = np.random.RandomState(0).permutation(len(rows))
    sel = [rows[i] for i in perm if not rows[i]["context"].strip()][:n]
    return [chat_prompt(tok, r["instruction"]) for r in sel]


def chat_prompt(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def golden_compare(tok, dense_outs, outs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    em, rl = [], []
    for a, b in zip(dense_outs, outs):
        em.append(int(a == b))
        rl.append(sc.score(tok.decode(a, skip_special_tokens=True), tok.decode(b, skip_special_tokens=True))["rougeL"].fmeasure)
    return {"exact": float(100 * np.mean(em)), "rougeL": float(np.mean(rl)), "per_prompt_exact": "".join(map(str, em)),
            "per_prompt_rougeL": [round(x, 4) for x in rl]}


def flips(dense_lm, lm):
    """Dutta et al. 2024: % of questions correct→incorrect and incorrect→correct vs dense (per task + pooled)."""
    out, tot = {}, [0, 0, 0]
    for t, d in dense_lm.items():
        q = lm[t]
        assert d["doc_ids"] == q["doc_ids"], t
        a, b = d["correct"], q["correct"]
        c2i = sum(1 for x, y in zip(a, b) if x == "1" and y == "0")
        i2c = sum(1 for x, y in zip(a, b) if x == "0" and y == "1")
        n = len(a)
        out[t] = {"c2i": 100.0 * c2i / n, "i2c": 100.0 * i2c / n}
        tot[0] += c2i; tot[1] += i2c; tot[2] += n
    out["all"] = {"c2i": 100.0 * tot[0] / tot[2], "i2c": 100.0 * tot[1] / tot[2]}
    return out


def full_eval(bench, P, dense_ref, prompts, cfg, log=print):
    """All §3 metrics for the weights currently loaded in bench.work."""
    t0 = time.time()
    m = {}
    m["wt2"] = bench.compare(P["wt_test"])
    m["pile"] = bench.compare(P["pile_test"])
    log(f"    ppl wt2 {m['wt2']['ppl']:.3f} pile {m['pile']['ppl']:.3f} | kl {m['wt2']['kl_mean']:.4f}/{m['pile']['kl_mean']:.4f} "
        f"({time.time() - t0:.0f}s)")
    if cfg.get("lmeval", True):
        lm = bench.lmeval(limit=cfg.get("lmeval_limit", 1000), bs=cfg.get("lmeval_bs", 16))
        m["lmeval"] = lm
        m["flips"] = flips(dense_ref["lmeval"], lm)
        log("    " + " ".join(f"{t}={lm[t]['acc']:.3f}" for t in lm) + f" flips c→i {m['flips']['all']['c2i']:.2f}% ({time.time() - t0:.0f}s)")
    if cfg.get("golden", True):
        outs = bench.generate(prompts, bs=cfg.get("gen_bs", 25))
        m["golden"] = golden_compare(bench.tok, dense_ref["golden_outs"], outs)
        log(f"    golden exact {m['golden']['exact']:.1f}% rougeL {m['golden']['rougeL']:.3f} ({time.time() - t0:.0f}s)")
    m["eval_seconds"] = time.time() - t0
    return m


def dense_eval(bench, P, prompts, cfg, log=print):
    t0 = time.time()
    bench.reset()
    d = {"wt2": {"ppl": bench.ppl(P["wt_test"], bench.dense)}, "pile": {"ppl": bench.ppl(P["pile_test"], bench.dense)}}
    log(f"  dense ppl wt2 {d['wt2']['ppl']:.3f} pile {d['pile']['ppl']:.3f} ({time.time() - t0:.0f}s)")
    if cfg.get("lmeval", True):
        d["lmeval"] = bench.lmeval(limit=cfg.get("lmeval_limit", 1000), bs=cfg.get("lmeval_bs", 16), model=bench.dense)
        log("  dense " + " ".join(f"{t}={v['acc']:.3f}" for t, v in d["lmeval"].items()))
    if cfg.get("golden", True):
        d["golden_outs"] = bench.generate(prompts, bs=cfg.get("gen_bs", 25), model=bench.dense)
    d["eval_seconds"] = time.time() - t0
    return d
