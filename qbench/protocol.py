"""HANDOVER §2: data, calibration and evaluation protocol (identical for every method)."""
import hashlib
import os

import numpy as np
import pandas as pd
import torch

MP = "Qwen/Qwen3-0.6B"
SEQ = 512
EVALTOK = 20000
DENSE_REF = {"wt2": 27.15, "pile": 9.38}
LINEAR_SUFFIXES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def linear_names(n_layers):
    return [f"model.layers.{i}.{s}" for i in range(n_layers) for s in LINEAR_SUFFIXES]


def download_raw():
    from huggingface_hub import hf_hub_download
    pq = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                         repo_type="dataset")
    docs = [t for t in pd.read_parquet(pq)["text"].tolist() if isinstance(t, str) and t.strip()]
    wq = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                         repo_type="dataset")
    wt = [x for x in pd.read_parquet(wq)["text"].tolist() if isinstance(x, str) and x.strip()]
    return docs, wt


def build_protocol(tok, docs, wt):
    """Line-by-line transcription of HANDOVER §2 (same RNG calls in the same order)."""
    docs = list(docs)
    np.random.RandomState(0).shuffle(docs)
    pile_test = tok("\n\n".join(docs[:200]), return_tensors="pt").input_ids[0][:EVALTOK]
    sep = tok("\n\n").input_ids
    pile_train = torch.tensor([x for ids in tok(docs[200:]).input_ids for x in ids + sep], dtype=torch.long)
    wt_test = tok("\n\n".join(wt), return_tensors="pt").input_ids[0][:EVALTOK]

    def batch(n, L=SEQ):
        js = torch.randint(0, len(pile_train) - L - 1, (n,))
        return torch.stack([pile_train[j:j + L + 1] for j in js])

    torch.manual_seed(1234)
    calib = batch(32)[:, :-1]
    sens_ids = batch(8, 512).reshape(-1)
    calib_text = "\n".join(tok.decode(c) for c in calib)
    return dict(pile_test=pile_test, wt_test=wt_test, calib=calib, sens_ids=sens_ids, calib_text=calib_text)


def fingerprint(P):
    h = {}
    for k in ("pile_test", "wt_test", "calib", "sens_ids"):
        h[k] = hashlib.sha256(P[k].numpy().tobytes()).hexdigest()[:16]
    h["calib_text"] = hashlib.sha256(P["calib_text"].encode()).hexdigest()[:16]
    h["n_tokens"] = {k: int(P[k].numel()) for k in ("pile_test", "wt_test", "calib", "sens_ids")}
    return h


def load_protocol(tok, cache_path=None, raw=None):
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path)
    docs, wt = raw if raw is not None else download_raw()
    P = build_protocol(tok, docs, wt)
    if cache_path:
        torch.save(P, cache_path)
    return P


def load_model(path, device, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    model.requires_grad_(False)
    return tok, model


def model_snapshot(model_id=MP):
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
