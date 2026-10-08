"""Offline stand-ins for tests: synthetic corpus, a small BPE tokenizer with a Qwen-style chat template, a
random-weight Qwen3 with the real 0.6B matrix shapes (fewer layers), and a local multiple-choice lm-eval task."""
import json
import os

import numpy as np

CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def _words(rng, n_vocab=3000):
    letters = list("abcdefghijklmnopqrstuvwxyz")
    return ["".join(rng.choice(letters, size=rng.integers(2, 9))) for _ in range(n_vocab)]


def _doc(rng, words, n):
    p = 1.0 / np.arange(1, len(words) + 1)
    p /= p.sum()
    w = rng.choice(words, size=n, p=p)
    out, line = [], []
    for i, x in enumerate(w):
        line.append(x)
        if rng.random() < 0.08:
            out.append(" ".join(line).capitalize() + ".")
            line = []
    out.append(" ".join(line))
    return " ".join(out)


def fake_corpus(n_docs=1500, n_wt=2500, seed=0):
    rng = np.random.default_rng(seed)
    words = _words(rng)
    docs = [_doc(rng, words, int(rng.integers(40, 300))) for _ in range(n_docs)]
    wt = [_doc(rng, words, int(rng.integers(5, 60))) for _ in range(n_wt)]
    return docs, wt


def fake_prompts(tok, n=6):
    from qbench.bench import chat_prompt
    rng = np.random.default_rng(1)
    words = _words(rng)
    return [chat_prompt(tok, _doc(rng, words, 12)) for _ in range(n)]


def make_tokenizer(out_dir, vocab_size=4000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    docs, wt = fake_corpus()
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t.train_from_iterator(docs + wt, tr)
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  bos_token=None, unk_token=None)
    tok.chat_template = CHAT
    tok.save_pretrained(out_dir)
    return tok


def make_model(out_dir, layers=2, seed=0):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    os.makedirs(out_dir, exist_ok=True)
    tok = make_tokenizer(out_dir)
    vocab = ((len(tok) + 127) // 128) * 128
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=1024, intermediate_size=3072, num_hidden_layers=layers,
                      num_attention_heads=16, num_key_value_heads=8, head_dim=128, max_position_embeddings=40960,
                      rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True, attention_bias=False,
                      bos_token_id=tok.convert_tokens_to_ids("<|endoftext|>"),
                      eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                      pad_token_id=tok.convert_tokens_to_ids("<|endoftext|>"))
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():  # give norms some spread so AWQ/scales are non-trivial
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.3 * torch.randn_like(p))
    m.save_pretrained(out_dir)
    return out_dir


def make_lmeval_task(task_dir, n=40, seed=0):
    rng = np.random.default_rng(seed)
    words = _words(rng)
    os.makedirs(task_dir, exist_ok=True)
    data = os.path.abspath(os.path.join(task_dir, "fake_mc.jsonl"))
    with open(data, "w") as f:
        for i in range(n):
            f.write(json.dumps({"question": _doc(rng, words, 10), "choices": [_doc(rng, words, 3) for _ in range(4)],
                                "label": int(rng.integers(0, 4))}) + "\n")
    yaml = f"""task: fake_mc
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: multiple_choice
doc_to_text: "{{{{question}}}}"
doc_to_choice: "{{{{choices}}}}"
doc_to_target: "{{{{label}}}}"
metric_list:
  - metric: acc
  - metric: acc_norm
"""
    with open(os.path.join(task_dir, "fake_mc.yaml"), "w") as f:
        f.write(yaml)
    return task_dir


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fakeqwen"
    make_model(out, layers=int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    make_lmeval_task(out + "_tasks")
    print("ok", out)
