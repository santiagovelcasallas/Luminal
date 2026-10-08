import json
import os
import platform
import sys
import time


def now():
    return time.strftime("%H:%M:%S")


def make_logger(prefix=""):
    def log(*a):
        print(f"[{now()}]{prefix}", *a, flush=True)
    return log


class Results:
    """One JSON file per run, rewritten atomically after every finished model so a crash loses at most one model."""

    def __init__(self, path, resume_from=None):
        self.path = path
        self.data = {}
        for p in (resume_from, path):
            if p and os.path.exists(p):
                with open(p) as f:
                    self.data = json.load(f)
                break
        self.data.setdefault("models", {})

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def done(self, key):
        return key in self.data["models"] and "metrics" in self.data["models"][key]

    def put_model(self, key, entry):
        self.data["models"][key] = entry
        self.save()


def env_info():
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    for mod in ("transformers", "lm_eval", "numpy", "datasets", "gguf"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info
