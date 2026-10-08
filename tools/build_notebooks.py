"""Generates self-contained Kaggle notebooks (all code embedded with %%writefile) from the repo sources."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = ["qbench/__init__.py", "qbench/protocol.py", "qbench/hadamard.py", "qbench/quant.py", "qbench/hessian.py",
       "qbench/mixed.py", "qbench/bench.py", "qbench/results.py", "qbench/llamacpp.py", "qbench/gptq_worker.py",
       "qbench/awq_worker.py", "run_phase1.py", "run_phase2.py", "aggregate.py"]

PINS = "transformers==5.19.0 lm-eval==0.4.13 rouge-score==0.1.2 gguf==0.19.0 sentencepiece"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(True)}


def code_cells():
    cells = [code("import os\nos.makedirs('/kaggle/working/qb/qbench', exist_ok=True)\n%cd /kaggle/working/qb")]
    for f in LIB:
        src = open(os.path.join(ROOT, f)).read()
        cells.append(code(f"%%writefile {f}\n{src}"))
    return cells


LAUNCH = r'''
import subprocess, time, sys, torch
def launch(jobs):
    """jobs: list of (name, shell command). Runs them in parallel, prints log tails, fails loudly."""
    procs = {n: subprocess.Popen(c, shell=True, stdout=open(f"/kaggle/working/logs/{n}.log", "w"),
                                 stderr=subprocess.STDOUT) for n, c in jobs}
    last = 0
    while any(p.poll() is None for p in procs.values()):
        time.sleep(30)
        if time.time() - last > 600:
            last = time.time()
            for n in procs:
                tail = open(f"/kaggle/working/logs/{n}.log").read().strip().splitlines()[-3:]
                print(f"--- {n}: " + " | ".join(tail), flush=True)
    bad = [n for n, p in procs.items() if p.returncode != 0]
    for n in procs:
        print(f"===== {n} (rc={procs[n].returncode}) last lines =====")
        print("\n".join(open(f"/kaggle/working/logs/{n}.log").read().splitlines()[-15:]))
    if bad:
        raise RuntimeError(f"failed: {bad} (see /kaggle/working/logs)")

def run(cmd):
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")
os.makedirs("/kaggle/working/logs", exist_ok=True)
NGPU = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, "Activa el acelerador GPU (T4 x2) en Settings"
'''

COMMON = "--out /kaggle/working/out --cache /kaggle/working/cache"

INTRO1 = """
# Fase 1 — V1 / V2 / V3 / control (Qwen3-0.6B, protocolo del HANDOVER)

**Cómo correrlo en Kaggle (sin dejar el computador prendido):**
1. *File → Import Notebook* y sube este `.ipynb`.
2. Panel derecho → *Settings*: **Accelerator = GPU T4 x2**, **Internet = On** (requiere teléfono verificado).
3. Arriba a la derecha: **Save Version → "Save & Run All (Commit)"**. Ya puedes cerrar el navegador: corre en segundo plano.
4. Al terminar, en la versión guardada → pestaña *Output*: `out/phase1_seed0.json`, `out/phase1_seed1.json`,
   `out/v1_screening.json`, `tables/tables.md` y `logs/`.

**Qué hace:** preflight rápido (falla en minutos si algo está mal, antes de gastar horas) → verifica la densa
(wt2 ≈ 27.15, Pile ≈ 9.38; aborta si se desvía >3 %) → elige (k, bz) de las escalas log con calibración →
corre las semillas 0 y 1 en paralelo (una por GPU): Hessianos, rotación Hadamard, GPTQ con todas las opciones,
sensibilidad, mochila en 2.04…4.25 bpw, y evalúa cada modelo (PPL, KL, top-1, lm-eval 0-shot ×1000, flips, set dorado).

**Tiempo estimado con T4 x2:** ~4–5 h (no supera el límite de 12 h por sesión). Con una sola GPU las semillas corren
en serie (~8–9 h): en ese caso pon `SEEDS = [0]` y corre la semilla 1 en otra sesión.
Si algo se corta, los JSON guardan cada modelo terminado: adjunta la salida anterior como *Input* y pon su ruta en
`RESUME_DIR` para continuar donde quedó.
"""

INTRO2 = """
# Fase 2 — Rivales con su pipeline oficial (misma calibración, mismo banco)

**Kaggle:** igual que la Fase 1 (GPU T4 x2, Internet On, *Save & Run All (Commit)*). Es independiente de la Fase 1:
pueden correr en sesiones distintas.

- **llama.cpp b11000 + imatrix** (GPU 0): IQ2_XXS, IQ2_XS, IQ2_M, Q2_K, IQ3_XXS, IQ3_M, Q3_K_M, IQ4_XS, Q4_K_M.
  GGUF bf16 con verificación de ida y vuelta exacta, imatrix con el mismo texto de calibración, descuantización con
  `llama-quantize --allow-requantize … F32`, bpw = bytes reales de las 196 lineales. Sirve de **control del protocolo**:
  el log compara cada formato con las cifras del HANDOVER (deben coincidir en ~2 %).
- **GPTQModel 7.5.0** (GPU 1): 4/3/2 bits g128; `desc_act` y `sym` se eligen con la calibración (nunca con evaluación).
- **AutoAWQ 0.2.9** (GPU 1, entorno aislado con transformers 4.51.3): 4 y 3 bits g128.

AQLM 2x8 no se incluye (≈3 h solo de cuantización en T4; el HANDOVER ya trae su cifra con este protocolo).
EXL3/QuIP# no son viables en T4.

**Tiempo estimado con T4 x2:** ~3–4 h. Salida: `out/phase2_*.json`, `tables/`, `logs/`.
Para la tabla conjunta: adjunta la salida de la Fase 1 como *Input* (queda en `/kaggle/input/...`) y la última celda la incluye.
"""


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def phase1():
    cells = [md(INTRO1),
             code("SEEDS = [0, 1]          # rotaciones aleatorias (mínimo 2 según el HANDOVER)\n"
                  "RESUME_DIR = None       # p.ej. '/kaggle/input/<salida-anterior>/out' para continuar\n"
                  "LMEVAL_LIMIT = 1000"),
             code(f"!pip install -q {PINS}")] + code_cells()
    cells += [code(LAUNCH),
              code(f"run('python run_phase1.py --preflight --device cuda:0 {COMMON}')"),
              code("import os\n"
                   f"if not os.path.exists('/kaggle/working/out/v1_screening.json'):\n"
                   f"    run('python run_phase1.py --screen_only --device cuda:0 {COMMON}')\n"
                   "print(open('/kaggle/working/out/v1_screening.json').read()[:600])"),
              code("def seed_cmd(s, gpu):\n"
                   "    r = f' --resume_from {RESUME_DIR}/phase1_seed{s}.json' if RESUME_DIR else ''\n"
                   f"    return f'python run_phase1.py --seed {{s}} --device cuda:{{gpu}} --lmeval_limit {{LMEVAL_LIMIT}} {COMMON}' + r\n"
                   "if NGPU >= 2:\n"
                   "    launch([(f'seed{s}', seed_cmd(s, i)) for i, s in enumerate(SEEDS)])\n"
                   "else:\n"
                   "    launch([('seeds', ' && '.join(seed_cmd(s, 0) for s in SEEDS))])"),
              code("run('python aggregate.py --inp /kaggle/working/out --out /kaggle/working/tables')")]
    return nb(cells)


def phase2():
    cells = [md(INTRO2),
             code("RESUME_DIR = None\nLMEVAL_LIMIT = 1000"),
             code(f"!pip install -q {PINS}\n"
                  "!pip install -q --no-deps gptqmodel==7.5.0\n"
                  "!pip install -q device-smi pypcre tokenicer logbar defuser torchao threadpoolctl\n"
                  "# AutoAWQ needs transformers 4.5x: isolated site, used only by the AWQ subprocess\n"
                  "!pip install -q --no-deps --target /tmp/site_awq autoawq==0.2.9 transformers==4.51.3 "
                  "\"tokenizers>=0.21,<0.22\" \"huggingface_hub>=0.30,<1.0\""),
             code("%%bash\nset -e\ncd /tmp\n[ -d llama.cpp ] || git clone -q --depth 1 --branch b11000 "
                  "https://github.com/ggml-org/llama.cpp\ncd llama.cpp\n"
                  "cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF "
                  "-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > /tmp/lc_cmake.log\n"
                  "cmake --build build -j$(nproc) --target llama-quantize llama-imatrix > /tmp/lc_build.log\n"
                  "ls build/bin | grep llama-")] + code_cells()
    cells += [code(LAUNCH),
              code("def part(p, gpu):\n"
                   "    r = f' --resume_from {RESUME_DIR}/phase2_{p}.json' if RESUME_DIR else ''\n"
                   f"    return f'python run_phase2.py --part {{p}} --device cuda:{{gpu}} --lmeval_limit {{LMEVAL_LIMIT}} {COMMON}' + r\n"
                   "g1 = 1 if NGPU >= 2 else 0\n"
                   "jobs = [('llamacpp', part('llamacpp', 0)), ('gptq_awq', part('gptq', g1) + ' ; ' + part('awq', g1))]\n"
                   "if NGPU >= 2:\n"
                   "    launch(jobs)\n"
                   "else:\n"
                   "    launch([('all', ' ; '.join(c for _, c in jobs))])"),
              code("run('python aggregate.py --inp /kaggle/working/out /kaggle/input --out /kaggle/working/tables')")]
    return nb(cells)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for name, fn in (("fase1_V1_V2_kaggle.ipynb", phase1), ("fase2_rivales_kaggle.ipynb", phase2)):
        with open(os.path.join(ROOT, "notebooks", name), "w") as f:
            json.dump(fn(), f, indent=1, ensure_ascii=False)
        print("wrote", name)
