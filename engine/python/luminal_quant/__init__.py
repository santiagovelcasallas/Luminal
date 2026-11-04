# Luminal Quant — Copyright (C) 2026 Santiago Vélez Casallas. AGPL-3.0-only.
# See LICENSE at the repository root.
"""Luminal Quant: high-ratio post-training quantization (Hadamard rotation + mixed precision).

The numerical core is compiled Rust (``luminal_quant._native``); this package is the Python surface.
Public API is intentionally small in 0.1 — the Hadamard rotation primitive — and grows per release.
"""
from ._native import __version__, gptq_uniform, rotate, rotation_signs

__all__ = ["__version__", "gptq_uniform", "rotate", "rotation_signs"]
