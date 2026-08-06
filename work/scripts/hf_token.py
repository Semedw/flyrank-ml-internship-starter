"""Resolve the Hugging Face READ token without ever printing it.

Token sources, in order:
1. the HF_TOKEN environment variable,
2. huggingface_hub's cached token (~/.cache/huggingface/token or ~/.huggingface/token,
   i.e. the file `huggingface-cli login` writes),
3. a getpass prompt when a terminal is available (Colab / interactive).

This module never writes the token to disk and never echoes it. Use it with
DuckDB's `hf` secret exactly like notebook 03 does.
"""

import getpass
import os
from pathlib import Path

from huggingface_hub import get_token

_LEGACY_TOKEN_FILES = [
    Path.home() / ".huggingface" / "token",          # huggingface-cli login
    Path.home() / ".cache" / "huggingface" / "token",
]


def _read_token_file() -> str | None:
    for p in _LEGACY_TOKEN_FILES:
        try:
            if p.is_file():
                t = p.read_text().strip()
                if t:
                    return t
        except OSError:
            continue
    return None


def resolve_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or get_token() or _read_token_file()
    if token:
        return token
    try:
        return getpass.getpass("Paste your Hugging Face READ token (hf_...): ")
    except (EOFError, KeyboardInterrupt):
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN (env), run `huggingface-cli login`, "
            "or create ~/.huggingface/token first."
        )