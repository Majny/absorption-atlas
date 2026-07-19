"""Model-agnostic loaders so the absorption pipeline runs on Gemma-2-2b OR -9b (+ their Gemma Scope
SAEs). 2b paths are byte-identical to the original library helpers (defaults preserved)."""
import re

import torch
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens import HookedTransformer

from sae_spelling.experiments.common import DEFAULT_DEVICE, SaeInfo, load_gemma2_model


def gemmascope_release(model_name: str) -> str:
    tag = "2b" if model_name.endswith("2b") else "9b"
    return f"gemma-scope-{tag}-pt-res"


def load_model(name: str = "google/gemma-2-2b"):
    if name.endswith("gemma-2-2b"):
        return load_gemma2_model()  # keep the exact tested 2b path
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    return HookedTransformer.from_pretrained(name, dtype=dtype, device=DEFAULT_DEVICE)


def load_sae(release: str, layer: int, width, l0, device: str = DEFAULT_DEVICE, dtype=None):
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    ws = f"{width // 1000}k" if isinstance(width, int) else width
    l0id = "canonical" if l0 == "canonical" else f"average_l0_{l0}"
    src = f"{release}-canonical" if l0 == "canonical" else release
    sae = SAE.from_pretrained(src, f"layer_{layer}/width_{ws}/{l0id}", device=device)[0].to(dtype=dtype)
    sae.fold_W_dec_norm()
    return sae


def saes_info(release: str, layer: int, width: int):
    d = get_pretrained_saes_directory()[release]
    out = []
    for name in d.saes_map:
        m = re.search(r"layer_(\d+)/width_(\d+)k/average_l0_(\d+)", name)
        if not m:
            continue
        L, w, l0 = int(m.group(1)), int(m.group(2)) * 1000, int(m.group(3))
        if L == layer and w == width:
            out.append(SaeInfo(l0, L, w, d.saes_map[name]))
    return out
