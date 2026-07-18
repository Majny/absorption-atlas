#!/bin/bash
# One-shot environment setup on the MetaCentrum frontend (skirit).
# Self-contained micromamba env — no reliance on the module system.
# Token is read from $WORK/.hf_token (created out-of-band, never committed).
set -uo pipefail
export KRB5CCNAME=FILE:/tmp/krb5cc_$(id -u)
WORK=/storage/brno2/home/${USER}/absorption-atlas
export HF_HOME=$WORK/hf_cache
export MAMBA_ROOT_PREFIX=$WORK/mamba
[ -f "$WORK/.hf_token" ] && source "$WORK/.hf_token"
mkdir -p "$WORK/bin" "$WORK/results" "$HF_HOME"
cd "$WORK"

echo "=== [1/6] micromamba ==="
if [ ! -x "$WORK/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$WORK" bin/micromamba
fi
"$WORK/bin/micromamba" --version
eval "$("$WORK/bin/micromamba" shell hook -s bash)"

echo "=== [2/6] env python 3.11 ==="
if [ ! -d "$WORK/mamba/envs/abs" ]; then
  micromamba create -y -n abs python=3.11 -c conda-forge
fi
micromamba activate abs
python --version

echo "=== [3/6] clone sae-spelling ==="
if [ ! -d "$WORK/sae-spelling" ]; then
  git clone --depth 1 https://github.com/lasr-spelling/sae-spelling.git "$WORK/sae-spelling"
fi

echo "=== [4/6] pip install -e sae-spelling (CUDA torch) ==="
cd "$WORK/sae-spelling"
pip install -q -e . 2>&1 | tail -8
python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"

echo "=== [5/6] pre-download gemma-2-2b to HF cache ==="
python - <<'PY'
import os
from huggingface_hub import login, snapshot_download
tok = os.environ.get("HF_TOKEN")
if tok:
    login(token=tok)
p = snapshot_download("google/gemma-2-2b",
                      allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"])
print("model cached at", p)
PY

echo "=== [6/6] GPU queues ==="
qstat -Q 2>&1 | awk 'NR<=2 || tolower($1) ~ /gpu/' | head -20

echo SETUP_DONE
