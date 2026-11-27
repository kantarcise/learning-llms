#!/usr/bin/env bash
# Run from a repository checkout; keep all downloaded and generated files local.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run --locked --group experiments --group book python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name())'
cache="$PWD/artifacts/chapter-07/inputs"
mkdir -p "$cache"
fetch() {
    local url="$1" name="$2" digest="$3" tmp
    if [[ -f "$cache/$name" ]] && echo "$digest  $cache/$name" | sha256sum --check --status; then
        return
    fi
    tmp=$(mktemp "$cache/$name.XXXXXX")
    curl --fail --location --silent --show-error --retry 3 "$url" -o "$tmp"
    echo "$digest  $tmp" | sha256sum --check
    mv "$tmp" "$cache/$name"
}
book=https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/f03d8e8feab1afdd9ac303e3e5c4a9d0dd6397ee/ch07/01_main-chapter-code
fetch "$book/previous_chapters.py" previous_chapters.py 9f5aa5f49072e6e935e08738f598f71a6f02753d5d79bfdebfd71dd1a4e5ffec
fetch "$book/instruction-data.json" instruction-data.json 7e09a9af353722ca15c18bc580f093f9c39d3178335a75177319cb1e1d163944
fetch 'https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-medium-355M.pth' gpt2-medium-355M.pth 9b6b658dc97b2f7a029990036638c7219619c6933eada7fb506c554561f73c60
run=$(mktemp -d "$PWD/artifacts/chapter-07/rtx3090-XXXXXXXX")
echo "Saving run to $run"
PYTHONPATH="$cache" MPLBACKEND=Agg uv run --locked --group experiments --group book python -u experiments/instruction-tuning/train.py "$cache" "$run" 2>&1 | tee "$run/training.log"
