#!/usr/bin/env bash
set -e

pip install -r requirements.txt

mkdir -p "$HOME/.tmp"
TMPDIR="$HOME/.tmp" pip install "flash-attn==2.6.3" --no-build-isolation

# Minference, FlexPrefill
TMPDIR="$HOME/.tmp" pip install "minference==0.1.5.post1" --no-build-isolation
pip install triton==3.0.0


