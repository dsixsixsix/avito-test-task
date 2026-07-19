#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
HF=https://huggingface.co

dl() {
  local repo=$1 dst=$2; shift 2
  mkdir -p "$dst"
  for f in "$@"; do
    [ -s "$dst/$f" ] || curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
      -o "$dst/$f" "$HF/$repo/resolve/main/$f"
  done
}

dl BAAI/bge-m3 solution/models/bge-m3 \
  config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
  sentencepiece.bpe.model colbert_linear.pt sparse_linear.pt pytorch_model.bin

dl cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 solution/models/mmarco-minilm \
  config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
  sentencepiece.bpe.model pytorch_model.bin

dl sergeyzh/BERTA solution/models/berta \
  config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
  vocab.txt model.safetensors

echo "OK"
