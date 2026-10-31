#!/usr/bin/env bash
# 在脚本所在目录生成 {id}_initial_program.cpp（id 见 PROBLEMS）

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEMS=(
  ahc008 ahc011 ahc015 ahc016 ahc024
  ahc025 ahc026 ahc027 ahc039 ahc046
)

for id in "${PROBLEMS[@]}"; do
  f="${DIR}/${id}_initial_program.cpp"
  if [[ ! -f "$f" ]]; then
    printf '// %s — initial program placeholder\n' "$id" > "$f"
    echo "created: $f"
  else
    echo "skip (exists): $f"
  fi
done
