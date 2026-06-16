#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"

# Persistent compilation cache is a low-risk speedup for repeated TTAC evals:
# shards use identical shapes/modes but run in separate Python processes.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${ROOT}/.jax_compilation_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"

mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

exec "${ROOT}/experiments/run_ttac_v5_2_full_eval_sharded.sh"
