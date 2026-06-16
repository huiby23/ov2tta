#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"

# Fast functional TTAC eval profile: batch all pairings inside each shard.
# This keeps old object eval untouched and requires the optimized functional path.
export FUNCTIONAL_EVAL="${FUNCTIONAL_EVAL:-1}"
export EVAL_BATCHES="${EVAL_BATCHES:-1}"
export SHARD_SIZE="${SHARD_SIZE:-10}"
export PARALLEL_JOBS="${PARALLEL_JOBS:-1}"

# Reuse XLA compilation across shards/processes.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${ROOT}/.jax_compilation_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

exec "${ROOT}/experiments/run_ttac_v5_2_full_eval_sharded.sh"
