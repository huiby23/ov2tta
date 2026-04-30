#!/usr/bin/env bash
# Fast reproducibility profile for JAX/XLA training.
# This keeps multi-GPU execution enabled, but reduces avoidable run-to-run variance.

# Stable Python hash order for any Python-side dict/set operations before JIT.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Keep GPU memory behavior stable across runs.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Ask XLA/GPU kernels to prefer deterministic implementations when available.
# This is best-effort and does not guarantee bitwise identical multi-GPU PPO.
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_deterministic_ops=true"* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
fi

# Keep host-side thread pools stable. Do not over-constrain GPU execution.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Make device order explicit when the caller has not specified it.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

print_repro_env() {
  echo "[repro-env] PYTHONHASHSEED=${PYTHONHASHSEED}"
  echo "[repro-env] CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[repro-env] XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
  echo "[repro-env] XLA_FLAGS=${XLA_FLAGS}"
  echo "[repro-env] OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
}
