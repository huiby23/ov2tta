#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${MANIFEST:-experiments/run_registry.tsv}"
INDEX_ROOT="${INDEX_ROOT:-runs_by_config}"
SUMMARY="$INDEX_ROOT/summary.tsv"
MARKER="$INDEX_ROOT/.generated_by_rebuild_run_index"

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

if [[ -e "$INDEX_ROOT" && ! -f "$MARKER" ]]; then
  echo "refusing to overwrite unmarked index directory: $INDEX_ROOT" >&2
  exit 1
fi

rm -rf "$INDEX_ROOT"
mkdir -p "$INDEX_ROOT"
touch "$MARKER"

cat > "$INDEX_ROOT/README.md" <<'README'
# Runs By Config

This directory is a generated pointer index over `runs/`.

- Original run directories are not moved.
- Remote filesystem may not support symlinks, so each leaf directory contains `RUN_PATH.txt`, `RESULT_PATH.txt`, and a copied `result.csv` when available.
- The source of truth is `experiments/run_registry.tsv`.
- Rebuild with `experiments/rebuild_run_index.sh`.
- Layout: `runs_by_config/<config>/<method>/<variant>/<label>`.
- `summary.tsv` contains SP/XP extracted from each registered reward CSV when available.

Use this index for browsing and reporting. Keep training outputs in their original `runs/` locations unless a future runner is explicitly changed to write structured paths.
README

printf 'config\tmethod\tvariant\tlabel\tstatus\tSP\tXP\tnSP\tnXP\trun_dir\tresult_file\tnotes\n' > "$SUMMARY"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r config method variant label run_dir result_file notes; do
  [[ -z "${config:-}" || -z "${method:-}" || -z "${variant:-}" || -z "${label:-}" ]] && continue
  if [[ "${result_file:-}" == "-" ]]; then
    result_file=""
  fi
  dest_parent="$INDEX_ROOT/$config/$method/$variant"
  mkdir -p "$dest_parent"
  dest="$dest_parent/$label"

  status="missing_run"
  sp="NA"
  xp="NA"
  nsp="0"
  nxp="0"
  result_path=""

  if [[ -d "$run_dir" ]]; then
    status="run_present"
    rm -rf "$dest"
    mkdir -p "$dest"
    printf '%s\n' "$run_dir" > "$dest/RUN_PATH.txt"
    if [[ -n "${result_file:-}" && -f "$run_dir/$result_file" ]]; then
      result_path="$run_dir/$result_file"
      printf '%s\n' "$result_path" > "$dest/RESULT_PATH.txt"
      cp "$result_path" "$dest/result.csv"
      stats="$(awk -F, '
        NR > 1 {
          split($2, a, "[-_]");
          i = a[2]; j = a[3];
          if (i == j) { sp += $4; nsp += 1 } else { xp += $4; nxp += 1 }
        }
        END {
          if (nsp > 0) spv = sp / nsp; else spv = "NA";
          if (nxp > 0) xpv = xp / nxp; else xpv = "NA";
          printf "%s\t%s\t%d\t%d", spv, xpv, nsp, nxp;
        }
      ' "$run_dir/$result_file")"
      IFS=$'\t' read -r sp xp nsp nxp <<< "$stats"
      status="result_present"
    elif [[ -n "${result_file:-}" ]]; then
      result_path="$run_dir/$result_file"
      printf '%s\n' "$result_path" > "$dest/RESULT_PATH.txt"
      status="pending_result"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$config" "$method" "$variant" "$label" "$status" "$sp" "$xp" "$nsp" "$nxp" "$run_dir" "$result_path" "$notes" >> "$SUMMARY"
done

echo "rebuilt $INDEX_ROOT"
echo "summary: $SUMMARY"
