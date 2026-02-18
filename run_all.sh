#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [--skip-fetch]

Options:
  --skip-fetch   Skip FRED refresh and use existing cached model panel in data/.
EOF
}

SKIP_FETCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fetch)
      SKIP_FETCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${SKIP_FETCH}" -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/fetch_macro_panel_fred.py \
    --raw-output data/macro_panel_quarterly_raw.csv \
    --model-output data/macro_panel_quarterly_model.csv \
    --metadata-output data/macro_panel_metadata.json
else
  echo "[INFO] --skip-fetch enabled; reusing cached panel files under data/."
  if [[ ! -f data/macro_panel_quarterly_model.csv ]]; then
    echo "[ERROR] Missing data/macro_panel_quarterly_model.csv; cannot run with --skip-fetch." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" scripts/run_macro_forecast_engine.py \
  --config macro_engine_config.json \
  --output-dir outputs

"${PYTHON_BIN}" scripts/export_pd_macro_subset.py \
  --input outputs/macro_forecast_paths.csv \
  --output-csv outputs/pd_macro_subset_sample.csv \
  --output-json outputs/pd_macro_subset_sample.json
