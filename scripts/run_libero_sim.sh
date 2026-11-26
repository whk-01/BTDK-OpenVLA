#!/usr/bin/env bash
# Evaluate a BTKD/OpenVLA checkpoint in the LIBERO simulator on one GPU.
#
# Smoke test (1 episode per task, 10 LIBERO-Spatial tasks):
#   bash scripts/run_libero_sim.sh --trials 1
#
# Full LIBERO-Spatial (10 tasks x 50 episodes):
#   nohup bash scripts/run_libero_sim.sh --trials 50 --note btkd_best_full \
#     > logs/libero_btkd_best_full.out 2>&1 &
#
# Evaluate an ordinary full HF checkpoint (e.g. the teacher) without merging:
#   bash scripts/run_libero_sim.sh --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#     --trials 1 --center-crop true --note teacher_smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENVLA_DIR="${ROOT}/openvla"

ENV_NAME="${BTKD_ENV_NAME:-whk_BTKD}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
SUITE="libero_spatial"
TRIALS="1"
NOTE="btkd_smoke"
CENTER_CROP="false"
CHECKPOINT=""
BASE="${OPENVLA_DIR}/openvla-7b"
ADAPTER=""
MERGED="${ROOT}/merged/btkd_spatial_best"
STATS="openvla/openvla-7b-finetuned-libero-spatial"
MERGE_DEVICE="cpu"
REBUILD_MERGE=0
MUJOCO_GL_VALUE="egl"
PYTHON_BIN=""
LIBERO_PATH=""
DRY_RUN=0
EVAL_EXTRA=()

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --env-name NAME       conda env (default: whk_BTKD)
  --gpu N               physical GPU id via CUDA_VISIBLE_DEVICES (default: 3)
  --suite NAME          libero_spatial | libero_object | libero_goal | libero_10 | libero_90
  --trials N            episodes per task (default: 1; paper protocol: 50)
  --adapter PATH        BTKD LoRA adapter (default: newest runs_btkd/*/best)
  --merged PATH         output of the merged checkpoint (default: merged/btkd_spatial_best)
  --base PATH           base OpenVLA checkpoint (default: openvla/openvla-7b)
  --stats PATH          dataset_statistics.json, teacher dir, or HF repo id
                        (default: openvla/openvla-7b-finetuned-libero-spatial)
  --checkpoint PATH     evaluate this complete HF checkpoint directly; skip merge
  --center-crop BOOL    true for the official OpenVLA LIBERO checkpoint, false for BTKD
  --merge-device DEV    cpu (default) or cuda:0 used only while merging
  --rebuild-merge       rebuild --merged even if it already exists
  --mujoco-gl VALUE     egl (default) or osmesa
  --python PATH         use this Python instead of activating --env-name
  --libero-path PATH    prepend an existing LIBERO source checkout to PYTHONPATH
  --dry-run             print the commands without running them
  --                    pass all remaining arguments to run_libero_eval.py
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)      ENV_NAME="$2"; shift 2 ;;
    --gpu)           GPU="$2"; shift 2 ;;
    --suite)         SUITE="$2"; shift 2 ;;
    --trials)        TRIALS="$2"; shift 2 ;;
    --note)          NOTE="$2"; shift 2 ;;
    --adapter)       ADAPTER="$2"; shift 2 ;;
    --merged)        MERGED="$2"; shift 2 ;;
    --base)          BASE="$2"; shift 2 ;;
    --stats)         STATS="$2"; shift 2 ;;
    --checkpoint)    CHECKPOINT="$2"; shift 2 ;;
    --center-crop)   CENTER_CROP="$2"; shift 2 ;;
    --merge-device)  MERGE_DEVICE="$2"; shift 2 ;;
    --rebuild-merge) REBUILD_MERGE=1; shift ;;
    --mujoco-gl)     MUJOCO_GL_VALUE="$2"; shift 2 ;;
    --python)        PYTHON_BIN="$2"; shift 2 ;;
    --libero-path)   LIBERO_PATH="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    --)              shift; EVAL_EXTRA=("$@"); break ;;
    *)               echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${OPENVLA_DIR}/experiments/robot/libero/run_libero_eval.py" ]]; then
  echo "[FAIL] OpenVLA source not found: ${OPENVLA_DIR}" >&2
  exit 1
fi

if [[ -z "${ADAPTER}" ]]; then
  ADAPTER="$(find "${ROOT}/runs_btkd" -mindepth 2 -maxdepth 2 -type d -name best -print 2>/dev/null | xargs -r ls -td 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -z "${ADAPTER}" || ! -f "${ADAPTER}/adapter_config.json" ]]; then
    echo "[FAIL] no BTKD adapter found. Pass --adapter runs_btkd/<exp>/best or --checkpoint <full model>." >&2
    exit 1
  fi
  if [[ ! -f "${BASE}/config.json" ]]; then
    echo "[FAIL] base OpenVLA checkpoint not found: ${BASE}" >&2
    exit 1
  fi
  CHECKPOINT="${MERGED}"
fi

# Activate the requested environment unless an explicit Python was supplied.
if [[ -z "${PYTHON_BIN}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[FAIL] conda not found; pass --python /path/to/env/bin/python" >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  PYTHON_BIN="$(command -v python)"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${MUJOCO_GL_VALUE}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
if [[ -n "${LIBERO_PATH}" ]]; then
  export PYTHONPATH="${LIBERO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["torch", "transformers", "peft", "tensorflow", "libero", "robosuite", "bddl", "imageio"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "[FAIL] missing Python packages: " + ", ".join(missing) + "\n"
        "One-time LIBERO setup in the BTKD env:\n"
        "  pip install 'robosuite==1.4.1' bddl easydict cloudpickle gym 'imageio[ffmpeg]'\n"
        "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
        "  pip install --no-deps -e ./LIBERO\n"
        "Then rerun with --libero-path /absolute/path/to/LIBERO if it is not importable."
    )
PY
fi

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

# A complete merged checkpoint needs real weight shards, not just config.json.
need_merge=0
if [[ "${CHECKPOINT}" == "${MERGED}" ]]; then
  if [[ "${REBUILD_MERGE}" == "1" ]]; then
    need_merge=1
  elif [[ ! -f "${MERGED}/config.json" ]] || ! compgen -G "${MERGED}/*.safetensors" >/dev/null; then
    need_merge=1
  fi
fi

if [[ "${need_merge}" == "1" ]]; then
  echo "==> merge BTKD adapter into a full OpenVLA checkpoint"
  echo "    adapter : ${ADAPTER}"
  echo "    merged  : ${MERGED}"
  merge_args=(
    "${ROOT}/scripts/merge_btkd_adapter.py"
    --base "${BASE}"
    --adapter "${ADAPTER}"
    --out "${MERGED}"
    --device "${MERGE_DEVICE}"
  )
  if [[ -n "${STATS}" ]]; then
    merge_args+=(--stats-from "${STATS}")
  fi
  if [[ "${REBUILD_MERGE}" == "1" ]]; then
    merge_args+=(--overwrite)
  fi
  run env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${merge_args[@]}"
fi

mkdir -p "${ROOT}/logs"

echo "==> LIBERO rollout"
echo "    checkpoint : ${CHECKPOINT}"
echo "    suite      : ${SUITE}"
echo "    trials/task: ${TRIALS}"
echo "    GPU        : physical ${GPU} (inside process: cuda:0)"
echo "    log dir    : ${OPENVLA_DIR}/experiments/logs"

eval_args=(
  "${ROOT}/scripts/libero_eval_entry.py"
  --model_family openvla
  --pretrained_checkpoint "${CHECKPOINT}"
  --task_suite_name "${SUITE}"
  --num_trials_per_task "${TRIALS}"
  --center_crop "${CENTER_CROP}"
  --run_id_note "${NOTE}"
)
if [[ ${#EVAL_EXTRA[@]} -gt 0 ]]; then
  eval_args+=("${EVAL_EXTRA[@]}")
fi

cd "${OPENVLA_DIR}"
run env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${eval_args[@]}"
