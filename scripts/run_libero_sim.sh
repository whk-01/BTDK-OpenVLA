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

# robosuite 1.4.1 (the version LIBERO pins) compares ``model.jnt_type`` -- a plain int --
# against members of ``mujoco.mjtJoint``.  Those members were IntEnum (int-comparable) up
# to mujoco 2.3.x but became plain Enum in mujoco 3.0, so the comparison silently turns
# False and robosuite dies with:
#   AssertionError  robosuite/utils/binding_utils.py:... in get_joint_qpos_addr
# LIBERO's own requirements pin mujoco==2.3.7; fail fast with the exact fix.
if [[ "${BTKD_SKIP_MUJOCO_CHECK:-0}" != "1" && "${DRY_RUN}" != "1" ]]; then
  MUJOCO_VER="$("${PYTHON_BIN}" - <<'PYVER' 2>/dev/null || true
try:
    import mujoco
    print(mujoco.__version__)
except Exception:
    print("")
PYVER
)"
  if [[ -z "${MUJOCO_VER}" ]]; then
    echo "[warn] mujoco not importable in this env; LIBERO simulation will fail." >&2
  elif [[ "${MUJOCO_VER%%.*}" -ge 3 ]]; then
    echo "[FAIL] mujoco ${MUJOCO_VER} is incompatible with robosuite 1.4.1." >&2
    echo "        mujoco 3.x turned mjtJoint/mjtObj into plain Enums, so robosuite's" >&2
    echo "        get_joint_qpos_addr() raises AssertionError." >&2
    echo "        Fix (LIBERO pins 2.3.7):" >&2
    echo "          \"${PYTHON_BIN}\" -m pip install --no-cache-dir 'mujoco==2.3.7' 'numpy==1.26.4'" >&2
    echo "        Skip this check with BTKD_SKIP_MUJOCO_CHECK=1 (not recommended)." >&2
    exit 1
  else
    echo "[info] mujoco ${MUJOCO_VER} (ok for robosuite 1.4.1)"
  fi
fi
# LIBERO's repo layout is nested (LIBERO/libero/libero).  `pip install -e .` from the
# repo ROOT in setuptools "strict" editable mode maps only `libero.libero`, so
# `import libero` fails with ModuleNotFoundError although pip reported success.
# Putting the repo ROOT on PYTHONPATH makes `libero` resolve as a namespace package.
libero_importable() {
  "${PYTHON_BIN}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("libero") else 1)' >/dev/null 2>&1
}

if [[ -n "${LIBERO_PATH}" ]]; then
  if [[ -f "${LIBERO_PATH}/libero/libero/__init__.py" ]]; then
    :  # already the repo root
  elif [[ -f "${LIBERO_PATH}/libero/__init__.py" ]]; then
    # caller passed .../LIBERO/libero; the importable root is its parent
    LIBERO_PATH="$(cd "${LIBERO_PATH}/.." && pwd)"
  fi
fi

if [[ -z "${LIBERO_PATH}" ]] && ! libero_importable; then
  for cand in "${ROOT}/LIBERO" "$(dirname "${ROOT}")/LIBERO" "${HOME}/LIBERO"; do
    if [[ -f "${cand}/libero/libero/__init__.py" ]]; then
      LIBERO_PATH="${cand}"
      break
    fi
  done
fi

if [[ -n "${LIBERO_PATH}" ]]; then
  export PYTHONPATH="${LIBERO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
  echo "[info] LIBERO source: ${LIBERO_PATH} (prepended to PYTHONPATH)"
  LIBERO_BENCH_ROOT="${LIBERO_PATH}/libero/libero"
else
  # Installed package (editable or not): ask importlib where libero/libero/__init__.py is.
  # find_spec does not execute libero/libero/__init__.py, so no interactive config prompt.
  LIBERO_BENCH_ROOT="$("${PYTHON_BIN}" -c '
import os
try:
    import importlib.util
    spec = importlib.util.find_spec("libero.libero")
    print(os.path.dirname(os.path.abspath(spec.origin)) if spec and spec.origin else "")
except Exception:
    print("")
' 2>/dev/null || true)"
fi

# ~/.libero/config.yaml is global to the user and is created once by whichever LIBERO
# install imported first -- possibly another conda env (e.g. whk_smolvla).  Its
# bddl_files/init_files then point at a path that does not exist here, and the run dies
# with FileNotFoundError on "*.pruned_init".  Pin a BTKD-local config instead.
if [[ -n "${LIBERO_BENCH_ROOT}" && -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  export LIBERO_CONFIG_PATH="${ROOT}/.libero"
  mkdir -p "${LIBERO_CONFIG_PATH}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" - "${LIBERO_BENCH_ROOT}" "${LIBERO_CONFIG_PATH}/config.yaml" <<'PYCFG'
import json, pathlib, sys

root, out = sys.argv[1], sys.argv[2]
cfg = {
    "benchmark_root": root,
    "bddl_files": f"{root}/bddl_files",
    "init_states": f"{root}/init_files",
    "datasets": f"{root}/../datasets",
    "assets": f"{root}/assets",
}
# JSON is a subset of YAML, so LIBERO's yaml.load() reads this fine and we do not
# depend on PyYAML being importable by whichever python is used.
pathlib.Path(out).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
missing = [k for k, v in cfg.items() if not pathlib.Path(v).exists() and k != "datasets"]
extra = f" (MISSING: {', '.join(missing)})" if missing else ""
print(f"[info] LIBERO config: {out}{extra}")
PYCFG
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["torch", "transformers", "peft", "tensorflow", "libero", "libero.libero", "robosuite", "bddl", "imageio"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "[FAIL] missing Python packages: " + ", ".join(missing) + "\n"
        "One-time LIBERO setup in the BTKD env:\n"
        "  pip install 'robosuite==1.4.1' bddl easydict cloudpickle gym 'imageio[ffmpeg]'\n"
        "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
        "  cd LIBERO && pip install --no-deps -e . --config-settings editable_mode=compat\n"
        "NOTE: the default 'strict' editable mode is broken for LIBERO's nested layout\n"
        "      (pip says OK, but `import libero` raises ModuleNotFoundError).\n"
        "Fallback without reinstalling: rerun with --libero-path /absolute/path/to/LIBERO\n"
        "      (the repo ROOT that contains the libero/ directory, not libero/libero)."
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
