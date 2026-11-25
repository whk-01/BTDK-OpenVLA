#!/usr/bin/env bash
# =============================================================================
# A6 单卡运行器（一块 24GB 3090 Ti 跑通 导出 → 校验 → 物化 → 蒸馏）
#
#   bash run_a6_single_gpu.sh export       --dataset bridge_orig  # 1) 导出 teacher logits
#   bash run_a6_single_gpu.sh check        --dataset bridge_orig  # 2) 只校验缓存
#   bash run_a6_single_gpu.sh materialize  --dataset bridge_orig  # 3) 把帧物化到磁盘（只跑一次）
#   bash run_a6_single_gpu.sh train        --dataset bridge_orig  # 4) BTKD 蒸馏（自动用 frames 快路径）
#   bash run_a6_single_gpu.sh eval         --dataset bridge_orig  # 5) 离线验证 adapter（对比缓存 teacher）
#   bash run_a6_single_gpu.sh all          --dataset bridge_orig  # 1+2+3+4 顺序执行
#
#   默认只把一张卡暴露给进程：CUDA_VISIBLE_DEVICES=<GPU>，脚本内一律 --device cuda:0
#   GPU 选择: --gpu 0 | --gpu 3 | --gpu auto（挑当前最空闲的一张）| --gpu all
#
# 24GB 的推荐值（脚本默认）
#   导出: batch_size=2（OOM 自动降 1）  显存 ~18-20GB
#   训练: bf16-LoRA, batch_size=1, grad_accum=16, gradient checkpointing（约 12GB）
#         若能装 bitsandbytes 则 4-bit QLoRA，batch_size=4 / accum 4
#
# 为什么需要 materialize
#   * 缓存对齐要求 RLDS 用确定性顺序（单读/单 map worker），训练时每步都要解 JPEG，
#     一张 3090 Ti 上只有个位数 step/s。
#   * materialize 只遍历一次，把 input_ids/labels/pixel_values(fp16) 写成按 cache 行号
#     对齐的分片；之后训练直接按行号随机读，缓存和帧天然对齐，可以安全 shuffle、跑多 epoch。
#   * 磁盘代价：约 3*224*224*2B ≈ 300KB/样本 → 5.3 万样本约 16GB。
#
# 注意
#   * 导出阶段不要加 --quantize_4bit：4-bit 噪声会污染 teacher logits（见 docs §9.4）
#   * 导出的 --shuffle_buffer_size 必须为 1、不要 --image_aug：否则缓存与数据集错位
#   * --resume 按 manifest 中实际落盘行数续写，可以更换 batch size
#   * 旧版缓存（CACHE_VERSION/data_order 不匹配）会自动备份为 *.legacy_时间戳 后重导
#   * 完整缓存可跨 epoch 回绕并在每次回绕时重新校验 GT token；--limit 禁止回绕
#   * 训练默认按 epoch 计步（--epochs 2），不再是 --max-steps 20000 的 micro-batch 语义
#   * final/ 只是最后一刻的权重；请优先用 best/（训练中评估选出的最好 checkpoint）
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="${1:-all}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then MODE="help"; else shift || true; fi

ENV_NAME="${BTKD_ENV_NAME:-BTKD}"
GPU="${BTKD_GPU:-0}"
DATASET="bridge_orig"
DATA_ROOT="${SCRIPT_DIR}/datasets/open-x-embodiment"
TEACHER="${SCRIPT_DIR}/openvla/openvla-7b"
CACHE=""
BS=2
SHARD=5000
LIMIT=-1
LIMIT_SET=0
TRAIN_BS=4
TRAIN_GA=4
EVAL_BS=4
ADAPTER=""
# 仅作为 --epochs 0 时的兜底；默认训练按 EPOCHS 计（见下方 schedule_args）
MAX_STEPS=20000
MAX_STEPS_SET=0
SAVE_STEPS=2000
EPOCHS=2
EPOCHS_SET=0
EVAL_STEPS=2000
EVAL_LIMIT=512
EASY_HOLD=0.4
TEMPERATURE=2.0
LR=5e-4
LR_WARMUP=0.03
FRAMES=""
MATERIALIZE=1
MAT_SHARD=1000
FRAME_SHARDS=8
FRAME_SHUFFLE="shard"
PROFILE_STEPS=0
PY=""
DRY_RUN=0
PY_EXTRA=()

LOG_DIR="${SCRIPT_DIR}/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/a6_$(date +%Y%m%d_%H%M%S).log"

C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_INFO=$'\033[36m'; C_OFF=$'\033[0m'
step() { printf "\n${C_INFO}== %s ==${C_OFF}\n" "$*"; }
ok()   { printf "${C_OK}[ OK ]${C_OFF} %s\n" "$*"; }
warn() { printf "${C_WARN}[WARN]${C_OFF} %s\n" "$*"; }
die()  { printf "${C_ERR}[FAIL]${C_OFF} %s\n" "$*" >&2; exit 1; }

usage() {
    sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'TIP'

参数:
  --dataset NAME          OXE 数据集/混合名（默认 bridge_orig）
  --data-root DIR         RLDS 数据根目录（默认 <repo>/datasets/open-x-embodiment）
  --teacher PATH          teacher 权重目录（默认 <repo>/openvla/openvla-7b）
  --cache DIR             缓存目录（默认 <repo>/caches/<dataset>_teacher）
  --gpu 0|N|auto|all      使用哪张卡（默认 0）
  --batch-size N          导出 batch（默认 2；OOM 自动降 1）
  --limit N               export/train 只处理前 N 条（调试用）；eval 下 -1 = 全部
  --train-batch-size N    训练 batch（默认 4）
  --grad-accum N          梯度累积（默认 4）
  --adapter DIR           eval 用的 LoRA adapter（默认自动取 runs_btkd/*/final 最新一个）
  --eval-batch-size N     eval batch（默认 4；不传 --limit 时只验证前 512 条）
  --epochs N              训练几个 epoch（默认 2；设为 0 才用 --max-steps）
  --max-steps N           micro-batch 步数（仅在 --epochs 0 时生效）
  --easy-hold-frac F      前 F 比例保持 easy 分支，再余弦降到 hard（默认 0.4）
  --temperature T         KD 温度（默认 2.0；4.0 = 论文原值，hard 分支权重更大）
  --learning-rate LR      学习率（默认 5e-4）
  --save-steps N          每 N 个 optimizer update 存 checkpoint（默认 2000）
  --eval-steps N          每 N 个 optimizer update 做一次离线评估（默认 2000；0=关闭）
  --eval-limit N          训练中评估用的样本数（默认 512）
  --frames-dir DIR        frames 目录（默认 <cache>_frames）
  --materialize-shards N  materialize 分片大小（默认 1000 行/片）
  --frame-shards N       训练时同时打开的 frames 分片数（默认 8；0=全部常驻）
  --frame-shuffle MODE   shard（默认，分片内乱序，最快）| global（全局乱序，慢）
  --profile-steps N      前 N 步打印 data/fwd/bwd 耗时拆分（默认 0=关闭）
  --no-materialize        train 前不自动物化 frames（沿用已有 frames 或回退 RLDS）
  --overwrite-frames      materialize 时覆盖已存在的 frames
  --python PATH           指定 python（默认用 conda 环境里的）
  --env-name NAME         conda 环境名（默认 BTKD / $BTKD_ENV_NAME）
  --train-arg ARG         透传给 train_btkd_student.py（可重复）
  --dry-run               只打印命令
TIP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)          DATASET="$2"; shift 2 ;;
        --data-root)        DATA_ROOT="$2"; shift 2 ;;
        --teacher)          TEACHER="$2"; shift 2 ;;
        --cache)            CACHE="$2"; shift 2 ;;
        --gpu)              GPU="$2"; shift 2 ;;
        --batch-size)       BS="$2"; shift 2 ;;
        --limit)            LIMIT="$2"; LIMIT_SET=1; shift 2 ;;
        --train-batch-size) TRAIN_BS="$2"; shift 2 ;;
        --adapter)          ADAPTER="$2"; shift 2 ;;
        --eval-batch-size)  EVAL_BS="$2"; shift 2 ;;
        --grad-accum)       TRAIN_GA="$2"; shift 2 ;;
        --max-steps)        MAX_STEPS="$2"; MAX_STEPS_SET=1; shift 2 ;;
        --epochs)           EPOCHS="$2"; EPOCHS_SET=1; shift 2 ;;
        --easy-hold-frac)   EASY_HOLD="$2"; shift 2 ;;
        --temperature)      TEMPERATURE="$2"; shift 2 ;;
        --learning-rate)    LR="$2"; shift 2 ;;
        --save-steps)       SAVE_STEPS="$2"; shift 2 ;;
        --eval-steps)       EVAL_STEPS="$2"; shift 2 ;;
        --eval-limit)       EVAL_LIMIT="$2"; shift 2 ;;
        --frames-dir)       FRAMES="$2"; shift 2 ;;
        --materialize-shards) MAT_SHARD="$2"; shift 2 ;;
        --frame-shards)     FRAME_SHARDS="$2"; shift 2 ;;
        --frame-shuffle)    FRAME_SHUFFLE="$2"; shift 2 ;;
        --profile-steps)    PROFILE_STEPS="$2"; shift 2 ;;
        --no-materialize)   MATERIALIZE=0; shift ;;
        --overwrite-frames) MAT_OVERWRITE=1; shift ;;
        --python)           PY="$2"; shift 2 ;;
        --env-name)         ENV_NAME="$2"; shift 2 ;;
        --train-arg)        PY_EXTRA+=("$2"); shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          MODE="help"; shift ;;
        *) die "未知参数: $1（--help 查看用法）" ;;
    esac
done

[[ "${CACHE}" == "" ]] && CACHE="${SCRIPT_DIR}/caches/${DATASET}_teacher"
[[ "${FRAMES}" == "" ]] && FRAMES="${CACHE%/}_frames"
MAT_OVERWRITE="${MAT_OVERWRITE:-0}"
if (( EPOCHS_SET == 1 && MAX_STEPS_SET == 1 )); then
    die "同时给了 --epochs 和 --max-steps；只保留一个（--epochs 0 才用 --max-steps）"
fi
# 只给 --max-steps 时自动切回旧的步数语义，避免被默认的 EPOCHS=2 静默吃掉
if (( MAX_STEPS_SET == 1 && EPOCHS_SET == 0 )); then
    EPOCHS=0
    warn "--max-steps 已给出：自动切到 --epochs 0（按 micro-batch 计步）"
fi

if [[ "${MODE}" == "help" ]]; then usage; exit 0; fi
case "${MODE}" in export|check|materialize|train|eval|all) ;; *) die "第一个参数必须是 export|check|materialize|train|eval|all，收到: ${MODE}" ;; esac

exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'die "脚本在第 $LINENO 行失败；日志: ${LOG_FILE}"' ERR

exec_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; return 0; fi
    "$@"
}

# ---------------------------------------------------------------------------
step "0. 单卡初始化"

# python：优先用 --python，其次 conda 环境
if [[ -z "${PY}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
    else
        CONDA_BIN=""
        for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
            [[ -x "$c" ]] && { CONDA_BIN="$c"; break; }
        done
    fi
    if [[ -n "${CONDA_BIN}" ]]; then
        CONDA_BASE="$("${CONDA_BIN}" info --base)"
        [[ -x "${CONDA_BASE}/envs/${ENV_NAME}/bin/python" ]] \
            && PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
    fi
fi
[[ -n "${PY}" ]] || PY="$(command -v python || true)"
[[ -n "${PY}" && -x "${PY}" ]] || die "找不到 python：先跑 install_btkd_oneclick.sh，或用 --python 指定"
export PATH="$(dirname "${PY}"):${PATH}"
ok "python: $("${PY}" -V 2>&1) @ ${PY}"

# 选卡：只把一张卡暴露给所有子进程（TF/OpenVLA 数据管道最容易吃满全部卡）
if [[ "${GPU}" == "auto" ]]; then
    command -v nvidia-smi >/dev/null || die "--gpu auto 需要 nvidia-smi"
    GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | awk -F', *' '{print $2" "$1}' | sort -rn | head -1 | awk '{print $2}')"
    ok "自动选择最空闲的 GPU ${GPU}"
fi

if [[ "${GPU}" == "all" ]]; then
    unset CUDA_VISIBLE_DEVICES || true
    warn "GPU=all：进程能看到所有卡，但仍只跑单进程、--device cuda:0"
else
    export CUDA_VISIBLE_DEVICES="${GPU}"
    ok "CUDA_VISIBLE_DEVICES=${GPU}（进程内只有 cuda:0）"
fi

# 显存相关环境变量：TF 不要一次占满整张卡；长时训练减少碎片
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${GPU}" != "all" ]]; then
    INFO="$(nvidia-smi -i "${GPU}" --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader 2>/dev/null || true)"
    [[ -n "${INFO}" ]] && ok "GPU: ${INFO}"
    FREE_MB="$(awk -F', *' '{print $5}' <<<"${INFO}" | tr -dc '0-9')"
    if [[ -n "${FREE_MB}" ]] && (( FREE_MB < 20000 )); then
        warn "该卡当前空闲 ${FREE_MB} MiB < 20 GB：导出 7B teacher 可能 OOM（可 --batch-size 1 或等别人释放）"
    fi
fi
ok "缓存目录: ${CACHE}"
ok "frames 目录: ${FRAMES}"
ok "日志: ${LOG_FILE}"

# ---------------------------------------------------------------------------
do_export() {
    step "1. 导出 teacher logits（bs=${BS}，单卡，一次前向）"
    mkdir -p "${CACHE}"

    local manifest="${CACHE}/manifest.json"
    local resume=""
    local overwrite=""
    local done_samples=0 cache_complete=false info=""
    local cache_version=-1 cache_order="" same_dataset=false same_root=false

    if [[ -f "${manifest}" ]]; then
        if ! info="$("${PY}" - "${manifest}" "${DATASET}" "${DATA_ROOT}" <<'PY'
import json, os, sys
from btkd_vla.cache import CACHE_VERSION, DATA_ORDER

m = json.load(open(sys.argv[1], encoding="utf-8"))
meta = m.get("meta") or {}
print("\t".join([
    str(bool(m.get("complete", False))).lower(),
    str(int(m.get("n_samples", 0))),
    str(int(m.get("version", -1))),
    str(meta.get("data_order", "")),
    str(meta.get("dataset") == sys.argv[2]).lower(),
    str(os.path.abspath(str(meta.get("data_root_dir", ""))) == os.path.abspath(sys.argv[3])).lower(),
    str(CACHE_VERSION),
    DATA_ORDER,
]))
PY
        )"; then
            die "无法读取缓存 manifest: ${manifest}（文件损坏？）"
        fi
        local expected_version expected_order
        IFS=$'\t' read -r cache_complete done_samples cache_version cache_order \
            same_dataset same_root expected_version expected_order <<<"${info}"

        if [[ "${same_dataset}" != "true" ]]; then
            die "缓存属于另一个数据集，拒绝覆盖: ${manifest}。请换 --cache 路径或手动处理旧缓存。"
        fi
        if [[ "${same_root}" != "true" ]]; then
            die "缓存的数据根目录与当前 --data-root 不一致，拒绝覆盖: ${manifest}。请换 --cache 路径或手动处理旧缓存。"
        fi

        if [[ "${cache_version}" != "${expected_version}" || "${cache_order}" != "${expected_order}" ]]; then
            local backup="${CACHE}.legacy_$(date +%Y%m%d_%H%M%S)"
            warn "旧缓存格式/顺序不兼容（version=${cache_version}, data_order=${cache_order:-<empty>}）"
            if [[ "${DRY_RUN}" == "1" ]]; then
                warn "[dry-run] 将把旧缓存备份到: ${backup}，然后重新导出"
            else
                mv "${CACHE}" "${backup}"
                mkdir -p "${CACHE}"
                ok "旧缓存已备份到: ${backup}"
            fi
            overwrite="--overwrite"
        elif [[ "${cache_complete}" == "true" ]]; then
            ok "缓存已完成（${done_samples} 条，确定性顺序，manifest complete=true），跳过导出"
            return 0
        else
            # An interrupted export may leave a valid manifest with zero rows. Always
            # resume in that case; otherwise the exporter would reject the manifest.
            resume="--resume"
            if (( done_samples > 0 )); then
                warn "检测到未完成的缓存（已落盘 ${done_samples} 条），将 --resume 续跑"
            else
                warn "检测到未完成但尚无数据的缓存（manifest n_samples=0），将 --resume 从第 0 条开始"
            fi
        fi
    fi

    local limit_args=()
    (( LIMIT > 0 )) && limit_args=(--limit "${LIMIT}")

    local -a cmd=("${PY}" scripts/export_teacher_logits.py
        --teacher_path "${TEACHER}"
        --data_root_dir "${DATA_ROOT}"
        --dataset_name "${DATASET}"
        --cache_dir "${CACHE}"
        --batch_size "${BS}"
        --shard_size "${SHARD}"
        --device cuda:0
        --shuffle_buffer_size 1
        "${limit_args[@]}")

    [[ -n "${resume}" ]] && cmd+=("${resume}")
    [[ -n "${overwrite}" ]] && cmd+=("${overwrite}")

    local rc=0
    if [[ "${DRY_RUN}" == "1" ]]; then exec_cmd "${cmd[@]}"; return 0; fi

    set +e
    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}.export"
    rc=${PIPESTATUS[0]}
    set -e

    if (( rc != 0 )); then
        if grep -qi "out of memory" "${LOG_FILE}.export" && (( done_samples == 0 )) && (( BS > 1 )); then
            warn "GPU 显存不足，自动降到 batch_size=1 重跑"
            BS=1
            "${PY}" scripts/export_teacher_logits.py \
                --teacher_path "${TEACHER}" --data_root_dir "${DATA_ROOT}" \
                --dataset_name "${DATASET}" --cache_dir "${CACHE}" \
                --batch_size 1 --shard_size "${SHARD}" --device cuda:0 \
                --shuffle_buffer_size 1 --overwrite "${limit_args[@]}" 2>&1 | tee -a "${LOG_FILE}.export"
        else
            die "导出失败（rc=${rc}）。日志: ${LOG_FILE}.export"
        fi
    fi
    if [[ "${DRY_RUN}" == "0" ]]; then ok "导出完成: ${CACHE}"; fi
}

do_check() {
    step "2. 校验缓存统计（CPU；不加载模型/数据集）"
    local limit_args=()
    (( LIMIT > 0 )) && limit_args=(--limit "${LIMIT}")
    local -a cmd=("${PY}" scripts/train_btkd_student.py
        --cache_dir "${CACHE}"
        --dataset_name "${DATASET}"
        --data_root_dir "${DATA_ROOT}"
        --vla_path "${TEACHER}"
        "${limit_args[@]}"
        --shuffle_buffer_size 1
        --check_cache_only)
    exec_cmd "${cmd[@]}"
    if [[ "${DRY_RUN}" == "0" ]]; then ok "缓存校验通过"; fi
}

do_materialize() {
    step "3. 物化训练帧（CPU，一次性；之后训练不再碰 TFDS）"
    if [[ -f "${FRAMES}/manifest.json" && "${MAT_OVERWRITE}" == "0" ]]; then
        local state
        state="$("${PY}" - "${FRAMES}" <<'PY'
import sys
from btkd_vla.frames import FrameStore
s = FrameStore(sys.argv[1])
print("complete" if s.is_complete else "incomplete", len(s))
PY
)"
        case "${state}" in
            complete*)
                ok "frames 已完整（${state#complete } 行），跳过: ${FRAMES}"
                ok "如需重建加 --overwrite-frames"
                return 0 ;;
            *)
                warn "检测到未完成的 frames（${state}），将重建"
                MAT_OVERWRITE=1 ;;
        esac
    fi
    local limit_args=()
    (( LIMIT > 0 )) && limit_args=(--limit "${LIMIT}")
    local -a overwrite_args=()
    if [[ "${MAT_OVERWRITE}" == "1" ]]; then
        warn "将覆盖已有 frames: ${FRAMES}"
        overwrite_args=(--overwrite)
    fi

    local -a cmd=("${PY}" scripts/materialize_frames.py
        --cache_dir "${CACHE}"
        --out_dir "${FRAMES}"
        --dataset_name "${DATASET}"
        --data_root_dir "${DATA_ROOT}"
        --vla_path "${TEACHER}"
        --shard_size "${MAT_SHARD}"
        "${limit_args[@]}"
        "${overwrite_args[@]}")

    local t0=$SECONDS
    exec_cmd "${cmd[@]}"
    if [[ "${DRY_RUN}" == "0" ]]; then
        ok "物化完成: ${FRAMES}（$(( (SECONDS - t0) / 60 )) 分钟）"
        du -sh "${FRAMES}" 2>/dev/null || true
    fi
}

do_train() {
    local quant=(--quantize_4bit)
    if ! "${PY}" -c "import bitsandbytes" >/dev/null 2>&1; then
        warn "bitsandbytes 不可用：退回 bf16 LoRA，并把 batch 降到 1 / accum 提到 16"
        quant=(); TRAIN_BS=1; TRAIN_GA=16
    fi

    # 没有可用 frames 就自动先物化（除非显式 --no-materialize）
    if [[ "${MATERIALIZE}" == "1" ]]; then
        local frames_complete=0
        if [[ -f "${FRAMES}/manifest.json" ]]; then
            frames_complete="$("${PY}" - "${FRAMES}" <<'PY'
import sys
from btkd_vla.frames import FrameStore
try:
    s = FrameStore(sys.argv[1], verbose=False)
    print(1 if s.is_complete else 0)
except Exception:
    print(0)
PY
)"
        fi
        if [[ "${frames_complete}" == "1" ]]; then
            ok "使用已有 frames: ${FRAMES}"
        else
            do_materialize
        fi
    else
        warn "--no-materialize：训练将回退到 RLDS 流式读取（慢，仅调试用）"
    fi

    local -a schedule_args=(--epochs "${EPOCHS}")
    if (( EPOCHS == 0 )); then
        schedule_args=(--max_steps "${MAX_STEPS}")
    fi
    local -a lr_args=(--lr_schedule cosine --lr_warmup_frac "${LR_WARMUP}"
                     --learning_rate "${LR}")
    local plan_label
    if (( EPOCHS > 0 )); then plan_label="epochs=${EPOCHS}"; else plan_label="max_steps=${MAX_STEPS}"; fi

    step "4. BTKD 蒸馏训练（bs=${TRAIN_BS} x accum ${TRAIN_GA}，$( (( ${#quant[@]} > 0 )) && echo 4-bit-QLoRA || echo bf16-LoRA)，${plan_label}）"
    local limit_args=()
    (( LIMIT > 0 )) && limit_args=(--limit "${LIMIT}")
    local -a cmd=("${PY}" scripts/train_btkd_student.py
        --cache_dir "${CACHE}"
        --frames_dir "${FRAMES}"
        --dataset_name "${DATASET}"
        --data_root_dir "${DATA_ROOT}"
        --vla_path "${TEACHER}"
        "${limit_args[@]}"
        --shuffle_buffer_size 1
        --batch_size "${TRAIN_BS}"
        --grad_accumulation_steps "${TRAIN_GA}"
        "${quant[@]}"
        --gradient_checkpointing
        "${schedule_args[@]}"
        --save_steps "${SAVE_STEPS}"
        --eval_steps "${EVAL_STEPS}"
        --eval_limit "${EVAL_LIMIT}"
        --temperature "${TEMPERATURE}"
        --easy_hold_frac "${EASY_HOLD}"
        --frame_cache_shards "${FRAME_SHARDS}"
        --frame_shuffle "${FRAME_SHUFFLE}"
        --profile_steps "${PROFILE_STEPS}"
        "${lr_args[@]}"
        --device cuda:0
        "${PY_EXTRA[@]}")
    exec_cmd "${cmd[@]}"
    if [[ "${DRY_RUN}" == "0" ]]; then
        ok "训练完成，检查 runs_btkd/ 下的 best/ final/ 与 metrics.jsonl"
    fi
}

do_eval() {
    step "4. 离线验证 adapter（不加载 teacher，不跑仿真）"
    local eval_limit=512
    if [[ "${LIMIT_SET}" == "1" ]]; then
        eval_limit="${LIMIT}"
    else
        warn "未指定 --limit：eval 默认只验证前 512 条（全量会非常慢，可用 --limit -1 显式开启）"
    fi
    if [[ -z "${ADAPTER}" ]]; then
        ADAPTER="$(ls -td "${SCRIPT_DIR}"/runs_btkd/*/final 2>/dev/null | head -1 || true)"
        [[ -n "${ADAPTER}" ]] || die "找不到 adapter：请用 --adapter 指定 runs_btkd/<exp>/final 或 checkpoint-XXXX"
        warn "未指定 --adapter，自动使用最新的: ${ADAPTER}"
    fi
    local -a cmd=("${PY}" scripts/eval_btkd_student.py
        --cache_dir "${CACHE}"
        --dataset_name "${DATASET}"
        --data_root_dir "${DATA_ROOT}"
        --vla_path "${TEACHER}"
        --adapter_path "${ADAPTER}"
        --batch_size "${EVAL_BS}"
        --limit "${eval_limit}"
        --device cuda:0)
    exec_cmd "${cmd[@]}"
    if [[ "${DRY_RUN}" == "0" ]]; then ok "验证完成"; fi
}

case "${MODE}" in
    export)      do_export ;;
    check)       do_check ;;
    materialize) do_materialize ;;
    train)       do_train ;;
    eval)        do_eval ;;
    all)         do_export; do_check; do_materialize; do_train ;;
esac

echo
printf "日志: %s\n" "${LOG_FILE}"
echo "提示：单卡 24GB 想再省显存 → 训练加 --train-arg '--lora_rank 16'"
echo "提示：优先用 best/ 而不是 final/（final 是最后一刻权重，可能已在 hard 分支上回退）"
