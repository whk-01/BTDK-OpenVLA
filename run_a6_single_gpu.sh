#!/usr/bin/env bash
# =============================================================================
# A6 单卡运行器（一块 24GB 3090 Ti 跑通 导出 → 校验 → QLoRA 蒸馏）
#
#   bash run_a6_single_gpu.sh export --dataset bridge_orig    # 1) 导出 teacher logits
#   bash run_a6_single_gpu.sh check  --dataset bridge_orig    # 2) 只校验缓存
#   bash run_a6_single_gpu.sh train  --dataset bridge_orig    # 3) QLoRA 蒸馏
#   bash run_a6_single_gpu.sh all    --dataset bridge_orig    # 1+2+3 顺序执行
#
#   默认只把一张卡暴露给进程：CUDA_VISIBLE_DEVICES=<GPU>，脚本内一律 --device cuda:0
#   GPU 选择: --gpu 0 | --gpu 3 | --gpu auto（挑当前最空闲的一张）| --gpu all
#
# 24GB 的推荐值（脚本默认）
#   导出: batch_size=2（OOM 自动降 1）  显存 ~18-20GB
#   训练: batch_size=4, grad_accum=4, --quantize_4bit  显存 ~10-12GB
#
# 注意
#   * 导出阶段不要加 --quantize_4bit：4-bit 噪声会污染 teacher logits（见 docs §9.4）
#   * A6 使用确定性 RLDS 顺序（TFDS 分片不 shuffle、读/map 各 1 个 worker）
#   * --shuffle_buffer_size 必须为 1、不要 --image_aug：否则缓存与数据集错位
#   * --resume 按 manifest 中实际落盘行数续写，可以更换 batch size
#   * 旧版缓存（CACHE_VERSION/data_order 不匹配）会自动备份为 *.legacy_时间戳 后重导
#   * 完整缓存可跨 epoch 回绕并在每次回绕时重新校验 GT token；--limit 禁止回绕
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
TRAIN_BS=4
TRAIN_GA=4
MAX_STEPS=20000
SAVE_STEPS=2000
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
    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'TIP'

参数:
  --dataset NAME          OXE 数据集/混合名（默认 bridge_orig）
  --data-root DIR         RLDS 数据根目录（默认 <repo>/datasets/open-x-embodiment）
  --teacher PATH          teacher 权重目录（默认 <repo>/openvla/openvla-7b）
  --cache DIR             缓存目录（默认 <repo>/caches/<dataset>_teacher）
  --gpu 0|N|auto|all      使用哪张卡（默认 0）
  --batch-size N          导出 batch（默认 2；OOM 自动降 1）
  --limit N               只导出前 N 条（调试用）
  --train-batch-size N    训练 batch（默认 4）
  --grad-accum N          梯度累积（默认 4）
  --max-steps N           训练步数（默认 20000）
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
        --limit)            LIMIT="$2"; shift 2 ;;
        --train-batch-size) TRAIN_BS="$2"; shift 2 ;;
        --grad-accum)       TRAIN_GA="$2"; shift 2 ;;
        --max-steps)        MAX_STEPS="$2"; shift 2 ;;
        --python)           PY="$2"; shift 2 ;;
        --env-name)         ENV_NAME="$2"; shift 2 ;;
        --train-arg)        PY_EXTRA+=("$2"); shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          MODE="help"; shift ;;
        *) die "未知参数: $1（--help 查看用法）" ;;
    esac
done

[[ "${CACHE}" == "" ]] && CACHE="${SCRIPT_DIR}/caches/${DATASET}_teacher"

if [[ "${MODE}" == "help" ]]; then usage; exit 0; fi
case "${MODE}" in export|check|train|all) ;; *) die "第一个参数必须是 export|check|train|all，收到: ${MODE}" ;; esac

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

do_train() {
    local quant=(--quantize_4bit)
    if [[ "${DRY_RUN}" == "0" ]] && ! "${PY}" -c "import bitsandbytes" >/dev/null 2>&1; then
        warn "bitsandbytes 不可用：退回 bf16 LoRA，并把 batch 降到 1 / accum 提到 16"
        quant=(); TRAIN_BS=1; TRAIN_GA=16
    fi
    step "3. 蒸馏训练（bs=${TRAIN_BS} x accum ${TRAIN_GA}，$( (( ${#quant[@]} > 0 )) && echo 4-bit-QLoRA || echo bf16-LoRA)）"
    local limit_args=()
    (( LIMIT > 0 )) && limit_args=(--limit "${LIMIT}")
    local -a cmd=("${PY}" scripts/train_btkd_student.py
        --cache_dir "${CACHE}"
        --dataset_name "${DATASET}"
        --data_root_dir "${DATA_ROOT}"
        --vla_path "${TEACHER}"
        "${limit_args[@]}"
        --shuffle_buffer_size 1
        --batch_size "${TRAIN_BS}"
        --grad_accumulation_steps "${TRAIN_GA}"
        "${quant[@]}"
        --max_steps "${MAX_STEPS}"
        --save_steps "${SAVE_STEPS}"
        --device cuda:0
        "${PY_EXTRA[@]}")
    exec_cmd "${cmd[@]}"
    if [[ "${DRY_RUN}" == "0" ]]; then ok "训练完成，检查 runs_btkd/ 下的 checkpoint"; fi
}

case "${MODE}" in
    export) do_export ;;
    check)  do_check ;;
    train)  do_train ;;
    all)    do_export; do_check; do_train ;;
esac

echo
printf "日志: %s\n" "${LOG_FILE}"
echo "提示：单卡 24GB 想再省显存 → --batch-size 1 / 训练加 --train-arg '--lora_rank 16'"
