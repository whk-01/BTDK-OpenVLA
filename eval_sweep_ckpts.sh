#!/usr/bin/env bash
# =============================================================================
# 扫 4 个 checkpoint 的离线验证（固定通过 run_a6_single_gpu.sh 调用，GPU 默认 2）
#
#   bash eval_sweep_ckpts.sh                    # 默认 GPU=2，扫 2000/8000/14000/18000
#   bash eval_sweep_ckpts.sh --gpu 1            # 换卡
#   bash eval_sweep_ckpts.sh --limit 128        # 更快
#   bash eval_sweep_ckpts.sh 2000 8000          # 只扫指定 checkpoint
#   bash eval_sweep_ckpts.sh --dry-run          # 只打印将执行的命令
#
# 关键点：run_a6_single_gpu.sh 内部会 `export CUDA_VISIBLE_DEVICES=<--gpu>`，
#        所以必须用 --gpu 传参，手动 export 会被脚本覆盖。
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU="${GPU:-2}"
DATASET="${DATASET:-libero_spatial_no_noops}"
CACHE="${CACHE:-}"
if [[ -z "${CACHE}" ]]; then
    # 优先用你已经导出的 caches/libero_spatial_teacher
    for cand in "${SCRIPT_DIR}/caches/libero_spatial_teacher" "${SCRIPT_DIR}/caches/${DATASET}_teacher"; do
        if [[ -d "${cand}" ]]; then CACHE="${cand}"; break; fi
    done
    CACHE="${CACHE:-${SCRIPT_DIR}/caches/${DATASET}_teacher}"
fi
EXP="${EXP:-${SCRIPT_DIR}/runs_btkd/btkd+openvla-7b+${DATASET}+b16+lr0.0005+T4.0+lam0.5}"
ENV_NAME="${ENV_NAME:-whk_BTKD}"
LIMIT="${LIMIT:-512}"
EVAL_BS="${EVAL_BS:-1}"
DRY_RUN=0
CKPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2"; shift 2 ;;
        --limit)      LIMIT="$2"; shift 2 ;;
        --batch-size) EVAL_BS="$2"; shift 2 ;;
        --cache)      CACHE="$2"; shift 2 ;;
        --exp)        EXP="$2"; shift 2 ;;
        --env-name)   ENV_NAME="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)           echo "未知参数: $1" >&2; exit 2 ;;
        *)            CKPTS+=("$1"); shift ;;
    esac
done
[[ ${#CKPTS[@]} -eq 0 ]] && CKPTS=(2000 8000 14000 18000)

LOG_DIR="${SCRIPT_DIR}/logs"; mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY_TSV="${LOG_DIR}/eval_sweep_${TS}.tsv"
: > "${SUMMARY_TSV}"

echo "GPU=${GPU}  DATASET=${DATASET}  LIMIT=${LIMIT}  EVAL_BS=${EVAL_BS}"
echo "CACHE=${CACHE}"
echo "EXP=${EXP}"
echo "checkpoints: ${CKPTS[*]}"

if [[ "${DRY_RUN}" == "0" ]]; then
    # 空卡检查：7B bf16 推理至少要 ~19 GB 空闲
    FREE_MIB="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU${GPU} 空闲显存: ${FREE_MIB} MiB"
    if [[ "${FREE_MIB}" -lt 19000 ]]; then
        echo "[WARN] 空闲不足 19000 MiB，7B bf16 很可能 OOM；换卡或等训练结束。" >&2
    fi
    [[ -f "${CACHE}/manifest.json" ]] || { echo "[FAIL] 找不到缓存: ${CACHE}/manifest.json" >&2; exit 1; }
fi

declare -a OK_LIST=() FAIL_LIST=()

for s in "${CKPTS[@]}"; do
    ck="${EXP}/checkpoint-${s}"
    log="${LOG_DIR}/eval_ckpt-${s}_${TS}.log"

    echo
    echo "===================== checkpoint-${s} ====================="

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run] bash run_a6_single_gpu.sh eval --dataset %q --cache %q --adapter %q --gpu %q --env-name %q --limit %q --eval-batch-size %q\n' \
            "${DATASET}" "${CACHE}" "${ck}" "${GPU}" "${ENV_NAME}" "${LIMIT}" "${EVAL_BS}"
        continue
    fi

    if [[ ! -f "${ck}/adapter_config.json" ]]; then
        echo "[SKIP] 不是有效的 adapter 目录: ${ck}"
        FAIL_LIST+=("${s}(missing)")
        continue
    fi

    if bash run_a6_single_gpu.sh eval \
            --dataset "${DATASET}" \
            --cache "${CACHE}" \
            --adapter "${ck}" \
            --gpu "${GPU}" \
            --env-name "${ENV_NAME}" \
            --limit "${LIMIT}" \
            --eval-batch-size "${EVAL_BS}" 2>&1 | tee "${log}"
    then
        OK_LIST+=("${s}")
    else
        echo "[FAIL] checkpoint-${s} 验证失败，日志: ${log}" >&2
        FAIL_LIST+=("${s}(error)")
        continue
    fi

    # 从日志里抽出 [eval] summary 后面的 JSON
    python - "${s}" "${log}" "${SUMMARY_TSV}" <<'PY'
import json, sys

ckpt, log_path, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log_path, encoding="utf-8", errors="replace").read()
i = text.rfind("[eval] summary")
if i < 0:
    print(f"[WARN] checkpoint-{ckpt}: 日志里没有 [eval] summary")
    sys.exit(0)
j = text.find("{", i)
depth, end = 0, None
for k in range(j, len(text)):
    if text[k] == "{":
        depth += 1
    elif text[k] == "}":
        depth -= 1
        if depth == 0:
            end = k + 1
            break
if end is None:
    print(f"[WARN] checkpoint-{ckpt}: JSON 不完整")
    sys.exit(0)

d = json.loads(text[j:end])
mean6 = sum(d["per_dim_acc"][:6]) / 6.0
with open(tsv, "a", encoding="utf-8") as f:
    f.write("\t".join([
        str(ckpt),
        f'{d["student_token_acc"]:.4f}',
        f'{mean6:.4f}',
        f'{d["teacher_token_acc"]:.4f}',
        f'{d["student_teacher_agreement"]:.4f}',
        f'{d["ce"]:.4f}',
        " ".join(f"{x:.3f}" for x in d["per_dim_acc"]),
    ]) + "\n")
print(f'[ckpt {ckpt}] student={d["student_token_acc"]:.4f}  '
      f'dim0-5={mean6:.4f}  teacher={d["teacher_token_acc"]:.4f}  '
      f'agreement={d["student_teacher_agreement"]:.4f}')
PY
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo; echo "[dry-run] 未真正执行。"
    exit 0
fi

echo
echo "===================== 汇总 ====================="
printf '%-8s %-10s %-10s %-10s %-11s %-8s\n' "ckpt" "student" "dim0-5" "teacher" "agreement" "ce"
while IFS=$'\t' read -r a b c d e f g; do
    printf '%-8s %-10s %-10s %-10s %-11s %-8s\n' "$a" "$b" "$c" "$d" "$e" "$f"
done < "${SUMMARY_TSV}"
echo
echo "明细: ${SUMMARY_TSV}"
echo "成功: ${OK_LIST[*]:-无}"
echo "失败: ${FAIL_LIST[*]:-无}"
echo
echo "判读:"
echo "  - 2000 最高、之后单调下降  -> hard 调度过早（加 --easy_hold_frac）"
echo "  - 一路上升且 18000 还在涨  -> 欠训练（补到 >=1 epoch）"
echo "  - 四个都 ≈ base 基线      -> LoRA 未生效，先查 base 对照"
