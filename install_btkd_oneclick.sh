#!/usr/bin/env bash
# =============================================================================
# BTKD + OpenVLA 一键部署（单文件、幂等、国内网络友好）
#
#   bash install_btkd_oneclick.sh
#       → 自动：conda 求解器切 libmamba → 建/修环境 → 装 PyTorch(cu121/cu124)
#               → 装 pip 依赖 → 克隆并安装 OpenVLA → 全链路自检 → A6 冒烟测试
#
# 常用
#   bash install_btkd_oneclick.sh --check                 # 只体检，不装任何东西
#   bash install_btkd_oneclick.sh --dry-run                # 只打印将要执行的命令
#   bash install_btkd_oneclick.sh --env-name whk_BTKD      # 指定环境名
#   bash install_btkd_oneclick.sh --recreate               # 删掉重建（危险）
#   bash install_btkd_oneclick.sh --torch-cuda 12.4        # 改用 cu124（默认 cu121）
#   bash install_btkd_oneclick.sh --download-model         # 额外下载 openvla-7b（约 15GB）
#
# 为什么网络部分这么写
#   * conda classic solver 在 pytorch+nvidia 双 channel 上会指数级回溯
#     （日志 "unsuccessful initial attempt using frozen solve" 就是它），常见几十分钟；
#     libmamba 3-10 秒解完。脚本会装插件并把 conda config 的默认 solver 也改掉。
#   * 直连 conda.anaconda.org 在部分网络会 TLS 中断（CondaSSLError: SSOLEOFError）。
#     默认走 pip 安装 torch，不依赖 nvidia conda 通道；若显式用 --conda-torch，
#     则使用清华 pytorch/main + 可选 nvidia 镜像（浙大/官方），并且必须 --override-channels，
#     否则 conda 仍会去 anaconda.org 取 repodata。
#   * torch 默认用 pip 安装：从 PyTorch 官方 cu121/cu124 wheel 索引（或国内镜像）
#     下载，wheel 自带 CUDA 运行时，不依赖 nvidia conda 通道；不会静默改用
#     PyPI 上 CUDA 版本不同的轮子。conda 模式保留给能访问 nvidia 通道的网络。
#   * 若发现 pip 侧残留 torch（版本/CUDA 不符），先卸载再重装。
#
# 这个脚本只负责"把环境装对"：不探测显卡、不设 CUDA_VISIBLE_DEVICES、不估算显存、
# 不管训练参数，也不关心机器上有几张卡。
# =============================================================================
set -Eeuo pipefail

# ------------------------------ 默认参数 -------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${BTKD_ENV_NAME:-BTKD}"
PY_VER="3.10"
TORCH_VER="${BTKD_TORCH_VERSION:-2.5.1}"
TV_VER="0.20.1"
TORCHAUDIO_VER="${BTKD_TORCHAUDIO_VERSION:-2.5.1}"
TORCH_CUDA="${BTKD_TORCH_CUDA:-12.1}"          # 12.1 | 12.4（按需手动指定）
TORCH_INDEX_OVERRIDE="${BTKD_TORCH_INDEX:-}"   # 可选：自定义 PyTorch wheel 索引
OPENVLA_DIR="${SCRIPT_DIR}/openvla"
OPENVLA_REPO="${BTKD_OPENVLA_REPO:-https://github.com/openvla/openvla.git}"
PIP_INDEX="${BTKD_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_ENDPOINT="${BTKD_HF_ENDPOINT:-https://hf-mirror.com}"

PYTORCH_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch"
MAIN_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
CONDA_FORGE_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
# 注意：清华/中科大/南大/北外的 anaconda/cloud 都【没有】nvidia 目录（我查过索引页），
# 所以 conda 模式下只有下面两个候选；默认走 pip，根本不需要这个通道。
NVIDIA_MIRROR_CANDIDATES=(
    "https://mirrors.zju.edu.cn/anaconda/cloud/nvidia"
    "https://conda.anaconda.org/nvidia"
)
NVIDIA_MIRROR=""
TORCH_INSTALL_MODE="${BTKD_TORCH_MODE:-auto}"   # auto | conda | pip

CHECK_ONLY=0; DRY_RUN=0; RECREATE=0; FORCE=0
SKIP_OPENVLA=0; SKIP_TESTS=0; WITH_BNB=1; DOWNLOAD_MODEL=0
CONDA_CONFIG_SOLVER=1
GIT_TIMEOUT="${BTKD_GIT_TIMEOUT:-600}"       # 每个 clone 源的最长等待秒数

LOG_DIR="${SCRIPT_DIR}/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/oneclick_$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="${LOG_DIR}/env_report_$(date +%Y%m%d_%H%M%S).txt"

C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_INFO=$'\033[36m'; C_OFF=$'\033[0m'
step() { printf "\n${C_INFO}==== %s ====${C_OFF}\n" "$*"; }
ok()   { printf "${C_OK}[ OK ]${C_OFF} %s\n" "$*"; }
warn() { printf "${C_WARN}[WARN]${C_OFF} %s\n" "$*"; }
die()  { printf "${C_ERR}[FAIL]${C_OFF} %s\n" "$*" >&2; exit 1; }
trap 'die "脚本在第 $LINENO 行失败；完整日志: ${LOG_FILE}"' ERR

exec > >(tee -a "${LOG_FILE}") 2>&1

usage() {
    sed -n '4,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'TIP'

附加参数:
  --env-name NAME         conda 环境名（默认 BTKD）
  --torch-cuda 12.1|12.4  PyTorch 的 CUDA 版本（默认 12.1，兼容驱动 >=530）
  --openvla-src DIR       已有 openvla 源码目录（默认 <repo>/openvla）
  --openvla-repo URL      openvla git 地址（默认官方 GitHub）
  --pip-index URL         pip 源（默认清华）
  --hf-endpoint URL       HF 镜像（默认 hf-mirror.com）
  --nvidia-mirror URL     强制指定 nvidia conda 通道
  --torch-index URL       强制指定 PyTorch wheel 索引（需含 cu121/cu124 wheel）
  --pip-torch             用 pip 装 torch（默认；wheel 自带 CUDA，不依赖 nvidia 通道）
  --conda-torch           改用 conda 装 torch（需要能访问 nvidia 通道）
  --git-timeout SEC       单个 OpenVLA 克隆源超时（默认 600 秒）
  --no-conda-config       不改写 conda 全局 solver 配置（只在本脚本命令里生效）
  --check                 只体检 + 自检，不安装
  --dry-run               只打印命令，不执行
  --recreate              删除同名环境后重建（危险）
  --skip-openvla          不克隆/安装 openvla
  --skip-tests            跳过单元测试与冒烟测试
  --no-bitsandbytes       不装 bitsandbytes（不启用 QLoRA）
  --download-model        额外下载 openvla-7b 权重到 ./openvla/openvla-7b
  --force                 磁盘告警也继续
  -h, --help              显示帮助
TIP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)        ENV_NAME="$2"; shift 2 ;;
        --torch-cuda)      TORCH_CUDA="$2"; shift 2 ;;
        --openvla-src)     OPENVLA_DIR="$2"; shift 2 ;;
        --openvla-repo)    OPENVLA_REPO="$2"; shift 2 ;;
        --pip-index)       PIP_INDEX="$2"; shift 2 ;;
        --hf-endpoint)     HF_ENDPOINT="$2"; shift 2 ;;
        --nvidia-mirror)   NVIDIA_MIRROR="$2"; shift 2 ;;
        --torch-index)     TORCH_INDEX_OVERRIDE="$2"; shift 2 ;;
        --git-timeout)     GIT_TIMEOUT="$2"; shift 2 ;;
        --pip-torch)       TORCH_INSTALL_MODE="pip"; shift ;;
        --conda-torch)     TORCH_INSTALL_MODE="conda"; shift ;;
        --no-conda-config) CONDA_CONFIG_SOLVER=0; shift ;;
        --check)           CHECK_ONLY=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --recreate)        RECREATE=1; shift ;;
        --skip-openvla)    SKIP_OPENVLA=1; shift ;;
        --skip-tests)      SKIP_TESTS=1; shift ;;
        --no-bitsandbytes) WITH_BNB=0; shift ;;
        --download-model)  DOWNLOAD_MODEL=1; shift ;;
        --force)           FORCE=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) die "未知参数: $1（--help 查看用法）" ;;
    esac
done

case "${TORCH_INSTALL_MODE}" in
    auto)
        # conda 的 pytorch-cuda 依赖 nvidia 通道，而国内镜像普遍没有同步它；
        # pip 的 torch wheel 自带 CUDA 运行时，不需要任何额外通道，所以默认走 pip。
        TORCH_INSTALL_MODE="pip"
        ;;
    pip|conda) ;;
    *) die "BTKD_TORCH_MODE/--pip-torch/--conda-torch 只支持 pip 或 conda，收到: ${TORCH_INSTALL_MODE}" ;;
esac

[[ "${TORCH_CUDA}" =~ ^(12\.1|12\.4)$ ]] || die "--torch-cuda 只支持 12.1/12.4，收到: ${TORCH_CUDA}"
[[ "${PY_VER}" == "3.10" ]] || die "当前只支持 Python 3.10（OpenVLA/TF 2.15 组合），收到: ${PY_VER}"

exec_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
        return 0
    fi
    "$@"
}

retry() {  # retry <次数> <命令...>
    local n="$1" i; shift
    for ((i=1; i<=n; i++)); do
        "$@" && return 0
        (( i < n )) || break
        warn "命令失败（第 ${i}/${n} 次），$((i*5)) 秒后重试: $*"
        sleep $((i*5))
    done
    return 1
}

# ---------------------------------------------------------------------------
step "0/8 参数"
ok "环境名: ${ENV_NAME}   仓库: ${SCRIPT_DIR}"
ok "PyTorch ${TORCH_VER} + CUDA ${TORCH_CUDA}   pip 源: ${PIP_INDEX}"
if [[ "${DRY_RUN}" == "1" ]]; then warn "dry-run 模式：只打印命令，不实际执行"; fi

# ---------------------------------------------------------------------------
step "1/8 预检"

find_conda() {
    if command -v conda >/dev/null 2>&1; then command -v conda; return; fi
    local c
    for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
             "$HOME/miniforge3/bin/conda" /opt/conda/bin/conda /opt/miniconda3/bin/conda; do
        [[ -x "$c" ]] && { echo "$c"; return; }
    done
    return 1
}

CONDA_BIN="$(find_conda)" || die "找不到 conda，请先安装 Miniconda/Anaconda 并确保可执行"
CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
CONDA_VER="$("${CONDA_BIN}" --version | awk '{print $2}')"
ok "conda: ${CONDA_BIN} (${CONDA_VER})"

command -v git >/dev/null 2>&1 || die "找不到 git：sudo apt install -y git"
ok "git: $(git --version)"

FREE_KB="$(df -Pk "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')"
FREE_GB=$(( FREE_KB / 1024 / 1024 ))
NEED_GB=25
[[ "${DOWNLOAD_MODEL}" == "1" ]] && NEED_GB=45
if (( FREE_GB < NEED_GB )); then
    warn "磁盘可用 ${FREE_GB} GB < 建议 ${NEED_GB} GB（环境约 25GB，权重另需 ~15GB）"
    if (( FREE_GB < 5 )); then
        (( FORCE == 1 )) || die "磁盘只剩 ${FREE_GB} GB，装不下；确认继续请加 --force"
    fi
else
    ok "磁盘可用: ${FREE_GB} GB"
fi

# 不探测显卡/驱动；如需 cu124（新驱动）显式传 --torch-cuda 12.4。
ok "目标: PyTorch ${TORCH_VER} + CUDA ${TORCH_CUDA}"

# ---------------------------------------------------------------------------
step "2/8 conda 求解器与镜像通道"

have_libmamba() {
    "${CONDA_BIN}" list -n base 2>/dev/null \
        | awk '$1 == "conda-libmamba-solver" {found=1} END {exit(found ? 0 : 1)}'
}
solver_supported() {
    "${CONDA_BIN}" install --help 2>/dev/null | grep -q -- '--solver'
}

CONDA_SOLVER_ARGS=()
if [[ "${CHECK_ONLY}" == "0" ]]; then
    if ! solver_supported; then
        warn "conda ${CONDA_VER} 太旧，不支持 libmamba 插件（建议 >=23.5）"
        warn "可手动升级 base：conda install -n base -c ${MAIN_MIRROR} conda"
    else
        if ! have_libmamba; then
            warn "未装 conda-libmamba-solver，正在从清华 conda-forge 镜像安装"
            exec_cmd "${CONDA_BIN}" install -y -n base --solver=classic \
                --override-channels -c "${CONDA_FORGE_MIRROR}" conda-libmamba-solver \
                || warn "libmamba 安装失败，将回退 classic solver（会很慢）"
        fi
        if have_libmamba; then
            CONDA_SOLVER_ARGS=(--solver=libmamba)
            if [[ "${CONDA_CONFIG_SOLVER}" == "1" ]]; then
                exec_cmd "${CONDA_BIN}" config --set solver libmamba || true
                ok "conda 默认 solver 已设为 libmamba（写入 ~/.condarc）"
            else
                ok "conda solver: libmamba（仅本脚本命令生效）"
            fi
        fi
    fi
fi

probe() {  # probe <url> → 0 表示可访问（要求 HTTP 200/206，重试 3 次）
    local url="$1" code i
    for i in 1 2 3; do
        if command -v curl >/dev/null 2>&1; then
            code="$(curl -sS -o /dev/null -w '%{http_code}' -m 25 -L \
                    -r 0-1023 "$url" 2>/dev/null || echo 000)"
            case "${code}" in 200|206) return 0 ;; esac
        else
            if "${CONDA_BIN}" run -n base --no-capture-output python - "$url" <<'PY' 2>/dev/null
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], headers={"Range": "bytes=0-1023"})
urllib.request.urlopen(req, timeout=25).read(1024)
PY
            then return 0; fi
        fi
        sleep 3
    done
    return 1
}

if [[ "${TORCH_INSTALL_MODE}" != "pip" ]]; then
    if [[ -n "${NVIDIA_MIRROR}" ]]; then
        probe "${NVIDIA_MIRROR}/linux-64/repodata.json" \
            || die "指定的 nvidia conda 通道不可用: ${NVIDIA_MIRROR}"
    else
        for m in "${NVIDIA_MIRROR_CANDIDATES[@]}"; do
            if probe "${m}/linux-64/repodata.json"; then NVIDIA_MIRROR="$m"; break; fi
            warn "镜像不可用: ${m}"
        done
    fi
    if [[ -z "${NVIDIA_MIRROR}" ]]; then
        if [[ "${TORCH_INSTALL_MODE}" == "conda" ]]; then
            die "所有 nvidia conda 通道都不可用；用 --nvidia-mirror 指定，或改用 --pip-torch"
        fi
        warn "所有 nvidia conda 通道都不可用 → 自动改用 pip 安装 PyTorch（不依赖 nvidia channel）"
        TORCH_INSTALL_MODE="pip"
    fi
fi

ok "conda 通道: pytorch=${PYTORCH_MIRROR}"
ok "            main   =${MAIN_MIRROR}"
if [[ "${TORCH_INSTALL_MODE}" == "pip" ]]; then
    ok "            nvidia = 不使用（torch 走 pip wheel，CUDA 运行时随包自带）"
    CONDA_CH_ARGS=(--override-channels -c "${PYTORCH_MIRROR}" -c "${MAIN_MIRROR}")
else
    ok "            nvidia =${NVIDIA_MIRROR}"
    CONDA_CH_ARGS=(--override-channels -c "${PYTORCH_MIRROR}" -c "${NVIDIA_MIRROR}" -c "${MAIN_MIRROR}")
fi
CONDA_MAIN_ARGS=(--override-channels -c "${MAIN_MIRROR}")

# ---------------------------------------------------------------------------
step "3/8 conda 环境: ${ENV_NAME}"

env_exists() {
    "${CONDA_BIN}" env list \
        | awk -v name="${ENV_NAME}" 'NF && $1 !~ /^#/ && $1 == name {found=1} END {exit(found ? 0 : 1)}'
}

if [[ "${CHECK_ONLY}" == "1" ]]; then
    env_exists || die "环境 ${ENV_NAME} 不存在；去掉 --check 后重新执行"
elif env_exists && [[ "${RECREATE}" == "1" ]]; then
    warn "--recreate：删除环境 ${ENV_NAME}"
    exec_cmd "${CONDA_BIN}" env remove -y -n "${ENV_NAME}" || true
    exec_cmd "${CONDA_BIN}" create "${CONDA_SOLVER_ARGS[@]}" -y -n "${ENV_NAME}" \
        "${CONDA_MAIN_ARGS[@]}" "python=${PY_VER}" pip
elif env_exists; then
    ok "环境已存在，复用（如需重装加 --recreate）"
else
    exec_cmd "${CONDA_BIN}" create "${CONDA_SOLVER_ARGS[@]}" -y -n "${ENV_NAME}" \
        "${CONDA_MAIN_ARGS[@]}" "python=${PY_VER}" pip
fi

get_prefix() {
    local p
    p="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys;print(sys.prefix)' 2>/dev/null | tail -n 1 || true)"
    [[ -n "${p}" && -x "${p}/bin/python" ]] || p=""
    [[ -z "${p}" ]] && p="${CONDA_BASE}/envs/${ENV_NAME}"
    [[ -x "${p}/bin/python" ]] && echo "${p}"
}

ENV_PREFIX="$(get_prefix || true)"
if [[ -z "${ENV_PREFIX}" ]]; then
    warn "环境里没有可用的 python（空环境/损坏环境），补装 python=${PY_VER} + pip"
    exec_cmd "${CONDA_BIN}" install "${CONDA_SOLVER_ARGS[@]}" -y -n "${ENV_NAME}" \
        "${CONDA_MAIN_ARGS[@]}" "python=${PY_VER}" pip
    ENV_PREFIX="$(get_prefix || true)"
fi
if [[ "${DRY_RUN}" == "1" && -z "${ENV_PREFIX}" ]]; then
    ENV_PREFIX="<env-prefix>"
fi
[[ -n "${ENV_PREFIX}" ]] || die "激活 ${ENV_NAME} 后仍找不到 python；可尝试 --recreate"

PY="${ENV_PREFIX}/bin/python"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export HF_ENDPOINT
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=60
unset PYTHONHOME PYTHONPATH || true

if [[ "${DRY_RUN}" == "0" ]]; then
    ok "python: $("${PY}" -V 2>&1) @ ${PY}"
fi

PIP_FLAGS=(--no-input --no-warn-script-location --disable-pip-version-check
           --retries 5 --timeout 60 --index-url "${PIP_INDEX}"
           --trusted-host pypi.tuna.tsinghua.edu.cn)
pip_install() { exec_cmd "${PY}" -m pip install "${PIP_FLAGS[@]}" "$@"; }

# ---------------------------------------------------------------------------
step "4/8 PyTorch ${TORCH_VER} + CUDA ${TORCH_CUDA}（${TORCH_INSTALL_MODE}）"

torch_ok() {
    if [[ "${DRY_RUN}" == "1" ]]; then return 1; fi
    "${PY}" - "$TORCH_VER" "$TORCH_CUDA" <<'PY' >/dev/null 2>&1
import sys, torch
want_ver, want_cuda = sys.argv[1], sys.argv[2]
ok_ver = torch.__version__.split("+")[0] == want_ver
cuda = torch.version.cuda or ""
ok_cuda = cuda == want_cuda
sys.exit(0 if ok_ver and ok_cuda else 1)
PY
}

# conda 装 torch + 固定 OpenMP/MKL（TF 与 torch 抢库是已知坑）
install_torch_conda() {
    exec_cmd "${CONDA_BIN}" install "${CONDA_SOLVER_ARGS[@]}" -y -n "${ENV_NAME}" \
        "${CONDA_CH_ARGS[@]}" \
        "python=${PY_VER}" \
        "pytorch=${TORCH_VER}" "torchvision=${TV_VER}" "torchaudio=${TORCHAUDIO_VER}" \
        "pytorch-cuda=${TORCH_CUDA}" "cuda-version=${TORCH_CUDA}" \
        numpy=1.26.4 || return 1
    exec_cmd "${CONDA_BIN}" install "${CONDA_SOLVER_ARGS[@]}" -y -n "${ENV_NAME}" \
        "${CONDA_MAIN_ARGS[@]}" \
        mkl=2023.1.0 intel-openmp=2023.1.0 "llvm-openmp<16" numpy=1.26.4
}

install_torch_pip() {
    local cu_tag="${TORCH_CUDA/./}" idx
    local local_ver="${TORCH_VER}+cu${cu_tag}"
    local -a indexes=()
    if [[ -n "${TORCH_INDEX_OVERRIDE}" ]]; then
        indexes=("${TORCH_INDEX_OVERRIDE%/}")
    else
        indexes=(
            "https://download.pytorch.org/whl/cu${cu_tag}"
            "https://mirror.sjtu.edu.cn/pytorch-wheels/cu${cu_tag}"
            "https://mirrors.nju.edu.cn/pytorch/whl/cu${cu_tag}"
        )
    fi

    for idx in "${indexes[@]}"; do
        if [[ "${DRY_RUN}" == "0" ]] && ! probe "${idx}/torch/"; then
            warn "PyTorch wheel 索引不可用: ${idx}"
            continue
        fi
        echo "      尝试 PyTorch wheel 索引: ${idx}"
        if exec_cmd "${PY}" -m pip install --no-input --disable-pip-version-check \
               --retries 5 --timeout 120 --index-url "${idx}" \
               "torch==${local_ver}" "torchvision==${TV_VER}+cu${cu_tag}" \
               "torchaudio==${TORCHAUDIO_VER}+cu${cu_tag}"; then
            return 0
        fi
        warn "安装失败，换下一个索引: ${idx}"
    done
    warn "所有 PyTorch cu${cu_tag} 索引均失败；不会回退到 CUDA 版本不确定的 PyPI torch"
    return 1
}

if [[ "${CHECK_ONLY}" == "1" ]]; then
    torch_ok && ok "torch 已就绪" || warn "torch 未就绪/版本不符"
else
    # pip 装的 torch 会盖住 conda 的 cu121（OpenVLA 依赖、torchaudio 都可能带进来）
    if torch_ok; then
        ok "torch 已就绪，跳过安装"
    else
        if [[ "${DRY_RUN}" == "0" ]] && "${PY}" -m pip show torch >/dev/null 2>&1; then
            warn "检测到 pip 侧的 torch（版本/CUDA 不符），先卸载以免它盖住 conda 版本"
            "${PY}" -m pip uninstall -y torch torchvision torchaudio torchtext || true
        fi
        echo "      安装方式: ${TORCH_INSTALL_MODE}（首次下载约 2-2.5 GB，耗时视带宽 5-20 分钟）"
        if [[ "${TORCH_INSTALL_MODE}" == "pip" ]]; then
            install_torch_pip || die "PyTorch cu${TORCH_CUDA/./} 安装失败；检查网络，或用 --torch-index 指定可用索引"
        elif ! install_torch_conda; then
            warn "conda 安装 torch 失败（nvidia 通道不稳）→ 回退到 pip"
            TORCH_INSTALL_MODE="pip"
            install_torch_pip || die "conda/pip 两种方式都未能装上 PyTorch cu${TORCH_CUDA/./}"
        fi
        if [[ "${TORCH_INSTALL_MODE}" == "pip" ]]; then
            warn "pip 模式：TF 2.15 与 torch 各带一份 libiomp5，若报 OMP Error #15 请 export KMP_DUPLICATE_LIB_OK=TRUE"
        fi
        if [[ "${DRY_RUN}" == "1" ]] || torch_ok; then
            ok "PyTorch 安装完成"
        else
            warn "torch 自检未通过，pip 依赖装完后会再修一次"
        fi
    fi
fi

# ---------------------------------------------------------------------------
step "5/8 pip 依赖"

if [[ "${CHECK_ONLY}" == "0" ]]; then
    pip_install "setuptools>=61" wheel "pip>=23"

    pip_install \
        "timm==0.9.10" \
        "transformers==4.40.1" \
        "tokenizers==0.19.1" \
        "sentencepiece==0.1.99" \
        "peft==0.11.1" \
        "accelerate==1.15.0" \
        "draccus==0.8.0" \
        "huggingface_hub==0.36.2" \
        "numpy==1.26.4" \
        "matplotlib==3.9.2" \
        einops json-numpy jsonlines rich tqdm pillow

    pip_install \
        "tensorflow==2.15.0" \
        "tensorflow_datasets==4.9.3" \
        "tensorflow_graphics==2021.12.3" \
        "tensorflow-metadata==1.15.0"

    if [[ "${DRY_RUN}" != "1" ]] && "${PY}" -c "import dlimp" >/dev/null 2>&1; then
        ok "dlimp 已安装"
    else
        DLIMP_OK=0
        for pre in "" "https://ghfast.top/" "https://gh-proxy.com/"; do
            url="git+${pre}https://github.com/moojink/dlimp_openvla"
            if pip_install "dlimp @ ${url}"; then DLIMP_OK=1; break; fi
            warn "dlimp 安装失败：${url}"
        done
        (( DLIMP_OK == 1 )) || warn "dlimp 未装上；OpenVLA 的 RLDS 训练会用到它"
    fi

    # 兜底：防止被上游依赖顺带升级
    pip_install "protobuf==3.20.3" "numpy==1.26.4" "wandb==0.17.8"

    if [[ "${WITH_BNB}" == "1" ]]; then
        pip_install "bitsandbytes>=0.43.1" \
            || warn "bitsandbytes 失败：bf16 LoRA 仍可用，QLoRA(--quantize_4bit) 不可用"
    fi
    # pip 依赖里若有谁把 torch 换成了不兼容的轮子（CPU/其他版本），这里再修一次。
    if [[ "${DRY_RUN}" == "0" ]] && ! torch_ok; then
        warn "pip 依赖后 torch 版本/CUDA 不符，卸载后按 ${TORCH_INSTALL_MODE} 方式重装"
        "${PY}" -m pip uninstall -y torch torchvision torchaudio torchtext || true
        if [[ "${TORCH_INSTALL_MODE}" == "pip" ]]; then
            install_torch_pip || die "pip 依赖后 torch 修复失败；检查 PyTorch wheel 索引网络"
        else
            install_torch_conda || {
                warn "conda 修复失败，回退到 pip"
                TORCH_INSTALL_MODE="pip"
                install_torch_pip || die "pip 依赖后 torch 修复失败；检查 PyTorch wheel 索引网络"
            }
        fi
        torch_ok && ok "torch 已修复" || warn "torch 仍未修复，见第 7 步自检"
    fi
    ok "pip 依赖完成"
fi

# ---------------------------------------------------------------------------
step "6/8 OpenVLA"

if [[ "${CHECK_ONLY}" == "0" && "${SKIP_OPENVLA}" == "0" ]]; then
    if [[ -f "${OPENVLA_DIR}/pyproject.toml" ]]; then
        ok "源码已存在: ${OPENVLA_DIR}"
    else
        # 目录存在但不是 openvla 源码：绝不删除用户目录
        if [[ -d "${OPENVLA_DIR}" ]] && [[ -n "$(ls -A "${OPENVLA_DIR}" 2>/dev/null || true)" ]]; then
            die "${OPENVLA_DIR} 存在但不像 openvla 源码（无 pyproject.toml）；请换 --openvla-src 或先自行清理"
        fi
        GIT_CANDIDATES=(
            "${OPENVLA_REPO}"
            "https://ghfast.top/${OPENVLA_REPO}"
            "https://gh-proxy.com/${OPENVLA_REPO}"
            "https://gitclone.com/${OPENVLA_REPO#https://}"
        )
        CLONED=0
        CLONE_TMP=""
        for u in "${GIT_CANDIDATES[@]}"; do
            echo "      尝试: ${u}"
            CLONE_TMP="${OPENVLA_DIR}.clone.$$"
            rm -rf "${CLONE_TMP}"
            # 卡住的代理会让 clone 一直挂着：用 git 低速中断 + 硬超时兜底
            if exec_cmd env GIT_TERMINAL_PROMPT=0 GIT_HTTP_LOW_SPEED_LIMIT=1024 \
                    GIT_HTTP_LOW_SPEED_TIME=60 \
                    timeout "${GIT_TIMEOUT}" git clone --depth 1 "${u}" "${CLONE_TMP}" \
               && [[ "${DRY_RUN}" == "1" || -f "${CLONE_TMP}/pyproject.toml" ]]; then
                if [[ "${DRY_RUN}" == "0" ]]; then
                    rm -rf "${OPENVLA_DIR}"
                    mv "${CLONE_TMP}" "${OPENVLA_DIR}"
                fi
                CLONED=1; break
            fi
            rm -rf "${CLONE_TMP}"
        done
        [[ -z "${CLONE_TMP}" ]] || rm -rf "${CLONE_TMP}" 2>/dev/null || true
        (( CLONED == 1 )) || die "OpenVLA 克隆失败（GitHub 及其代理均不可用）；可用 --openvla-src 指定已有源码"
        ok "OpenVLA 克隆完成"
    fi
    # --no-deps 是关键：openvla 的 pyproject 要求 torch==2.2.0，不能让它降级
    exec_cmd "${PY}" -m pip install -e "${OPENVLA_DIR}" --no-deps --no-build-isolation "${PIP_FLAGS[@]}"
    ok "openvla editable 安装完成"
fi

# ---------------------------------------------------------------------------
if [[ "${DOWNLOAD_MODEL}" == "1" && "${CHECK_ONLY}" == "0" ]]; then
    step "6b openvla-7b 权重（约 15 GB，HF 镜像: ${HF_ENDPOINT}）"
    MODEL_DIR="${OPENVLA_DIR}/openvla-7b"
    if [[ -f "${MODEL_DIR}/config.json" ]]; then
        ok "权重已存在: ${MODEL_DIR}"
    else
        exec_cmd "${PY}" - "${MODEL_DIR}" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
snapshot_download(
    repo_id="openvla/openvla-7b",
    local_dir=dest,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.h5", "*.msgpack", "*.ot"],
)
print("saved to", dest)
PY
        ok "权重下载完成"
    fi
fi

# ---------------------------------------------------------------------------
step "7/8 自检"

mkdir -p "${SCRIPT_DIR}/datasets/open-x-embodiment" "${SCRIPT_DIR}/caches" "${SCRIPT_DIR}/runs_btkd"
ok "datasets/ caches/ runs_btkd/ 就绪"

verify() {
    local name="$1" code="$2" out
    if out="$("${PY}" -c "${code}" 2>&1)"; then
        ok "${name}: ${out}"
        printf '%s: %s\n' "${name}" "${out}" >> "${REPORT_FILE}"
        return 0
    fi
    warn "${name}: 失败 → ${out##*$'\n'}"
    printf '%s: FAILED (%s)\n' "${name}" "${out##*$'\n'}" >> "${REPORT_FILE}"
    return 1
}

if [[ "${DRY_RUN}" == "0" ]]; then
    FAILS=0; TORCH_BROKEN=0
    verify "python"        'import sys;print(sys.version.split()[0])' || FAILS=$((FAILS+1))
    # 环境自检不访问 CUDA / 不查询显卡；只报告 PyTorch 编译时携带的 CUDA 运行时。
    verify "torch"         "import torch;assert torch.__version__.split('+')[0] == '${TORCH_VER}', torch.__version__;assert (torch.version.cuda or '') == '${TORCH_CUDA}', torch.version.cuda;print(torch.__version__, 'cuda', torch.version.cuda)" \
        || { FAILS=$((FAILS+1)); TORCH_BROKEN=1; }
    verify "transformers"  'import transformers;print(transformers.__version__)' || FAILS=$((FAILS+1))
    verify "peft"          'import peft;print(peft.__version__)' || FAILS=$((FAILS+1))
    verify "timm"          'import timm;print(timm.__version__)' || FAILS=$((FAILS+1))
    verify "tensorflow"    'import os;os.environ["TF_CPP_MIN_LOG_LEVEL"]="3";import tensorflow as tf;print(tf.__version__)' || FAILS=$((FAILS+1))
    verify "tfds"          'import os;os.environ["TF_CPP_MIN_LOG_LEVEL"]="3";import tensorflow_datasets as tfds;print(tfds.__version__)' || FAILS=$((FAILS+1))
    verify "protobuf"      'import google.protobuf as p;print(p.__version__)' || FAILS=$((FAILS+1))
    verify "btkd_vla"      'import sys;sys.path.insert(0,"'"${SCRIPT_DIR}"'");import btkd_vla;print(btkd_vla.__file__)' || FAILS=$((FAILS+1))
    if [[ "${SKIP_OPENVLA}" != "1" ]]; then
        # 只读 dist-info，避免 import openvla 顺带把 TensorFlow 拉起来（慢且有噪声）
        verify "openvla" 'import importlib.metadata as m;d=m.distribution("openvla");print(m.version("openvla"), d.locate_file("openvla"))' \
            || FAILS=$((FAILS+1))
    fi
    verify "bitsandbytes"  'import bitsandbytes as b;print(b.__version__)' || true

    if (( FAILS > 0 )); then
        warn "自检有 ${FAILS} 项失败；报告: ${REPORT_FILE}"
        if [[ "${TORCH_BROKEN}" == "1" ]]; then
            die "torch 不可用，环境未配置成功；请查看日志: ${LOG_FILE}"
        fi
        if [[ "${CHECK_ONLY}" == "1" ]]; then exit 1; fi
    else
        ok "全部自检通过；报告: ${REPORT_FILE}"
    fi
fi

# ---------------------------------------------------------------------------
step "8/8 冒烟测试（CPU，无需模型/数据/GPU）"

if [[ "${CHECK_ONLY}" == "0" && "${DRY_RUN}" == "0" && "${SKIP_TESTS}" == "0" ]]; then
    ( cd "${SCRIPT_DIR}" && "${PY}" tests/test_core.py ) && ok "单元测试通过" \
        || warn "单元测试未通过（不影响环境可用性）"
    ( cd "${SCRIPT_DIR}" && "${PY}" scripts/smoke_test_a6.py ) && ok "A6 冒烟测试通过" \
        || warn "A6 冒烟测试未通过（多为缓存接口问题）"
    TMP_CACHE="$(mktemp -d)/btkd_dummy_cache"
    ( cd "${SCRIPT_DIR}" \
      && "${PY}" scripts/export_teacher_logits.py --dummy --limit 128 --cache_dir "${TMP_CACHE}" \
      && "${PY}" scripts/train_btkd_student.py --cache_dir "${TMP_CACHE}" --dataset_name dummy --check_cache_only ) \
        && ok "缓存读写链路通过（${TMP_CACHE}）" || warn "缓存链路测试未通过"
else
    ok "跳过（--check / --dry-run / --skip-tests）"
fi

# ---------------------------------------------------------------------------
step "完成"

cat <<TIP

环境名      : ${ENV_NAME}
环境路径    : ${ENV_PREFIX}
PyTorch     : ${TORCH_VER} + cu${TORCH_CUDA/./}
日志        : ${LOG_FILE}
自检报告    : ${REPORT_FILE}

下一步（A6 离线缓存路线）：

  conda activate ${ENV_NAME}
  cd ${SCRIPT_DIR}

  # 1) 用训练好的 OpenVLA 当教师，导出 action-token logits 缓存（不要加 --quantize_4bit）
  python scripts/export_teacher_logits.py \\
      --teacher_path openvla/openvla-7b \\
      --data_root_dir datasets/open-x-embodiment \\
      --dataset_name bridge_orig \\
      --cache_dir caches/bridge_orig_teacher \\
      --batch_size 2 --shuffle_buffer_size 1   # 24 GB 卡；OOM 时降到 1

  # 2) 先校验缓存（逐 token），再训练
  python scripts/train_btkd_student.py \\
      --cache_dir caches/bridge_orig_teacher --dataset_name bridge_orig --check_cache_only
  python scripts/train_btkd_student.py \\
      --cache_dir caches/bridge_orig_teacher \\
      --dataset_name bridge_orig \\
      --data_root_dir datasets/open-x-embodiment \\
      --batch_size 4 --grad_accumulation_steps 4 --quantize_4bit \\
      --shuffle_buffer_size 1

# batch_size / 卡数等运行参数与本安装脚本无关，按自己的机器调整。
细节见 docs/BTKD_OpenVLA_A6_离线缓存.md
TIP
