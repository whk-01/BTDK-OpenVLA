# BTKD

This repository provides the official implementation of **BTKD**. BTKD is a knowledge distillation framework for image classification, object detection, and semantic segmentation.

---

## Environment Setup

Our experiments were conducted with the following software environment:

- Python 3.9
- PyTorch 1.9.0 (CUDA 11.3, cuDNN 8.0)
- NumPy 1.21.0
- OpenCV-Python 4.1.2.30
- Matplotlib 3.4.3

These are common dependencies and can be installed via `pip` or `conda`.

Additional dependencies are required for different tasks:

- For YOLO-based object detection models: `ultralytics`
- For MiT-based semantic segmentation models: `timm`

This codebase supports both **Ubuntu** and **Windows**. The main difference between the two systems is the file path format.

### Pretrained Weights and Checkpoints

- **Image Classification**:  
  Please download the pretrained weights required for classification experiments and place them in the `./pth/` folder under the project root directory.

- **Semantic Segmentation**:  
  Please download the pretrained weights and trained checkpoints required for semantic segmentation experiments and place them in the `./segment/pth/` folder.

All experiments were conducted on **2 × RTX 4090 GPUs**.

---

## Usage

### 1. Image Classification

1. In `dataload_cf100.py`, modify the dataset path to your own data directory.
2. Scripts named `train_XXX_YYY.py` are used for from-scratch training, where:
   - `XXX` denotes the model or backbone network, such as ResNet or ViT;
   - `YYY` denotes the dataset, such as CIFAR-10 or CIFAR-100.
3. Scripts whose filenames start with `dis_` are used for BTKD-based knowledge distillation training.

   The corresponding best checkpoints are listed below:

   - Teacher: [h_resnet50_cf10.pth](https://drive.google.com/file/d/1Z1D65CAkQZSwZv0zrhk5JypnfC84hjm9/view?usp=drive_link)

   - Teacher: [h_resnet50_cf100.pth](https://drive.google.com/file/d/1C5eYZYx_yRb-XWeSeva3vktp_pfFrLFM/view?usp=drive_link)

   - Teacher: [h_resnet50_imagenet.pth](https://drive.google.com/file/d/11MJe7-bfDDt9oGVWxRwDcPwOJJ7NAiAP/view?usp=drive_link)

4. Run the following commands:

   - CIFAR-10: ResNet-50 → ResNet-18

     ```bash
     python dis_rn18_cf10.py
     ```

   - CIFAR-100: ResNet-50 → ResNet-18

     ```bash
     python dis_rn18_cf100.py
     ```

   - CIFAR-100: ResNet-50 → MobileNet-V1

     ```bash
     python dis_mobilenet_cf100.py
     ```

   - ImageNet: ResNet-50 → MobileNet

     ```bash
     python dis_mobilenet_imagenet.py
     ```

---

### 2. Object Detection

For object detection experiments, please enter the `detect` directory and run the scripts whose filenames start with `train_`.

These scripts are used to train object detection models, such as YOLO, under the BTKD framework.

---

### 3. Semantic Segmentation

For semantic segmentation experiments, please enter the `segment` directory and run the scripts whose filenames start with `train_`.

The default teacher–student settings and the corresponding best checkpoints are listed below:

- Teacher: [deeplabv3_resnet101.pth](https://drive.google.com/file/d/1qGLpc-RsLroaeW3hcFaZhmxIvSYdNloL/view?usp=sharing)  
  Student: [deeplabv3_resnet18.pth](https://drive.google.com/file/d/1KTh6FSdJHuq4KCFrPEsHZOHRqo1L0gAj/view?usp=sharing)  
  Best checkpoint: [btkd_deeplabv3_resnet18.pth](https://drive.google.com/file/d/1afcmZ30mT7FboKtxRdHP98U58xnfUbIf/view?usp=sharing)

- Teacher: [mit_B4.pth]()  
  Student: [mit_B0.pth](https://drive.google.com/file/d/1Ave0JlLWIPHZifBCjzr68FyOG9N9BHht/view?usp=sharing)  
  Best checkpoint: [btkd_mit_B0.pth](https://drive.google.com/file/d/1aG02z4vxBE89nxpOJ33VltQMQhrE1bMj/view?usp=sharing)

All the above weights and checkpoint files should be placed in the `./segment/pth/` folder.

---

### 4. Configuration Files and Hyperparameter Settings

This repository currently does not provide additional standalone YAML or JSON configuration files. The main training configurations have been directly written into the corresponding training scripts.

In other words, each released script corresponds to a fixed experimental setting, and the script itself serves as the executable configuration file for that experiment. It includes:

- teacher model architecture;
- student model architecture;
- dataset name;
- batch size;
- optimizer;
- learning rate;
- number of training epochs;
- distillation temperature;
- BTKD / BTKD++ related loss weights;
- checkpoint saving path;
- evaluation settings.

To modify the hyperparameters, please directly edit the default settings in the corresponding script.

### 5. Random Seed Settings

To improve the stability and reproducibility of experimental results, this repository recommends using the following five random seeds:

```text
1, 2, 3, 4, 5
```

## Notes

Due to limited time and computational resources, this repository is still being continuously improved.  
We will gradually release more standardized and complete code, more experimental results, and additional pretrained models.

---

## Citation

If you find this repository helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{zhang2025can,
  title={Can students beyond the teacher? distilling knowledge from teacher’s bias},
  author={Zhang, Jianhua and Gao, Yi and Liu, Ruyu and Cheng, Xu and Zhang, Houxiang and Chen, Shengyong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={22434--22442},
  year={2025}
}
```

---

## Reproducible environment (conda `BTKD`)

`install_btkd_oneclick.sh` is the single, self-contained setup script: idempotent, safe to
re-run, and machine-agnostic. It installs an environment only — it does not probe GPUs,
query VRAM, set `CUDA_VISIBLE_DEVICES`, or touch training settings.

```bash
bash install_btkd_oneclick.sh --env-name whk_BTKD     # full install / repair
bash install_btkd_oneclick.sh --check                 # health check only
bash install_btkd_oneclick.sh --dry-run               # print commands, run nothing
bash install_btkd_oneclick.sh --recreate              # delete and rebuild (dangerous)
bash install_btkd_oneclick.sh --torch-cuda 12.4       # optional: cu124 instead of cu121
bash install_btkd_oneclick.sh --conda-torch           # optional: use conda instead of pip
bash install_btkd_oneclick.sh --torch-index URL       # custom PyTorch wheel index
```

What it handles:

| Problem on the server | What the script does |
| --- | --- |
| CUDA wheel ambiguity / wrong torch build | installs the exact `torch==2.5.1+cu121` (or `+cu124`) wheel from the PyTorch index; never silently falls back to an unspecified-CUDA PyPI wheel |
| PyTorch download index unreachable | tries the official PyTorch index, then SJTU and NJU mirrors; a custom index can be supplied with `--torch-index` |
| `solver: classic`, long solves in conda mode | installs `conda-libmamba-solver`, uses `--solver=libmamba`, and persists `solver: libmamba` in `~/.condarc` (`--no-conda-config` to opt out) |
| `CondaSSLError` on `conda.anaconda.org` | default mode does not use the nvidia conda channel; optional conda mode uses override channels and a tested nvidia mirror |
| driver 535 | defaults to cu121, which is the safe choice for a 24 GB 3090 Ti; cu124 must be requested explicitly |
| empty/corrupt env without `python` | detects it and installs `python=3.10` + `pip` into the environment |
| pip dependency trying to replace torch | re-checks version and CUDA after the pip phase and repairs torch using the same exact wheel index |
| GitHub unreachable | tries the repo, then `ghfast.top` / `gh-proxy.com` / `gitclone.com` mirrors for both OpenVLA and `dlimp` |
| checks accidentally requiring a GPU | `--check` only imports packages and reports `torch.version.cuda`; it never initializes CUDA |

Steps: preflight → solver/mirrors → env → PyTorch (pip by default) → pip deps → OpenVLA
(editable, `--no-deps`) → import self-check (`logs/env_report_*.txt`) → CPU smoke test
(`tests/test_core.py`, `scripts/smoke_test_a6.py`, dummy cache round-trip).

Key pins: Python 3.10, torch 2.5.1 + cu121, torchvision 0.20.1, torchaudio 2.5.1,
transformers 4.40.1, peft 0.11.1, timm 0.9.10, tensorflow 2.15.0, tfds 4.9.3,
protobuf 3.20.3. OpenVLA is installed editable with `--no-deps`, so its `torch==2.2.0`
requirement cannot downgrade the environment. Logs go to `logs/oneclick_*.log`.
Other useful flags: `--env-name`, `--torch-cuda 12.1`, `--torch-index URL`,
`--openvla-src DIR`, `--skip-openvla`, `--skip-tests`, `--no-bitsandbytes`,
`--download-model`.

### Troubleshooting: solver hangs / `CondaSSLError` on the server

These failures only apply when PyTorch is installed through conda (`--conda-torch`) or when
the environment itself is created. The default PyTorch path is pip and does not touch the
nvidia conda channel.

1. **`unsuccessful initial attempt using frozen solve. Retrying with flexible solve.`**
   This text comes from the *classic* solver. Pytorch + nvidia channels make classic
   backtracking effectively unbounded. Use libmamba:

   ```bash
   conda config --set solver libmamba
   conda config --show solver          # -> solver: libmamba
   ```

2. **`CondaSSLError: ... EOF occurred in violation of protocol`** on the `nvidia`
   channel. Use mirrors and always `--override-channels`, otherwise conda still
   fetches repodata from anaconda.org. TUNA mirrors pytorch + main but not nvidia;
   ZJU mirrors nvidia.

   ```bash
   bash install_btkd_oneclick.sh --env-name BTKD --conda-torch \
     --nvidia-mirror https://mirrors.zju.edu.cn/anaconda/cloud/nvidia
   ```

   If the nvidia channel remains unreachable, use the default pip mode. It downloads a
   wheel that bundles the CUDA runtime and therefore does not need that channel.

### Troubleshooting: `ModuleNotFoundError: No module named 'dlimp'`

`dlimp` is a hard dependency of OpenVLA (`prismatic.vla`), but OpenVLA itself is installed
with `--no-deps`, so `dlimp` must be installed separately. The one-click installer now does
this and verifies both `dlimp` and `prismatic` during self-check. To repair an existing env:

```bash
conda activate whk_BTKD
cd /path/to/BTDK-OpenVLA

python -m pip install --no-deps \
  "dlimp @ git+https://ghfast.top/https://github.com/moojink/dlimp_openvla"

python -c "import dlimp, prismatic; print(dlimp.__file__); print(prismatic.__file__)"
```

Use the `moojink/dlimp_openvla` fork, not the PyPI `dlimp` package. If `git+https` is
blocked, download the source archive through the same mirror and install the directory:

```bash
wget -O /tmp/dlimp_openvla.tar.gz \
  https://ghfast.top/https://github.com/moojink/dlimp_openvla/archive/refs/heads/main.tar.gz
tar -xzf /tmp/dlimp_openvla.tar.gz -C /tmp
python -m pip install --no-deps /tmp/dlimp_openvla-main
```

`dlimp` also needs `tensorflow==2.15.0` and `tensorflow_datasets>=4.9.2`; the one-click
installer installs both. If they are missing:

```bash
python -m pip install "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" \
  "tensorflow-metadata==1.15.0" "protobuf==3.20.3"
```

### Troubleshooting: `ModuleNotFoundError: No module named 'tensorflow_graphics'`

OpenVLA imports `tensorflow_graphics.geometry.transformation` while loading the RLDS/OXE
dataset registry, even for LIBERO. Install it **without dependencies**: its package metadata
pulls `tensorflow-addons`, `OpenEXR`, `trimesh`, etc., which are not needed for this import
and often fail to build.

```bash
python -m pip install --no-deps "tensorflow_graphics==2021.12.3"

python - <<'INNER'
import tensorflow_graphics
import tensorflow_graphics.geometry.transformation as tfg
print("tensorflow_graphics:", tensorflow_graphics.__version__)
print("transformation   :", tfg.__file__)
INNER
```

The one-click installer now installs `tensorflow_graphics` with `--no-deps` and checks it in
the final self-test.

## BTKD for OpenVLA (offline teacher-logit cache, "A6")

`openvla/` is cloned into this repo and installed into the same `BTKD` conda environment,
but it is **not** wired into BTKD's training scripts. The adaptation lives in new code only
(`btkd_vla/`, `scripts/`, `tests/`); nothing under `openvla/` or the original `dis_*`/`detect/`/
`segment/` scripts was modified.

Why offline: BTKD needs `teacher + student` forward passes on the same batch. Two OpenVLA-7B
models need ~38-45 GB, so instead the teacher runs once, its 7x256 action-token logits are cached
(~3.5 KB/sample), and training only keeps the student resident (~10 GB with QLoRA).

```bash
conda activate BTKD && cd /home/whk/BTKD

# optional run-only helper: pins one visible GPU and picks batch sizes (not part of env install)
bash run_a6_single_gpu.sh export --dataset bridge_orig   # bs=2, auto-falls back to 1 on OOM
bash run_a6_single_gpu.sh check  --dataset bridge_orig
bash run_a6_single_gpu.sh train  --dataset bridge_orig   # QLoRA bs=4 x accum 4

# step 1: export teacher logits once, in deterministic dataset order
# (uses btkd_vla/rlds_dataset.py: TFDS shard shuffle off, single read/map worker)
# 24 GB card: start at batch_size 2 (drop to 1 on OOM); 48 GB cards can use 8
python scripts/export_teacher_logits.py --teacher_path openvla/openvla-7b \
    --data_root_dir /data/open-x-embodiment --dataset_name bridge_orig \
    --cache_dir caches/bridge_orig_teacher --batch_size 2 --shuffle_buffer_size 1

# step 2: sanity-check the cache (CPU only; original data path is not required)
python scripts/train_btkd_student.py --cache_dir caches/bridge_orig_teacher \
    --dataset_name bridge_orig --check_cache_only

# step 3: train the student (image augmentation and shuffling stay off)
python scripts/train_btkd_student.py --cache_dir caches/bridge_orig_teacher \
    --data_root_dir /data/open-x-embodiment --dataset_name bridge_orig \
    --batch_size 1 --grad_accumulation_steps 16 --quantize_4bit \
    --shuffle_buffer_size 1

# CPU-only verification of the whole pipeline
bash run_a6_smoke.sh
```

Export semantics: without `--limit`, exactly one dataset pass is cached. If an export is
interrupted, `--resume` continues from the rows recorded in `manifest.json` (not from the
approximate `progress.json`), and the export batch size may be changed. A complete cache may
wrap to row 0 for additional epochs; every wrap re-checks the cached GT action tokens. A cache
exported or trained with `--limit` is a prefix only and never wraps silently.

Cache order contract: export and training both use `DeterministicSingleRLDSDataset`, which
disables TFDS shard-file shuffling and pins all RLDS read/map/flatten parallelism to 1.
`--shuffle_buffer_size 1` and `--image_aug` off are still required but are **not** sufficient
on their own (upstream `RLDSDataset` still shuffles shards across processes). Caches whose
manifest lacks `data_order=deterministic_single_rlds_v1` must be re-exported;
`run_a6_single_gpu.sh export` backs them up to `*.legacy_<timestamp>` and re-exports automatically.

Full walkthrough (loss derivation, parameter table, failure modes, and the list of deliberate
deviations from the paper): **[docs/BTKD_OpenVLA_A6_离线缓存.md](docs/BTKD_OpenVLA_A6_离线缓存.md)**.
