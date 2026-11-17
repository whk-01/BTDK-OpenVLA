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
re-run, and machine-agnostic. It installs an environment only — it does not probe GPUs, pin
devices or touch training settings, so the number of cards on the box does not matter.

```bash
bash install_btkd_oneclick.sh --env-name whk_BTKD     # full install / repair
bash install_btkd_oneclick.sh --check                 # health check only
bash install_btkd_oneclick.sh --dry-run               # print commands, run nothing
bash install_btkd_oneclick.sh --recreate              # delete and rebuild (dangerous)
bash install_btkd_oneclick.sh --torch-cuda 12.4       # optional: cu124 instead of cu121
```

What it handles:

| Problem on the server | What the script does |
| --- | --- |
| `solver: classic`, 30+ min solves | installs `conda-libmamba-solver`, uses `--solver=libmamba`, and persists `solver: libmamba` in `~/.condarc` (`--no-conda-config` to opt out) |
| `CondaSSLError` on `conda.anaconda.org` | TUNA pytorch + ZJU nvidia + TUNA main, always with `--override-channels`; the nvidia mirror is probed first and can be overridden with `--nvidia-mirror` |
| driver 535 cannot run cu124 | defaults to `--torch-cuda 12.1` (torch 2.5.1 + cu121); the script never probes or pins GPUs |
| empty/corrupt env without `python` | detects it and reinstalls `python=3.10` + `pip` into the env |
| pip torch shadowing conda's cu121 | detects a pip-side `torch`, uninstalls it, reinstalls from conda — also re-checks after the pip phase |
| GitHub unreachable | tries the repo, then `ghfast.top` / `gh-proxy.com` / `gitclone.com` mirrors for both OpenVLA and `dlimp` |
| CUDA 13 packages sneaking in | pins `cuda-version=<12.1|12.4>` and `numpy=1.26.4`, `mkl=2023.1.0`, `intel-openmp=2023.1.0`, `llvm-openmp<16` |

Steps: preflight → solver/mirrors → env → PyTorch (conda) → pip deps → OpenVLA
(editable, `--no-deps`) → import self-check (`logs/env_report_*.txt`) → CPU smoke test
(`tests/test_core.py`, `scripts/smoke_test_a6.py`, dummy cache round-trip). Torchaudio is
installed from the conda pytorch channel, so pip never pulls a second torch build.
A torch failure at the end makes the script exit non-zero instead of silently finishing.

Key pins: Python 3.10, torch 2.5.1 (cu121, or cu124 on newer drivers), torchvision 0.20.1,
transformers 4.40.1, peft 0.11.1, timm 0.9.10, tensorflow 2.15.0, tfds 4.9.3, protobuf 3.20.3.
OpenVLA is installed editable with `--no-deps`, so its `torch==2.2.0` requirement
cannot downgrade the working environment. Logs go to `logs/oneclick_*.log`.
Other useful flags: `--env-name`, `--torch-cuda 12.1`, `--openvla-src DIR`,
`--skip-openvla`, `--skip-tests`, `--no-bitsandbytes`, `--download-model`.

### Troubleshooting: solver hangs / `CondaSSLError` on the server

Two independent problems show up on machines whose conda is old and whose network
to `conda.anaconda.org` is filtered:

1. **`unsuccessful initial attempt using frozen solve. Retrying with flexible solve.`**
   This text only comes from the *classic* solver. pytorch + nvidia channels make
   classic backtracking effectively unbounded (tens of minutes, sometimes OOM).
   Fix by using libmamba, either per command or permanently:

   ```bash
   conda config --set solver libmamba
   conda config --show solver          # -> solver: libmamba
   ```

   (`WARNING conda.models.version:get_matcher ... Your spec was 1.8.0.*` lines are
   harmless deprecation noise from the classic solver parsing repodata.)

2. **`CondaSSLError: ... EOF occurred in violation of protocol`** on the `nvidia`
   channel. Use mirrors and *always* `--override-channels`, otherwise conda still
   fetches repodata from anaconda.org. TUNA mirrors pytorch + main but **not** nvidia;
   ZJU/XJTU mirror nvidia.

   ```bash
   conda install -y -n BTKD --solver=libmamba --override-channels \
     -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch \
     -c https://mirrors.zju.edu.cn/anaconda/cloud/nvidia \
     -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
     python=3.10 pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 \
     cuda-version=12.1 numpy=1.26.4 mkl=2023.1.0 intel-openmp=2023.1.0 "llvm-openmp<16"
   ```

   `install_btkd_oneclick.sh` does exactly this for both conda installs (override
   channels + `cuda-version` pin + libmamba, and it also persists `solver: libmamba`
   into the conda config). Pinning `cuda-version` matters: without it the solver happily
   pulls CUDA 13.x and a pile of cu13 packages.
   On a 535.x driver use `--torch-cuda 12.1` (cu124 wants >= 550).

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

# step 1: export teacher logits (needs a bigger GPU)
python scripts/export_teacher_logits.py --teacher_path openvla/openvla-7b \
    --data_root_dir /data/open-x-embodiment --dataset_name bridge_orig \
    --cache_dir caches/bridge_orig_teacher --batch_size 8

# step 2: sanity-check the cache (CPU is fine)
python scripts/train_btkd_student.py --cache_dir caches/bridge_orig_teacher \
    --dataset_name bridge_orig --check_cache_only

# step 3: train the student
python scripts/train_btkd_student.py --cache_dir caches/bridge_orig_teacher \
    --data_root_dir /data/open-x-embodiment --dataset_name bridge_orig \
    --batch_size 1 --grad_accumulation_steps 16 --quantize_4bit

# CPU-only verification of the whole pipeline
bash run_a6_smoke.sh
```

Full walkthrough (loss derivation, parameter table, failure modes, and the list of deliberate
deviations from the paper): **[docs/BTKD_OpenVLA_A6_离线缓存.md](docs/BTKD_OpenVLA_A6_离线缓存.md)**.
