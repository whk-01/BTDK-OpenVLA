# BTKD × OpenVLA：A6 离线缓存蒸馏路线（实现说明）

> 适用仓库：`/home/whk/BTKD`（BTKD 官方实现 + `openvla/`）
> conda 环境：`BTKD`（Python 3.10 / PyTorch 2.5.1+cu121；cu124 需显式 `--torch-cuda 12.4`）
> 代码位置：`btkd_vla/`、`scripts/`、`tests/`
> 论文：Zhang et al., *Can students beyond the teacher? Distilling knowledge from teacher's bias*, AAAI 2025

---

## 1. 这个方案解决什么问题

BTKD 的标准形态是 **teacher + student 同时前向**：

```
同一 batch 图像 ──┬──> Teacher(7B, 冻结) ──> y_tea
                  └──> Student(7B+LoRA)  ──> y_stu
```

两份 7B 在 bf16 下光权重就是 30 GB，加上 KV cache、激活、优化器状态，实测需要 **38–45 GB**，
24 GB 卡放不下，8 GB 笔记本更不可能。

**A6 的做法：把 teacher 从训练时的显存里彻底移除。**

```
离线阶段（一次性，可在任意大显存机器/云端）
    Teacher(7B) ──前向一次──> 只存 7×256 个 action logits ──> cache/*.pt

训练阶段（你的机器，只需要 student）
    dataset ──> Student(7B+LoRA) ──> 7×256 logits
                                        └── 从 cache 读出 teacher logits
                                            └── 算 BTKD loss
```

关键点：**蒸馏 loss 只依赖 teacher 的输出，不依赖 teacher 的参数**。
所以只要把 teacher 的输出存下来，数学上与在线蒸馏**完全等价**（teacher 冻结、eval、无随机性）。

一个样本的缓存只有 `7 × 256 × 2 byte ≈ 3.5 KB`：

| 数据量 | 缓存体积 |
|---|---|
| 1 万条 | ~35 MB |
| 10 万条 | ~350 MB |
| 100 万条 | ~3.5 GB |

---

## 2. BTKD 路线到 OpenVLA 的映射

BTKD 原始实现（`tool.py` / `dis_resnet18_cf100.py`）是分类任务，OpenVLA 是自回归动作 token，
逐项对应如下。

| BTKD（分类） | 本实现（OpenVLA） | 代码位置 |
|---|---|---|
| 一条图像样本 | 一个 (图像, 指令, 状态) 样本 | `RLDSDataset` |
| 1 个 100 维 logits | `D` 个位置 × 256 维 action logits | `vla_glue.extract_action_logits` |
| `argmax(y_tea) == label`（样本级） | 逐 token：teacher argmax token == GT action token | `core.split_easy_hard` |
| `calibrate3`（100 类中改 2 个） | 在 256 路 action 子空间里改 2 个 | `core.calibrate3_logits` |
| `loss_easy` / `loss_hard` | 同左，mask 换成 token 级 | `core.btkd_term` |
| `b = cos²(π/2 · epoch/epochs)` | `b = cos²(π/2 · step/total_steps)` | `core.cosine_square_schedule` |
| `loss = λ·CE + (1-λ)·T²·(b·easy + (1-b)·hard)` | 同左，CE 来自 OpenVLA 原生 loss | `core.total_loss` |

### 2.1 为什么要取「最后 256 个 token 的 logits」

OpenVLA 的 `ActionTokenizer` 把每维动作离散到 256 个 bin，并把 bin 映射到**词表末尾的 256 个 token**：

```
token_id = vocab_size - digitized          # digitized ∈ [1, 256]
=> token_id ∈ [vocab_size - 256, vocab_size - 1]   # Llama-2: [31744, 31999]
=> action_token_begin_idx = vocab_size - 257 = 31743
```

推理时 `predict_action` 也只在这 256 个 token 上解码，因此 **KD 只需要在这 256 维上做**，
不需要缓存 32000 维的完整 logits（那会是 448 KB/样本，大 128 倍）。

本实现统一采用「**升序 token id 排列**」的 256 维切片 `logits[..., -256:]`：

```
col c  ↔  token_id = vocab_size - 256 + c
```

顺序本身不影响 BTKD（校准只动两列），统一约定是为了避免 off-by-one。

### 2.2 easy / hard 的判定（含越界处理）

```python
in_space   = (31744 <= teacher_argmax_token < 32000)
easy       = in_space and (teacher_argmax_token == gt_token)
hard       = not easy
```

比论文多了一个 **越界分支**：teacher 的 full-vocab argmax 有可能落在动作 token 之外
（它毕竟是个语言模型）。这种情况一律判为 hard，并在做校准时退回到「256 维切片内的 argmax」。
缓存里同时记录了 `out_of_space_frac`，便于诊断（正常情况下应该接近 0）。

### 2.3 错误知识校准（等价改写 `calibrate3`）

论文代码是一个 Python 双层循环：

```python
a, c = y[l_idx], y[y_idx]          # c == teacher argmax, 必然 c >= a
sum1 = a + c
a1, c1 = (a + 1) / 2, c / 2
scale = sum1 / (a1 + c1)
y[l_idx], y[y_idx] = a1 * scale, c1 * scale
```

代数化简后：

```
a2 = a + c / (a + c + 1)
c2 = c - c / (a + c + 1)        # 且 a2 + c2 == a + c
```

本实现用这个闭式做**向量化**改写，并加了 `strength` 系数：

```
a2 = a + α · c/(a+c+1)
c2 = c - α · c/(a+c+1)          # α = 1.0 与论文数值等价
```

测试 `tests/test_core.py::test_calibrate3_matches_reference` 直接拿 `tool.calibrate3` 的
循环实现做对照，最大绝对误差 `5e-7`（float32 精度级别）。

### 2.4 训练进度权重 b

```python
b = cos²(π/2 · step / total_steps)
```

- `step = 0` → `b = 1.0`：全部走 easy
- 中点 → `b = 0.5`
- 结束 → `b = 0.0`：全部走 hard

注意 BTKD 源码里的注释 `lr_values 0-------->1` 是**写反的**，实际公式是 1→0。

---

## 3. 显存账单

| 阶段 | 内容 | 显存 |
|---|---|---|
| **离线导出**（一次性） | 7B teacher bf16 前向，batch 1–2 | **16–20 GB** |
| **训练** | 7B + LoRA(bf16)，batch 1 | **20–27 GB** |
| **训练** | 7B + QLoRA(4-bit base)，batch 1 | **9–12 GB** |
| 训练（小 student） | 1B 级 VLA + LoRA | 5–7 GB |

对你这台 **8 GB 笔记本 4060**：

- 导出阶段跑不了 -> 需要换机器或租卡（导出只需一次，A100/4090 上几个小时）
- 训练阶段用 `--quantize_4bit`，**约 10 GB**；如果显存还是不够，再把 `--batch_size` 降到 1、
  `--grad_accumulation_steps` 提到 8–16（不改变有效 batch 的梯度尺度）

> 8 GB 下 `batch_size=1` + QLoRA 可能仍然卡在 10–11 GB 峰值，取决于 `model_max_length`
> 与图像分辨率。真跑不动就用 12–16 GB 的卡做训练，导出则一定要更大的卡。

### 单张 24 GB（3090 Ti / 4090）怎么配

服务器上只分到**一张** 3090 Ti 时，两件事必须做对：**只用这张卡**、**导出 batch 不要贪**。

| 阶段 | 推荐参数 | 峰值显存 | 说明 |
|---|---|---|---|
| 导出 teacher | `--batch_size 2` | 约 18–20 GB | 最稳的性价比；OOM 就降到 1 |
| 导出 teacher | `--batch_size 1` | 约 16–17 GB | 卡上还跑着别人的任务时用这个 |
| 导出 teacher | `--batch_size 4` | 约 21–23 GB | 能跑但没余量，容易在长序列样本上 OOM |
| 训练 QLoRA | `--batch_size 4 --grad_accumulation_steps 4 --quantize_4bit` | 约 10–12 GB | 24 GB 很宽裕，可以再试 `batch_size 6~8` |
| 训练 LoRA(bf16) | `--batch_size 1` | 20–27 GB | 24 GB 上很紧，不建议；除非把 `max_length` 调小 |

只暴露一张卡（`tf.data` 默认会把所有可见卡都占上，共享服务器上尤其要注意）：

```bash
export CUDA_VISIBLE_DEVICES=0        # 进程内只剩 cuda:0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

仓库里的 `run_a6_single_gpu.sh` 已经把上面这些、批量大小、OOM 自动降档都封好了：

```bash
bash run_a6_single_gpu.sh export --dataset bridge_orig   # 导出（内部 --device cuda:0）
bash run_a6_single_gpu.sh check  --dataset bridge_orig   # 校验缓存
bash run_a6_single_gpu.sh train  --dataset bridge_orig   # QLoRA 蒸馏
```

`--gpu 3` 指定别的卡，`--gpu auto` 挑当前最空闲的一张。

导出是**一次性**开销：bridge_orig 这类 ~6 万条的集子，单卡 3090 Ti、bs=2、7B bf16 前向，
大约 1–3 小时（tqdm 会给出 ETA）。**默认只遍历数据集一遍**；RLDS 本身是无限重复流，
必须由 `dataset_length` 或 `--limit` 指定停止点。

中断后加 `--resume`：续写点取自 `manifest.json` 中**实际落盘的行数**，不是 `progress.json` 的
近似 batch 计数。因此可以更换导出 `--batch_size`（例如 OOM 后从 2 降到 1），不会错位。
数据流仍需从第 0 条重新读到续写点，所以恢复启动会慢一些，但不会重复写行。

完整缓存训练超过一个 epoch 时可以回到 row 0 继续；batch 跨过数据集末尾时按
`[tail..., head...]` 拼接读取，并在每次回绕时重新做 GT token 对齐校验。用 `--limit`
导出的部分缓存、或训练时带 `--limit`，都不会自动回绕。

---

## 4. 目录结构

```
/home/whk/BTKD/
├── btkd_vla/
│   ├── spec.py        # ActionSpec（词表/分箱/维度）、BTKDConfig（T/λ/b/α）
│   ├── core.py        # 纯张量实现：easy-hard 划分、calibrate3、KL、权重、总损失
│   ├── cache.py       # 分片 fp16 缓存：写入 / mmap 读取 / 统计 / 断点续写
│   ├── rlds_dataset.py# 确定性单数据集 RLDS 管线（导出/训练共享的逐行顺序契约）
│   └── vla_glue.py    # OpenVLA 模型加载、action logits 抽取（懒加载 prismatic）
├── scripts/
│   ├── export_teacher_logits.py   # 第 1 步：导出 teacher 缓存（支持 --dummy 与 --resume）
│   ├── train_btkd_student.py      # 第 3 步：LoRA/QLoRA student + BTKD loss
│   ├── eval_btkd_student.py       # 第 4 步：离线验证 adapter（对比缓存 teacher）
│   └── smoke_test_a6.py           # CPU 全链路冒烟测试（不需要模型和数据）
├── tests/
│   └── test_core.py               # 19 个单元测试（核心数学 + 缓存续写/回绕）
└── docs/
    └── BTKD_OpenVLA_A6_离线缓存.md  # 本文档
```

本方案**没有修改** `openvla/` 以及 BTKD 原有的 `dis_*.py` / `detect/` / `segment/`，
只新增了上面这些文件，因此原仓库的实验脚本仍然可以照常运行。

---

## 5. 三步操作流程

### Step 0：环境

```bash
source /home/whk/miniconda3/etc/profile.d/conda.sh
conda activate BTKD
cd /home/whk/BTKD
```

`btkd_vla` 不需要 `pip install`：`scripts/` 下的脚本都会把仓库根目录插入 `sys.path`。

### Step 1：导出 teacher logits（大显存机器）

```bash
python scripts/export_teacher_logits.py \
    --teacher_path  openvla/openvla-7b \
    --data_root_dir /data/open-x-embodiment \
    --dataset_name  bridge_orig \
    --cache_dir     caches/bridge_orig_teacher \
    --batch_size    2 \
    --shuffle_buffer_size 1 \
    --n_dims        7 \
    --vocab_size    32000
```

要点：

- **只做一次前向**，没有 `generate()`、没有 KV cache 增长：teacher 在 GT 动作 token 上做
  teacher-forcing，取 `logits[t-1]` 预测 `label[t]`，与 `vla-scripts/finetune.py` 的对齐方式一致。
- `--shard_size 5000`：每 5000 条落一个 `shard_XXXXX.pt`。
- 上面的 `--batch_size 2` 是单张 24GB 卡的稳妥值（OOM 降到 1）；48GB 级卡可尝试 8。
- 不加 `--limit` 时只导出**一遍完整数据集**；`--limit` 大于数据集长度会自动截断。
- 中断后加 `--resume` 从 manifest 实际落盘行继续，可以更换 `--batch_size`。
- `progress.json` 只用于观察进度，**不是**续写依据。
- `--shuffle_buffer_size` 必须为 1，且 **`--image_aug` 必须关闭**（默认关闭），原因见 §7。
- 导出不再使用 OpenVLA 自带的 `RLDSDataset`，而是 `btkd_vla/rlds_dataset.py` 中的
  `DeterministicSingleRLDSDataset`：TFDS 分片文件**不 shuffle**，读/映射/展平并行度全部固定为 1。

导出结果：

```
caches/bridge_orig_teacher/
├── manifest.json      # spec + teacher 信息 + 分片表 + complete 标记
├── progress.json      # 仅用于观察；续写点来自 manifest.json
├── shard_00000.pt     # {gt_token, teacher_logits(fp16), teacher_argmax_token, start}
└── shard_00001.pt
```

### Step 2：检查缓存（任何机器，CPU 即可）

```bash
python scripts/train_btkd_student.py \
    --cache_dir    caches/bridge_orig_teacher \
    --dataset_name bridge_orig \
    --check_cache_only
```

输出示例：

```json
{
  "n_samples": 512,
  "n_tokens": 3584,
  "teacher_token_acc": 0.8276,
  "easy_frac": 0.8276,
  "out_of_space_frac": 0.0,
  "nonfinite_logits": 0,
  "logit_min": -9.22,
  "logit_max": 20.0
}
```

判读标准：

| 指标 | 期望 | 异常含义 |
|---|---|---|
| `teacher_token_acc` | 微调过的 teacher 在自身训练分布上通常 **0.85–0.99** | 明显偏低 → 动作归一化统计不一致，或数据/顺序错位 |
| `easy_frac` | 与 `teacher_token_acc` 基本相同 | 差很多 → `out_of_space_frac` 高，先看那一项 |
| `out_of_space_frac` | ≈ 0 | 偏高 → teacher 未在该数据集上微调，argmax 溢出到语言 token |
| `nonfinite_logits` | 0 | > 0 → 导出时有 NaN（bf16 溢出或 4-bit 量化异常） |

### Step 3：训练 student

24 GB（bf16 LoRA，batch 1）：

```bash
python scripts/train_btkd_student.py \
    --cache_dir    caches/bridge_orig_teacher \
    --vla_path     openvla/openvla-7b \
    --data_root_dir /data/open-x-embodiment \
    --dataset_name  bridge_orig \
    --batch_size 1 --grad_accumulation_steps 16 \
    --max_steps 20000 --learning_rate 5e-4 \
    --lora_rank 32 \
    --temperature 4.0 --lambda_stu 0.5 \
    --shuffle_buffer_size 1 \
    --run_root_dir runs_btkd
```

8–12 GB（QLoRA）：

```bash
python scripts/train_btkd_student.py \
    --cache_dir    caches/bridge_orig_teacher \
    --vla_path     openvla/openvla-7b \
    --data_root_dir /data/open-x-embodiment \
    --dataset_name  bridge_orig \
    --batch_size 1 --grad_accumulation_steps 16 \
    --max_steps 20000 --learning_rate 5e-4 \
    --lora_rank 32 --quantize_4bit \
    --shuffle_buffer_size 1 \
    --run_root_dir runs_btkd
```

日志每 `--log_steps` 打印：

```
[step   100] loss=0.4123 ce=0.3810 kd=0.0626 easy=0.0310 hard=0.0316 b=0.994 teacher_tok_acc=0.912 easy_frac=0.905
```

需要重点盯的三件事：

1. `ce` 要持续下降（这是学生真正的任务项）
2. `b` 按 `cos²` 从 1 掉到 0，`easy`/`hard` 两列在 `b≈0.5` 之后应该由 `hard` 主导
3. `teacher_tok_acc` 应稳定在导出时的水平；如果逐 step 抖动极大，说明缓存与数据错位（见 §7.1）

产物：`runs_btkd/<exp>/final/`（PEFT adapter），可参考 `openvla/README.md` 的
"merge LoRA" 流程合并回完整权重再部署。

### Step 4：离线验证 adapter（推荐）

只加载学生一个模型，用同一份缓存对比 student / teacher：

```bash
bash run_a6_single_gpu.sh eval \
  --dataset libero_spatial_no_noops \
  --cache caches/libero_spatial_teacher \
  --adapter runs_btkd/btkd+openvla-7b+libero_spatial_no_noops+b16+lr0.0005+T4.0+lam0.5/final \
  --gpu 3 --env-name whk_BTKD \
  --limit 512
```

不传 `--adapter` 时 runner 会自动取最新的 `runs_btkd/*/final`；不传 `--limit` 时默认只
验证前 512 条（全量很慢）。想看 base 基线就直接调用脚本且不传 `--adapter_path`：

```bash
python scripts/eval_btkd_student.py \
  --cache_dir caches/libero_spatial_teacher \
  --dataset_name libero_spatial_no_noops --limit 512
```

先测 base、再测 adapter，对比提升。

核心指标：

| 指标 | 含义 |
|---|---|
| `student_token_acc` | 学生动作 token 命中 GT 的比例 |
| `teacher_token_acc` | 缓存 teacher 命中 GT 的比例（离线上限） |
| `student_teacher_agreement` | 学生 argmax 与 teacher argmax 的一致率 |
| `student_token_acc_easy/hard` | 按 teacher 是否正确划分的两个分支各自的命中率 |

注意：验证样本就是训练缓存本身，所以这是"是否学会"的拟合检查，不是泛化指标。
真正的成功率要用 LIBERO 仿真 rollout 评估。

---

## 6. 参数速查

| 参数 | 默认 | 说明 / 建议 |
|---|---|---|
| `--temperature` | 4.0 | KD 温度。**OpenVLA 的 logits 尺度与 ResNet 不同**，如果发现 `kd` 比 `ce` 小两个数量级，先试 2.0；反之试 6.0 |
| `--lambda_stu` | 0.5 | CE 权重，BTKD 论文默认 0.5。学生基础能力弱时调高（0.7） |
| `--calibrate_strength` | 1.0 | 校准强度 α，1.0 = 论文原版，0.0 = 关闭校准（退化为普通 KD） |
| `--schedule` | cos2 | 1→0 的余弦平方调度 |
| `--logit_norm` | none | `none`/`center`/`std`。**teacher 和 student 同架构（LoRA）时保持 none**；若换成异构小模型，用 `std` |
| `--renormalize_empty_branch` | 关 | 开：某分支为空时把权重让给另一个分支（偏离论文但更稳） |
| `--verify_batches` | 5 | 前 N 个 batch 校验 GT token；每次 epoch 回绕也会强制校验 |
| `--shuffle_buffer_size` | 1 | 必须为 1；确定性顺序由 `DeterministicSingleRLDSDataset` 保证，导出与训练必须使用同一数据集 |
| `--limit` | -1 | 只用前 N 条（调试用）；会禁用缓存回绕 |
| `--quantize_4bit` | 关 | QLoRA，省显存；训练侧 4-bit 是可接受的（只有 teacher 侧不能 4-bit） |
| `--attn_implementation` | sdpa | 该环境未装 flash-attn，`sdpa` 是正确选择 |

---

## 7. 三条硬约束（很重要）

### 7.1 数据顺序必须一致

缓存是**按导出顺序**逐行追加的，训练时按同样顺序逐行读取，**不存 key**（存 key 需要每帧
稳定的 traj/frame id，RLDS 重构后的 batch 里拿不到）。跨进程复现顺序需要比
"`shuffle_buffer_size=1`"更强的约束，因为 OpenVLA 自带的 `RLDSDataset` 还会：

- 随机化 TFDS 分片文件（shard）的读取顺序；
- 让 TensorFlow AUTOTUNE 决定 `num_parallel_reads` / `num_parallel_calls`。

这两点都会让导出进程和训练进程从不同样本开始，即使帧级 shuffle buffer 为 1。因此两个脚本
统一改用 `btkd_vla.rlds_dataset.DeterministicSingleRLDSDataset`：

- TFDS `shuffle=False`，分片文件按固定顺序读取；
- `num_parallel_reads=1`、trajectory/frame transform 单线程、`flatten(num_parallel_calls=1)`；
- 只支持单数据集（多数据集混合的随机采样流无法提供稳定的逐行契约）；
- 仍然要求 `--shuffle_buffer_size 1` 且关闭 `--image_aug`。

训练脚本默认用前 `--verify_batches` 个 batch 做**逐 token 比对**：

```
cache/dataset misalignment at batch 3 (cache rows starting at 12).
  cached : [[31790, 31611, ...]]
  dataset: [[31876, 31999, ...]]
```

一旦出现，说明缓存来自旧版顺序或数据集内容已变化 —— 用当前代码重新导出即可。运行器
`run_a6_single_gpu.sh export` 会自动把旧版本/旧顺序缓存备份成
`caches/<name>.legacy_YYYYMMDD_HHMMSS` 后重导。

> 副作用：训练是顺序遍历，没有 shuffle。对 KD 训练影响有限（teacher 信号本身就有正则性），
> 但如果你要做严格的 shuffle，需要额外实现「按 key 索引」的缓存（用 traj id + frame idx 作 key），
> 这需要改 RLDS 数据管线，不在本次范围内。

### 7.2 图像增强必须关闭

teacher 的图像是**离线、干净**的；student 如果开 `--image_aug`，同一帧看到的图与 teacher 不同。
GT token 仍然一致（动作不受增强影响），所以顺序校验照样通过，但 teacher 的分布对应的是另一张图，
easy/hard 判定会失真。默认关闭；如果你确实想用增强，请把它理解为「带噪 KD」，并降低 λ_kd。

### 7.3 回绕规则

- 只有**完整导出的缓存**可以在训练超过一个 epoch 后回到 row 0。
- 若 `dataset_length % batch_size != 0`，最后一个 batch 会跨 epoch，缓存按
  `[末尾若干行, 开头若干行]` 拼接，顺序与 RLDS 无限数据流一致。
- 每次回绕训练脚本都会重新比对 GT token；错位会立即报错。
- 用 `--limit` 导出的部分缓存，或训练命令带 `--limit`，都禁止回绕，避免悄悄训练到
  用户没有请求的样本。

---

## 8. 与 BTKD 原始实现的有意差异

| # | 差异 | 原因 |
|---|---|---|
| 1 | easy/hard 是 **token 级**而非样本级 | 7 个动作位置里通常只错 1–2 个；按整条轨迹划分会让 hard 占比爆炸，easy 分支失效 |
| 2 | KD 只在 256 维 action 子空间 | 与推理解码路径一致，缓存体积从 448 KB/样本降到 3.5 KB |
| 3 | `masked_mean` 对空集合返回 0（原代码会 NaN） | 原实现 `KLDivLoss(reduction='batchmean')` 在子集为空时除零；本项目里「整批全对」并不罕见 |
| 4 | `calibrate3` 闭式向量化，并加 `strength` 系数 | 逐样本 Python 循环在 7B 训练循环里太慢；`strength=1.0` 与论文数值等价（有对照测试） |
| 5 | `b` 按 **step** 而非 epoch | VLA 一个 epoch 可能是几十万 step |
| 6 | teacher 前向显式 `inference_mode()` + 参数 `requires_grad_(False)` | 原 BTKD 脚本没用 `no_grad`，直接搬到 7B 会凭空多出一份梯度和激活 |
| 7 | 新增越界 argmax 分支 | teacher 是语言模型，full-vocab argmax 可能不在动作 token 区间 |

另外：`renormalize_empty_branch` 默认**关闭**，即严格遵循论文的 `b·easy + (1-b)·hard`；
开启后在某些 batch 上会与论文略有偏离（文档里已标注）。

---

## 9. 故障排查

### 9.1 `cache/dataset misalignment`

缓存与数据集顺序不一致。可能原因：

1. 缓存是**旧版代码**导出的（manifest 中没有 `data_order=deterministic_single_rlds_v1`）；
2. 导出后 `--dataset_name` / 数据目录内容发生了变化；
3. 导出或训练时改了 `--shuffle_buffer_size`，或打开了 `--image_aug`。

处理：用当前代码重新导出。`run_a6_single_gpu.sh export` 会检测旧格式并自动备份重导；
手动执行时加 `--overwrite`。

### 9.2 训练 loss 出现 NaN

本实现已经消除了空 mask 的除零。若仍出现：

1. `--check_cache_only` 看 `nonfinite_logits` 是否为 0
2. 降低 `--learning_rate`（QLoRA 常常需要 1e-4 ~ 2e-4）
3. 检查 `--temperature` 是否过小（T<0.5 时 softmax 会数值爆炸）
4. `teacher_tok_acc` 极低（<0.3）时 hard 分支占比接近 100%，校准量变大，先把 `--calibrate_strength 0.5` 试一下

### 9.3 `expected exactly 7 supervised action tokens per sample`

`--n_dims` 与该数据集的真实动作维度不一致，或 collator 的 label 约定不同。
先 `--check_cache_only` 确认缓存里的 `n_dims`，训练时用同一个值。

### 9.4 teacher 能不能 4-bit 导出？

**不能**。导出脚本会直接拒绝 `--quantize_4bit`。`calibrate3` 的转移量是 `c/(a+c+1)`，量级 O(1)，而 4-bit 量化给 logits 引入的
噪声与之同量级，校准在数值上失去意义。导出阶段请用 bf16；只有**训练侧**的 student 适合 4-bit。

### 9.5 OOM

按顺序尝试：`--batch_size 1` → `--grad_accumulation_steps` 翻倍 → `--quantize_4bit` →
换 16 GB 以上的卡。注意 7B LoRA 本身的固定开销就有 ~16 GB（权重 15 GB），降到 8 GB 只能靠 QLoRA。

---

## 10. 验证清单（已在 `BTKD` 环境实际执行）

```bash
conda activate BTKD && cd /home/whk/BTKD

python tests/test_core.py          # 18/18 passed
python scripts/smoke_test_a6.py    # SMOKE TEST PASSED
```

`tests/test_core.py` 覆盖：

| 测试 | 检查内容 |
|---|---|
| `calibrate3_matches_reference` | 与 `tool.calibrate3` 的循环实现逐值一致（最大误差 5e-7） |
| `calibrate3_preserves_sum_...` | `a2+c2 == a+c`，真实类 logit 上升、错误类下降、差距收窄 |
| `calibrate_strength_zero` | α=0 时输出与输入完全一致 |
| `split_masks` | easy/hard 划分 + 越界 argmax 强制为 hard |
| `masked_mean_empty` | 空 mask 返回 0 且**梯度为 0**（不产生 NaN） |
| `kl_identical_is_zero` | `KL(x‖x)=0` |
| `schedule` | `b(0)=1, b(0.5T)=0.5, b(T)=0` |
| `btkd_term_...` | 全 easy / 混合 / 全 hard × b∈{0,0.5,1} 全部有限 |
| `total_loss_weighting` | `λ·CE + (1-λ)·KD` 权重正确 |
| `extract_action_logits` | `logits[t-1]` 与 `label[t]` 对齐；label 布局错误会报错 |
| `cache_roundtrip` | 分片写入 / 跨分片读取 / 断点续写 / fp16 精度 |
| `cache_resume_after_abort` | 中断后按实际落盘行续写，无重复行 |
| `cache_overwrite_removes_old_shards` | `--overwrite` 清理旧 shard/manifest，避免混入旧数据 |
| `cache_resume_rejects_missing_shard` | 缺 shard 时拒绝续写 |
| `cache_resume_from_empty_manifest` | 0 行 manifest 可以从 row 0 续写 |
| `cache_window_crosses_epoch_boundary` | batch 跨 epoch 时按 `[tail, head]` 读取且保持对齐 |
| `cache_wrap_safety` | 只有完整缓存且训练未设 `--limit` 时才允许回绕 |

`scripts/smoke_test_a6.py` 跑的是完整链路：**写缓存 → 读缓存 → 300 步优化 toy student**，
KD loss 从 `0.1231` 降到 `0.0008`，并验证了空分支行为：

```
[smoke] all-easy, b=1: kd=1.0504 (easy_raw=1.0504)
[smoke] all-hard, b=1: faithful kd=0.0000, renormalised kd=1.0551
```

另外用 `--dummy` 端到端跑过 CLI：

```bash
python scripts/export_teacher_logits.py --dummy --limit 512 --cache_dir /tmp/btkd_dummy_a6
python scripts/train_btkd_student.py --cache_dir /tmp/btkd_dummy_a6 --dataset_name dummy --check_cache_only
# -> teacher_token_acc = 0.8276（构造值 0.82）
```

**未验证的部分**（当前机器无法验证）：真实 7B teacher 的前向、RLDS 数据管线、LoRA 训练本身。
这三块依赖模型权重与数据集，需要在有卡的机器上跑一次 §5 的完整流程。

---

## 11. 后续可以怎么扩展

1. **时间维度加权**：动作序列里早期 token 更重要（决定大方向），可以给 `D` 个位置加
   `w = [1.0, 0.9, 0.8, ...]` 的权重，只需改 `masked_mean` 的调用处。
2. **feature-level KD**：缓存 teacher 的中间层特征（体积会大很多，需要 fp16 + 降维）。
3. **方案 B（共享骨干双 adapter）**：一份 7B + 两个 LoRA，前向只算一次骨干；
   24 GB 可行，但仍受「初始 KD=0」的限制，需要 warmup。
4. **缓存 key 化**：给 RLDS 每帧生成 `traj_id/frame_idx` 作为 key，就可以支持 shuffle 训练。
