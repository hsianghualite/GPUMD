# LSC-IVR 热导率测试指南

本文档描述如何使用 GPUMD 的 `ensemble lsc_ivr` 计算周期性固体体系的
热导率，并给出可复现的测试流程、输入文件、SLURM 脚本和后处理方法。

适用对象：

- 想在凝聚相/周期性体系中运行 LSC-IVR 的用户；
- 想对比经典 MD、LSC-IVR、RPMD、QTB 热导率的用户；
- 想在单机多 GPU 上并行多个 LSC-IVR 轨迹的用户。

---

## 1. 物理背景

LSC-IVR（Linearized Semiclassical Initial Value Representation）用 Wigner
初始分布生成初始相位点，然后用经典轨迹传播这些相位点。对热导率
Green–Kubo 关联函数，LSC-IVR 近似等价于：

1. 用谐振子 Wigner 分布初始化简正模坐标和动量；
2. 用经典 NVE 动力学传播轨迹；
3. 用 `compute_hac` 计算热流自关联函数；
4. 对多个初始相位点做系综平均，得到热导率。

与经典 MD 相比，LSC-IVR 的初始条件包含零点能和量子化简正模激发，因此在
低温或高频声子明显的体系中更接近量子统计。与 RPMD 不同，LSC-IVR 不引入
虚拟弹簧环聚合物，因此不会出现 RPMD 弹簧势在热流定义和长关联时间中的
问题，但其动力学仍是线性的经典动力学。

---

## 2. 测试体系

本文档使用仓库中的标准测试体系：

- 材料：金刚石结构 Si
- 超胞：3×3×3，216 原子
- 势函数：NEP89
- 目标温度：300 K
- 时间步：0.5 fs
- 周期边界：是
- 热导率方法：Green–Kubo HAC

相关文件位于：

```
tests/gpumd/qct_nep89_si/
├── lsc_ivr_ensemble/       # 单 GPU LSC-IVR 测试
├── lsc_ivr_multigpu/       # 多 GPU LSC-IVR 测试模板
├── classical_kappa/        # 经典 MD 对比
├── rpmd_8beads/            # RPMD 8-bead 对比
├── qtb_kappa/              # QTB 对比
└── ...
```

---

## 3. 单 GPU 测试

### 3.1 输入文件

以下 `run.in` 对应 216 原子 Si/NEP89 周期体系：

```
potential    ../../../../potentials/nep/nep89_20250409/nep89_20250409.txt
time_step    0.5
ensemble     lsc_ivr 300 seed 12345 replicas 1 \
             hessian_displacement 0.001 anharmonic_reweighting no
compute_hac  20 500 10
dump_thermo  100
run          200000
```

参数说明：

- `ensemble lsc_ivr 300`
  - 使用 LSC-IVR 初始化；
  - 目标温度为 300 K；
  - `seed` 指定随机数种子；
  - `replicas 1` 表示每个进程运行一条轨迹；
  - `hessian_displacement 0.001` 指定 Hessian 有限差分位移；
  - `anharmonic_reweighting no` 表示不做非谐重加权，所有权重为 1。
- `compute_hac 20 500 10`
  - 每 20 步采样热流；
  - 最长关联时间对应 500 个采样点；
  - 每 10 个采样点输出一次。
- `dump_thermo 100`
  - 每 100 步输出一次热力学量，用于监测温度和能量漂移。
- `run 200000`
  - 100 ps 生产轨迹，用于 3×3×3 小体系的快速验证。
  - 生产计算建议根据体系尺寸和声子寿命延长。

### 3.2 本地直接运行

```bash
cd tests/gpumd/qct_nep89_si/lsc_ivr_ensemble
/path/to/gpumd < run.in
```

### 3.3 SLURM 运行

已有脚本：

```bash
sbatch tests/gpumd/qct_nep89_si/lsc_ivr_ensemble/lsc_ivr_ensemble.batch
```

该脚本默认使用 1 块 GPU，会依次执行：

1. 打印 GPU、输入文件和结构信息；
2. 运行 GPUMD；
3. 检查 `qct_initial_summary.csv`、`hac.out`、模式频率；
4. 从 `hac.out` 计算三个方向的热导率；
5. 输出 PASS/FAIL 汇总。

---

## 4. 多 GPU 测试（推荐用于生产）

### 4.1 为什么推荐多 GPU

QCT/LSC-IVR 的 native batch 模式为小分子设计，使用 O(N²) 邻居表，
不适合大周期体系。对于固体热导率，应采用“每条轨迹一个 GPUMD 进程”
的方案 A：

- 每个进程 `replicas 1`；
- 走标准 MD 路径和 cell-list 邻居搜索；
- 完整支持周期边界；
- 不同轨迹之间无通信，天然并行；
- 不需要 MPI；
- 合并 HAC 时自动执行 Wigner 加权。

### 4.2 模板输入

`tests/gpumd/qct_nep89_si/lsc_ivr_multigpu/run.in` 是多 GPU 模板：

```
potential    ../../../../potentials/nep/nep89_20250409/nep89_20250409.txt
time_step    0.5
ensemble     lsc_ivr 300 seed 12345 replicas 1 \
             hessian_displacement 0.001 anharmonic_reweighting no
compute_hac  20 500 10
dump_thermo  100
run          2000000
```

`run_multigpu.py` 会为每个轨迹改写：

- `seed`：全局唯一；
- `replicas 1`：强制单轨迹标准路径。

### 4.3 启动命令

以 4 条轨迹、4 块 GPU 为例：

```bash
python3 tools/qct/run_multigpu.py \
  --template tests/gpumd/qct_nep89_si/lsc_ivr_multigpu \
  --gpumd src/gpumd \
  --total-replicas 4 \
  --num-gpus 4 \
  --base-seed 12345 \
  --output merged_output/
```

输出目录结构：

```
replica_000000/
replica_000001/
replica_000002/
replica_000003/
merged_output/
```

合并结果包括：

- `qct_trajectory.xyz`
- `qct_initial_summary.csv`
- `qct_thermo.csv`
- `qct_zpe.csv`（若启用 ZPE 监测）
- `hac.out`（Wigner 加权后的 HAC）

### 4.4 SLURM 脚本

已有脚本：

```bash
sbatch tests/gpumd/qct_nep89_si/lsc_ivr_multigpu/lsc_ivr_multigpu.batch
```

脚本内容要点：

1. 申请 4 GPU；
2. 调用 `run_multigpu.py`；
3. 检查每条轨迹的 `gpumd.log`；
4. 合并 Wigner 加权 HAC；
5. 输出每条轨迹和合并后的热导率。

---

## 5. 热导率后处理

### 5.1 `hac.out` 格式

GPUMD 的 `hac.out` 包含 11 列：

```
time(ps)  hac_xi  hac_xo  hac_yi  hac_yo  hac_z  rtc_xi  rtc_xo  rtc_yi  rtc_yo  rtc_z
```

热导率为运行热导率（running thermal conductivity）：

```
kappa_x = rtc_xi + rtc_xo
kappa_y = rtc_yi + rtc_yo
kappa_z = rtc_z
kappa_avg = (kappa_x + kappa_y + kappa_z) / 3
```

### 5.2 快速分析脚本

```python
import numpy as np

data = np.loadtxt("hac.out")
assert data.shape[1] == 11, data.shape

t = data[:, 0]
kx = data[:, 6] + data[:, 7]
ky = data[:, 8] + data[:, 9]
kz = data[:, 10]
kavg = (kx + ky + kz) / 3

for frac in (0.25, 0.50, 0.75, 1.00):
    i = max(0, int(len(kavg) * frac) - 1)
    print(f"{frac:5.0%}: t={t[i]:8.3f} ps, kappa={kavg[i]:10.4f} W/m/K")

q = kavg[-max(1, len(kavg) // 4):]
print(f"last-quarter mean = {q.mean():.4f} +/- {q.std():.4f} W/m/K")
```

### 5.3 收敛判断

不要只取 `hac.out` 最后一行。推荐同时检查：

1. `kappa(t)` 是否出现平台区；
2. 不同起点、不同轨迹之间的方差；
3. 温度是否稳定在目标温度附近；
4. 总能量漂移是否远小于热流涨落尺度；
5. 对更大超胞或更长轨迹是否收敛。

对于 Si 这类高热导材料，3×3×3 体系通常太小，主要用于代码路径验证；
定量热导率应使用更大超胞（如 4×4×4、6×6×6、8×8×8 或更大）并延长轨迹。

---

## 6. 与经典 MD / RPMD / QTB 的对比

仓库中已提供同一 Si/NEP89 体系的对比测试：

| 方法 | 目录 | 说明 |
|---|---|---|
| 经典 MD | `classical_kappa` | NVT 平衡后 NVE HAC |
| LSC-IVR | `lsc_ivr_ensemble` | Wigner 初始条件 + NVE HAC |
| RPMD 8 beads | `rpmd_8beads` | PIMD 采样 + RPMD 动力学 |
| QTB | `qtb_kappa` | 量子热浴 |
| 多 GPU LSC-IVR | `lsc_ivr_multigpu` | 生产推荐方案 |

启动对比脚本：

```bash
sbatch tests/gpumd/qct_nep89_si/lsc_ivr_kappa.batch
```

建议的对比分析维度：

- 三个方向 `kappa_x/y/z`；
- `kappa_avg(t)` 平台区；
- 温度随时间变化；
- 动能、势能和总能量漂移；
- ZPE 泄漏情况（若启用 `dump_qct ... zpe`）；
- 不同方法的成本和收敛难度。

---

## 7. 稀疏 Hessian 对热导率的影响

### 7.1 精确结构稀疏与数值截断必须区分

有限 cutoff 势的 Hessian 本来就具有块稀疏结构：一个原子的力只依赖
邻域原子。若只是把理论上为零的块从 dense 格式改为 sparse 格式，并完整
保留周期边界、多体势项、质量加权和平移零模，则这是存储格式变化，理论上
不会改变 QCT/LSC-IVR 热导率。

相反，把小元素、跨域耦合或远邻项主动设为零属于近似 Hessian。它会改变

```text
Hessian -> normal modes -> Wigner 初始相空间 -> HAC -> thermal conductivity
```

因此可能改变低频声学模式、声速、初始能量分布、Wigner weight 以及 HAC
长时间尾部。不能因为被删除的元素数值较小，就直接认为热导率不受影响。

### 7.2 四种可扩展方案的取舍

| 方法 | 优点 | 对热导率的主要风险 |
| --- | --- | --- |
| 精确 sparse Hessian | 保留全局耦合，内存约为非零块数量；物理上可等价 dense | 需要 sparse eigensolver，完整特征分解仍很昂贵 |
| Matrix-free Davidson/Block-Lanczos | 只存 `O(kD)` 向量，最适合 5 万原子低频模式 | 需要 Hessian-vector product 和收敛验证 |
| Domain decomposition | 子域小、易并行，适合局部反应/缺陷 | 丢失跨域耦合、长波声子和全局声学模式 |
| 有限 active modes | 最省显存、实现简单 | 不再是完整 Wigner sampling，可能产生系统性偏差 |

如果只计算 Hessian 的低频 `k` 个模式，必须明确标注为 reduced-mode
近似，不能直接宣称与 full QCT/LSC-IVR 等价。后续动力学仍使用完整势函数，
也只能保证力场正确，不能保证初始量子分布没有偏差。

### 7.3 热导率验收方法

在较小体系上比较 dense 与 sparse exact 结果，并对数值截断或 active modes
做收敛测试。至少检查：

- Hessian 最大绝对差和低频本征值相对差；
- 三个平移零模、声速和低频态密度；
- Wigner 初始能量、权重和 `N_eff`；
- `hac.out` 的完整 HAC 曲线和热导率平台，而不只比较末点；
- 多副本之间的统计误差和总能量漂移。

建议的 active-mode 序列为 `k=256, 512, 1024, 2048`。只有当热导率变化
小于统计误差时，才能接受相应截断。对 5 万原子体系，推荐先实现精确
sparse Hessian-vector product，再接 matrix-free Davidson；domain
decomposition 只作为局部振动近似，不作为周期热导率默认方案。

## 8. 常见问题

### 8.1 LSC-IVR 是否能用于固体热导率？

可以用于以 Green–Kubo 热流自关联为核心的近似量子校正计算。它保留了
Wigner 初始分布中的零点能和模式激发，动力学仍为经典线性动力学，因此
属于半经典近似，不是精确的量子热输运。

### 8.2 为什么我的热导率偏低？

可能原因包括：

- 体系太小，声子平均自由程超过盒子长度；
- 关联时间不够长，`kappa(t)` 尚未进入平台区；
- 采样轨迹太少；
- Hessian 求解的简正模质量不佳；
- 后处理取了尚未收敛的末端值；
- 输入结构未充分平衡；
- 时间步过大导致能量漂移。

### 8.3 为什么经典 MD 的动能较低？

经典 MD 在低温下没有零点能，动能对应 `3N k_B T / 2`；LSC-IVR、RPMD
和 QTB 会以不同方式引入量子能量，因此动能通常高于经典 MD。高温极限
下几种方法的动能会趋近。

### 8.4 多 GPU 是否需要 MPI？

不需要。`run_multigpu.py` 使用进程级并行，每个 GPU 运行独立的
GPUMD 进程，轨迹间无通信，结果在运行结束后合并。

### 8.5 什么时候需要空间分解？

当前多 GPU 方案按轨迹划分而不是按空间划分。单个 GPUMD 进程内部仍
使用 GPU 标准邻居搜索。对单条轨迹非常大的体系，如果单卡显存或吞吐
成为瓶颈，才需要考虑空间分解；当前 LSC-IVR 生产路径优先采用方案 A。

---

## 9. 最小检查清单

运行完成后，请依次检查：

```text
[ ] gpumd exit code 为 0
[ ] gpumd.log 无 CUDA / 输入错误
[ ] qct_initial_summary.csv 存在且权重有限
[ ] 简正模频率合理（无非物理大频或复频）
[ ] thermo.out 温度稳定
[ ] 总能量漂移可接受
[ ] hac.out 存在且为 11 列
[ ] kappa(t) 进入或接近平台
[ ] 多轨迹间差异在统计误差内
[ ] 与经典 MD/RPMD/QTB 的差异可解释
```

---

## 10. 快速命令汇总

```bash
# 单 GPU
cd tests/gpumd/qct_nep89_si/lsc_ivr_ensemble
/path/to/gpumd < run.in

# 单 GPU SLURM
sbatch lsc_ivr_ensemble.batch

# 多 GPU
python3 tools/qct/run_multigpu.py \
  --template tests/gpumd/qct_nep89_si/lsc_ivr_multigpu \
  --gpumd src/gpumd \
  --total-replicas 4 \
  --num-gpus 4 \
  --base-seed 12345 \
  --output merged_output/

# 多 GPU SLURM
sbatch tests/gpumd/qct_nep89_si/lsc_ivr_multigpu/lsc_ivr_multigpu.batch

# 热导率快速查看
python3 - <<'PY'
import numpy as np
d = np.loadtxt("hac.out")
kx = d[:, 6] + d[:, 7]
ky = d[:, 8] + d[:, 9]
kz = d[:, 10]
ka = (kx + ky + kz) / 3
print("kappa_end =", ka[-1])
print("kappa_last_quarter =", ka[-len(ka)//4:].mean(), "+/-", ka[-len(ka)//4:].std())
PY
```

---

## 11. 参考资料与相关文档

- `tools/qct/MULTIGPU_GUIDE.md`
- `tools/qct/QCT_LSC_IVR_TEST_REPORT.md`
- `tools/qct/SEMICLASSICAL_ROADMAP.md`
- `tools/qct/SC_IVR_FBTS_PLAN.md`
- `doc/gpumd/input_parameters/ensemble_lsc_ivr.rst`
- `docs/lsc_ivr.md`
