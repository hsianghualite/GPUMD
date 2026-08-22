# Full-GPU QCT/LSC-IVR Hessian 实施计划

本计划基于 `FULL_GPU_HESSIAN_DESIGN.md`，目标是为 CUDA 周期体系的自动
QCT/LSC-IVR Hessian 建立完整的 device-resident 路径，同时保留孤立分子
的现有 CPU 路径和旧输入兼容性。

当前本地实现已完成阶段 1、阶段 2 的主体以及阶段 3 的 GPU 模态位置/速度
累加入口；审计文件仍使用现有 host 结果接口，ZPE 投影和完整 host-copy 消除
需要后续增量完成。本文档中的验收项尚未宣称通过。

## 目标与边界

- CUDA + PBC 自动 Hessian：有限差分、Hessian 组装、质量加权、FP64 对称
  eigensolve、模态重建和可选 ZPE 投影使用 GPU 缓冲。
- 每个有限差分列保留在 GPU：两次 force evaluation 之间只做 device-to-device
  位置更新，不把力向量复制到 host。
- CPU RNG、seed/retry 规则和现有输出格式保持不变；只复制最终 replica 位置/速度
  和必要的标量结果到 host。
- 非周期分子继续使用当前六刚体模态 CPU 逻辑；HIP 首版继续使用旧路径并明确打印
  后端状态，不静默回退。

## 实施阶段

1. **基础 API 与 GPU kernel**
   - 增加 `Molecular_Hessian_Options`、进度选项和 device 结果所有权。
   - 增加位移、差分、对称化、质量加权、归约、模态累加 kernel。
   - 增加 CUDA 显存查询、FP64 GEMM 和 `Dsyevd` wrapper。

2. **设备有限差分与求解**
   - 重写周期 Hessian 分支，复用 `Atom::position_per_atom` 和
     `Atom::force_per_atom`，结束时恢复并重新计算 reference force。
   - 完成 workspace/显存预检，solver `info` 检查和确定性的平移模态识别。
   - 保留旧 CPU 分支作为非 PBC/测试参考路径。

3. **QCT 集成**
   - 新增 `hessian_progress yes|no` 和
     `hessian_progress_interval N` key-value 输入。
   - 对 device eigenvectors 使用 GPU GEMM 完成 Q/P 到笛卡尔位置/速度的累加，
     仅复制最终 phase points；反应模式最多复制一个向量。
   - `dump_qct ... zpe` 直接消费 device mode basis；审计文件分块写出。

4. **验证与启用**
   - 增加 CPU/GPU 数值对比、PBC 非等质量、孤立分子回归、进度日志、显存泄漏和
     compute-sanitizer 测试入口。
   - 只有周期矩阵、平移模态、模式正交性、输出和内存验收通过后，才默认启用 CUDA
     PBC 路径。

## 固定接口和验收标准

- `hessian_displacement` 继续有效；默认进度输出开启。
- 进度按 `hessian_progress_interval` 输出，默认自适应为约 12 个更新点，首列和末列强制输出；包含 force evaluation 计数、elapsed 和 ETA。
- Hessian CPU/GPU 最大绝对差 `<=1e-8 eV/A^2`；非零本征值相对差 `<=1e-6`；
  `V^T V` 残差 `<=1e-8`；acoustic sum-rule 残差 `<=1e-6`。
- 不允许部分分配后静默转 CPU；错误必须报告 requested/available bytes。
- 设计要求的 SAI/GPU、长轨迹、compute-sanitizer 和 6x6x6 benchmark 在实现后单独执行；
  本次修改阶段不运行编译或 GPU 作业。

## 文件范围

- `src/phonon/molecular_hessian.cu/.cuh`
- `src/utilities/cusolver_wrapper.cu/.cuh`
- `src/utilities/gpu_macro.cuh`
- `src/integrate/ensemble_qct.cu/.cuh`
- `src/measure/dump_qct.cu/.cuh`
- `tests/gpumd/qct/` 及对应 QCT 文档
