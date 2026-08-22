# QCT/LSC-IVR 测试报告

日期：2026-08-10  
分支：`qct`  
测试平台：SAI 集群 V100 GPU 节点

## 1. 测试环境

- 登录节点：`login-01.mr-sai.ai`
- 计算节点：`4v100n22`
- GPU：Tesla V100-SXM2-32GB
- CUDA：12.4.1
- 远程工作目录：`~/workdir_sinano-fanzhaochuan/yishangzhao/gpumd-qct-review`
- 测试代码由本地 `qct` 工作树同步到远程目录后执行。

## 2. 编译验证

执行：

```bash
module load cuda/12.4.1
make -C src -j8 gpumd
```

结果：通过。输出显示 `gpumd` 已是最新目标，没有编译错误。

## 3. Python 回归测试

执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/gpumd/qct
```

结果：

```text
34 passed in 7.69s
```

覆盖范围包括 QCT 分析器、LSC-IVR 相关统计、权重处理、轨迹校验、批量基准工具和错误输入处理。

## 4. GPU QCT batch 回归

测试输入：

```text
tests/gpumd/qct_nep89_oh/batch/run.in
```

关键配置：

- QCT canonical sampling
- `temperature 0`
- `replicas 2`
- `phase random`
- `dump_qct 1`
- `run 5`

由于该输入是 2 原子小体系，4 GPU 分配会触发 GPUMD 的小体系保护提示。因此在 4 GPU 作业中固定使用：

```bash
CUDA_VISIBLE_DEVICES=0
```

结果：通过。

- GPUMD 返回码：`0`
- QCT batch 成功生成：`2 replicas`
- 完成：`5/5 steps`
- 运行耗时：约 `0.256 s`
- 速度：约 `887 atom*step/s`
- 日志确认完成 harmonic QCT 初始化、独立 replica 采样和轨迹输出。

## 5. 失败与修正记录

### 5.1 初始 GPU 作业请求

使用 1 GPU 的默认 QOS 请求被 Slurm 拒绝：

```text
QOSMinGRES
```

原因是默认 QOS 要求至少 4 个 GPU。随后改用 `4V100` 分区申请 4 GPU，并在作业内通过 `CUDA_VISIBLE_DEVICES=0` 让小体系只使用单卡。

### 5.2 首次输出检查

首次 shell 检查额外要求 `qct_zpe.csv` 存在，导致作业步骤返回 1。该输入仅使用 `dump_qct 1`，没有启用 ZPE 输出，因此该文件不存在是预期行为；GPUMD 本身返回码为 0。该检查错误不属于代码或运行时失败。

## 6. 未覆盖项目

当前 SAI 回归未完成以下项目：

- 长时间 QCT/LSC-IVR 物理轨迹稳定性；
- 真实多进程、多 GPU replica 合并等效性；
- `compute-sanitizer` 内存检查；
- MDI 目标在 GPU 节点上的独立运行；
- 周期性体系和大规模 NEP89 生产测试。

这些项目需要更长的 GPU 作业或专门的集群测试配额。

## 7. 总结

本次测试确认：

1. `qct` 分支可在 SAI CUDA 12.4.1/V100 环境编译；
2. QCT Python 测试套件全部通过；
3. 原生 QCT batch 在真实 GPU 上可完成初始化、采样、积分和输出；
4. 当前发现的失败均来自 Slurm 资源策略或测试脚本输出断言，不是 GPUMD 运行时错误。

## 8. 本地 CUDA 实现回归（2026-08-18）

针对 Full-GPU Hessian 实现补充了本地验证：

- `git diff --check`：通过；
- `make -C src -j4 gpumd`：通过；
- `make -C src -f makefile_mdi -j4`：通过；
- QCT Python 测试：`53 passed in 30.70s`；
- `nvcc`：CUDA 12.0；
- `nvidia-smi`：可见 `NVIDIA RTX A2000 Laptop GPU, 8192 MiB`。

尝试运行 216 原子周期 Si 自动 Hessian 时，GPUMD 在
`cudaGetDeviceCount()` 处返回：

```text
CUDA Error code: 100
no CUDA-capable device is detected
```

因此本次未能执行 GPU kernel、`Dsyevd`、显存预检和真实 Hessian 数值回归。
`nvidia-smi` 可见设备但 CUDA runtime 不可见，属于当前进程/驱动访问环境问题，
不是 GPUMD 编译错误。需要在允许 CUDA device node 的进程环境中重新运行 GPU 回归。

## 9. Hessian 日志精简

针对 6x6x6 测试中按 1% 输出导致约 100 条进度日志的问题，默认进度间隔改为
自适应约 12 个更新点。显式 `hessian_progress_interval N` 仍可覆盖默认值；首列、
末列和阶段摘要始终保留。

## 10. 独立副本与 HAC 合并回归（2026-08-18）

本地对 `run_multigpu.py` 的进程级独立副本流程进行了确定性测试：

- 定向测试：`11 passed in 3.57s`；
- 完整 QCT Python 测试：`64 passed in 31.52s`；
- Python 语法检查和 `git diff --check`：通过；
- 五个副本使用全局 seed `12345` 至 `12349`，并在两个可见设备 token
  上调度；
- 每张 GPU 使用独立串行任务队列，测试确认同一设备不会同时启动两个进程；
- 验证 resume 直接复用有效结果，并在模板哈希变化后重新运行；
- 验证失败进程、进程启动错误、陈旧产物、错误 seed、缺失完成标记和错误 HAC
  schema 均返回失败；
- 验证极端 log weight、有效样本数、完整 HAC 曲线加权、时间网格一致性、
  acceptance 诊断和原子文件替换；
- manifest 记录 executable、model、共享 eigenvector 及所有外部输入的 SHA-256。

上述编排测试使用临时 fake-GPUMD 可执行文件，不属于真实 CUDA 数值或 5 ns
物理收敛测试。真实生产运行前仍需完成五条短独立轨迹的 GPU smoke test，并检查
`N_eff >= 3` 与最大归一化权重 `<= 0.5`。

## 11. 本地重新编译与启动测试（2026-08-19）

在 `qct` 分支执行强制重编译：

```text
make -C src -B -j4 gpumd       -> success
make -C src -f makefile_mdi -B -j4 gpumd-mdi -> success
```

随后运行 `tests/gpumd/qct`，结果为 `64 passed in 30.00s`；
`git diff --check` 通过。`src/gpumd` 和 `src/gpumd-mdi` 均能启动并输出版本及
编译选项信息，但在 CUDA 初始化阶段返回：

```text
CUDA Error code: 100
no CUDA-capable device is detected
```

因此本次环境未能执行真实 GPU kernel、自动 Hessian、显存策略或物理轨迹测试。
该失败发生在 CUDA runtime 设备探测阶段，两个目标的编译和链接本身均成功；需要
在 GPU 设备节点/允许访问 `/dev/nvidia*` 的 shell 中重新运行真实 smoke test。

## 13. 原生 batch 加权 HAC 修复（2026-08-19）

修复了 native `replicas > 1` 的 HAC 统计路径：

- 每个副本独立归约热流并计算 HAC/RTC，消除副本之间的交叉相关项；
- 使用 `log_wigner_weight` 做 log-sum-exp 归一化后合并 `hac.out`；
- `anharmonic_reweighting no` 使用等权平均；
- 新增 `hac_replica.out` 和 `hac_reweighting.csv` 审计输出；
- `replicas 1` 仍输出条件 HAC，由 `run_multigpu.py` 在独立进程之间完成唯一一次权重归一化；
- native periodic batch 仍保持禁用，周期热导率继续使用独立进程方案。

验证结果：

- `tests/gpumd/qct/test_run_multigpu.py`：`12 passed`；
- 完整 QCT Python 套件此前通过 `65 passed`；
- `src/measure/hac.cu` 已成功编译为 `src/measure/hac.o`；
- 本次最终链接受当前 shell 缺少 CUDA 开发链接库（`-lcublas`、`-lcusolver`、
  `-lcufft`、`-lcudart_static`）影响，未宣称完整可执行文件重链成功。

## 12. WSL CUDA 环境修复验证（2026-08-19）

新增 `tools/qct/run_wsl_cuda.sh`，并让 `run_multigpu.py` 和 LSC-IVR 示例脚本
统一使用以下策略：

- 清除 `CUDA_VISIBLE_DEVICES`（单轨迹 wrapper）或由 launcher 重新设置为当前副本设备；
- 清除 `NVIDIA_VISIBLE_DEVICES`；
- 将 `/usr/lib/wsl/lib` 放到 `LD_LIBRARY_PATH` 首位。

环境单元测试通过：`12 passed`。使用 stale GPU 环境变量启动真实小 QCT 输入后，
GPUMD 已不再返回 `error 100 (no CUDA-capable device)`，而是进入 CUDA driver
初始化并返回 `error 35 (CUDA driver version is insufficient for CUDA runtime
version)`。这证明设备可见性和 WSL driver bridge 配置已生效；当前 shell 剩余问题是
CUDA 12.0 runtime 与 WSL Windows driver 版本不匹配，需要在 GPU 驱动支持的 toolkit
环境中重新链接/运行，不能由 GPUMD 输入脚本规避。
