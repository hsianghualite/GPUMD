# NEP_EFA Integration — TODO & Issues

## Overview

Implementing equivariant EFA (Euclidean Fast Attention) into GPUMD's NEP framework as an additive energy term `E_total = E_NEP + E_EFA`, following the NEP_Charge integration pattern, on both training and MD sides.

**Status as of 2026-07-23:** Both `nep` (training) and `gpumd` (MD) binaries compile successfully. EFA now uses a shared ERoPE equivariant-channel descriptor and power-spectrum scalar head on both paths. Runtime validation still requires a CUDA device.

---

## Completed Work

### Training side (`src/main_nep/`)
- [x] `efa_utilities.cuh` — EFA math primitives (Complex, Ylm, spherical Bessel, CG coefficients, ERoPE kernel)
- [x] `nep_efa.cuh` / `nep_efa.cu` — Training NEP_EFA class (composition with NEP)
- [x] `parameters.cuh` / `.cu` — EFA parameter fields + `q_scaler_efa`
- [x] `fitness.cu` — `write_nep_txt` with EFA serialization
- [x] `snes.cu` — EFA `type_of_variable` mapping
- [x] ERoPE descriptor — `A(l,m,r)` channels followed by `sum_m |A|^2`
- [x] Training force path — shared ERoPE definition with finite-difference pair derivatives
- [x] Compiles successfully (`make nep` produces `nep` binary)

### MD side (`src/force/`)
- [x] `nep_efa.cuh` — `NEP_EFA_Data` struct, `NEP_EFA` class with `NEP nep;` member, `ParaMB`/`ANN`/`ExpandedBox`/`EFA_Para`/`Small_Box_Data` structs
- [x] `nep_efa.cu` — ERoPE descriptor, ANN, structure factor, reciprocal-space forces, and pairwise ERoPE back-prop for large/small boxes
- [x] `nep.cuh` — `Small_Box_Data` struct moved to public; `paramb`, `annmb`, `zbl`, `ebox` moved from private to public; added `bool is_efa = false` to `ParaMB`
- [x] `nep.cu` — Header parsing recognizes `nep4_efa`/`nep4_zbl_efa`; skips `efa ...` hyperparameter line; reads NEP params, then skips EFA params block + EFA q_scaler
- [x] `force.cu` — Registered `NEP_EFA` dispatch for `nep4_efa`/`nep4_zbl_efa` headers; added `#include "nep_efa.cuh"`
- [x] `nep.cu` — skips the actual EFA parameter block and EFA scaler dimensions
- [x] Compiles successfully (`make -C src` produces both binaries)

---

## TODO

## Equivariant EFA MD Acceleration

The current EFA MD path uses ERoPE equivariant channels followed by a power
spectrum and a scalar ANN. Its dominant cost is the ERoPE force back-propagation
and the direct reciprocal-space sum; the ANN itself is small.

### Current EFA Hotspots

- [ ] `find_force_erope_large_box` and `find_force_erope_small_box` rebuild the
  full channel accumulator `A(l,m,r)` by looping over all neighbors for every
  target pair. This gives approximately `O(N z^2 R L^2)` work.
- [ ] The current force derivative uses central differences and evaluates
  positive/negative displacements for all three Cartesian directions. Replace
  it with the analytic derivative of the ERoPE kernel, using the existing
  spherical-Bessel derivative helper and analytic angular derivatives.
- [ ] Store the forward `A(l,m,r)` values for reuse by the force kernel. For
  the default `l_max=3, num_radial=4`, the cache is small enough to make this
  a practical first optimization. Use tiled storage for larger dimensions.
- [ ] Exploit the real spherical-harmonic basis. The current complex ERoPE
  path recomputes radius, angle, spherical harmonics, and Bessel values for
  every `m`; compute geometry once per pair and evaluate all `m` channels as a
  tile.
- [ ] Use the `m < 0` conjugate symmetry or a real basis to halve channel
  storage and arithmetic where the coefficient constraints permit it.
- [ ] Replace one-thread-per-atom neighbor loops with warp-per-atom or
  warp-per-`(l,r)` reductions. Avoid register/local-memory spills from the
  channel accumulators and ANN derivative arrays.

### EFA Reciprocal-Space Operators

- [ ] Cache the `k/G` mesh when the simulation box is unchanged. The current MD
  path rebuilds it on every EFA pass and may copy the mesh from host to device.
- [ ] Tile the direct structure-factor and reciprocal-force kernels over atoms
  and k-points. Use shared-memory `k/G` tiles, packed `float2/float4` data, and
  `__sincosf` for phase evaluation.
- [ ] Replace the one-thread-per-k-point structure-factor loop and the
  one-thread-per-atom k-point force loop with 2D tiled reductions when the
  `N*K` product is large.
- [ ] Replace the single-block neutrality and `D_real` mean reductions with
  scalable two-stage/CUB reductions for large structures.
- [ ] Add an automatic direct-Ewald versus PPPM/PME dispatch. Direct summation
  is preferable for small systems; a cuFFT-based mesh can reduce large-system
  reciprocal work from `O(NK)` toward `O(N log N)` at controlled numerical
  error.
- [ ] Keep reciprocal-space box updates asynchronous where possible and avoid
  CPU reconstruction of a fixed mesh inside the MD step.

### EFA Fusion and Precision

- [ ] Fuse ERoPE descriptor, power-spectrum contraction, and ANN evaluation
  when the descriptor cache is not needed by diagnostics. Otherwise retain a
  cache for force reuse and fuse only the ANN stage.
- [ ] Pre-scale ANN input weights by `q_scaler_efa` to remove a per-atom scaling
  pass and reduce temporary descriptor arrays.
- [ ] Benchmark `rsqrtf`, `__sincosf`, and `__tanhf` before considering inline
  PTX. Inline assembly should only be used after SASS inspection proves that
  the compiler fails to emit the desired instruction.
- [ ] Evaluate FP32 local channel accumulation with double-precision final
  force/virial accumulation. Validate force error and long-time NVE drift.
- [ ] Consider CUDA Graphs for fixed-box small systems, where descriptor,
  reciprocal, and force kernel launch overhead is significant.
- [ ] Remove EFA device debug `printf` calls before performance measurement;
  device-side printf can serialize or distort the MD step.

### EFA Validation and Scaling

- [ ] Benchmark descriptor, force back-propagation, reciprocal structure factor,
  reciprocal force, and total MD step separately with Nsight Systems/Compute.
- [ ] Compare the cached/analytic force path against the central-difference
  baseline using finite-difference forces and rotational covariance tests.
- [ ] Validate energy, force, virial, and NVE drift after every fast-math or
  mixed-precision change.
- [ ] Investigate a dedicated multi-GPU EFA path. The existing multi-GPU NEP
  implementation does not automatically cover EFA's global reciprocal
  structure factor, which requires a cross-device reduction of `S(k)`.

## Ordinary NEP MD Acceleration

This section covers the standard NEP path without EFA. The main cost is the
angular descriptor/force pipeline, not the small ANN head. The current flow is
neighbor-list filtering -> fused descriptor + ANN -> radial force -> angular
partial force -> many-body force reduction.

### Current Hotspots

- [ ] `find_descriptor` recomputes angular pair geometry and radial basis values
  inside every angular-order loop. Reorder the loops so MIC, distance, cutoff,
  and Chebyshev basis values are evaluated once per pair and reused across
  angular orders.
- [ ] `find_partial_force_angular` uses large per-thread arrays such as
  `Fp[MAX_DIM_ANGULAR]` and `sum_fxyz[NUM_OF_ABC * MAX_NUM_N]`. Check register
  usage and local-memory spills; tile angular orders/channels through shared
  memory or split the kernel by descriptor order.
- [ ] `gpu_find_force_many_body` performs a binary search for the reverse edge
  of every angular neighbor pair. Build a reverse-edge index when the neighbor
  list is rebuilt, reducing this from `O(N z log z)` to `O(N z)`.
- [ ] Large-box NEP recomputes MIC, distance, and cutoff information in the
  descriptor and force kernels after neighbor-list construction. Evaluate a
  reusable pair-geometry cache versus the additional memory traffic.
- [ ] The one-thread-per-atom mapping serializes all neighbors. Benchmark a
  warp-per-atom or warp-per-angular-tile mapping for high coordination and
  small/medium systems.

### Neighbor-List and Host Synchronization

- [ ] `Neighbor::find_neighbor_global` checks atom displacement with a device
  kernel followed by a device-to-host copy on every force call. Replace this
  with a device-side/graph conditional path or a conservative check interval
  based on the skin and maximum displacement.
- [ ] The global Verlet list uses a skin, but radial/angular exact lists are
  filtered and written every step. Benchmark retaining skin-expanded radial
  and angular lists and applying the exact cutoff inside descriptor/force
  kernels.
- [ ] Disable diagnostic neighbor statistics in production. The current
  `neighbor.out` path copies neighbor counts to the host and writes a file
  periodically, which can synchronize the MD loop.
- [ ] Tune the skin distance automatically from rebuild cost versus extra
  neighbors rather than assuming the current fixed value.

### Operator and Memory Optimizations

- [ ] Generate specialized kernels for common fixed model dimensions
  (`L_max`, `n_max`, and `basis_size`) so loops can be unrolled and local
  arrays can use actual sizes instead of `MAX_DIM` limits.
- [ ] Use `rsqrtf`/`__sincosf`/`__tanhf` or fast-math variants only after force
  and energy tolerances are measured. The cutoff derivative currently computes
  sine and cosine separately in `find_fc_and_fcp`.
- [ ] Use read-only/vectorized coefficient loads where profiling shows cache
  misses. Keep the existing type-pair coefficient layout consistent with
  coalesced access.
- [ ] Consider spatial atom sorting to improve position, type, and neighbor
  locality. Preserve an inverse permutation for output buffers.
- [ ] Use CUDA Graphs to reduce launch overhead for small systems and fixed
  MD pipelines. The ordinary NEP path launches separate descriptor, radial,
  angular, reduction, and optional ZBL kernels.
- [ ] Build for the actual GPU architecture. The legacy Makefile defaults to
  `sm_60`; use the CMake `native` architecture or an explicit `sm_80/sm_89/sm_90`
  target where appropriate.

### Optional Approximate or Multi-GPU Paths

- [ ] Evaluate mixed precision for pair geometry and intermediate descriptor
  accumulations while retaining double-precision output force accumulation.
  Validate energy, force, and long-time NVE drift before enabling it.
- [ ] Use the existing `NEP_MULTIGPU` path for sufficiently large systems;
  verify that the box length and halo communication amortize peer-transfer
  overhead.
- [ ] For small independent systems, batch multiple replicas/configurations
  or use persistent CUDA Graphs to improve GPU occupancy.
- [ ] Consider Tensor Core/GEMM acceleration only after batching coefficient
  contractions by type pair. The current fused ANN is too small and the
  neighbor/angular operations are irregular for direct Tensor Core use.
- [ ] Keep inline PTX as a last-mile optimization. First compare compiler
  output for CUDA intrinsics (`rsqrtf`, `__sincosf`, `__tanhf`, `__ldg`) and
  inspect SASS before introducing architecture-specific assembly. Provide a
  HIP/fallback implementation if PTX is added.

### Validation and Profiling

- [ ] Profile `find_descriptor`, `find_force_radial`,
  `find_partial_force_angular`, `gpu_find_force_many_body`, and neighbor-list
  construction with Nsight Systems/Compute.
- [ ] Record kernel time, end-to-end steps/s, achieved occupancy, register and
  local-memory traffic, memory throughput, and host-device synchronization.
- [ ] For every numerical optimization, compare energy/force tolerances and
  long-time NVE energy drift against the baseline model.

### High Priority — Correctness

- [ ] **Run smoke test with a real `nep4_efa` nep.txt file** (requires CUDA device)
  - Generate a dummy nep.txt with `nep4_efa` header
  - Run a few MD steps on a simple system
  - Verify no segfault / NaN

- [ ] **Finite-difference force/energy consistency check** (requires CUDA device)
  - Compare analytic forces against numerical derivatives of energy
  - Test both small-box and large-box paths
- [ ] **Validate rotational covariance numerically** by rotating a structure and comparing energy/rotated forces

### Medium Priority — Performance & Robustness

- [x] Replaced the legacy NEP radial/angular EFA calls with the ERoPE path
- [ ] Remove now-unused legacy kernels from the EFA translation units (cleanup only)

- [ ] **Optimize `zero_total_attention_single` and `zero_mean_D_real_single`**
  - Currently `<<<1, 1024>>>` single-block reduction
  - For large N (>1024), this serializes batches
  - Consider multi-block reduce + final atomic, or just accept for now (single-structure use case)

- [ ] **Cache `find_k_and_G` results if box doesn't change**
  - Currently recomputes k-mesh every MD step
  - If box is fixed (NVT/NVE), k-mesh is constant → cache it
  - Add a `box_changed` flag

- [ ] **Verify `efa_para.alpha` k-space cutoff formula** (see Issue #4)
  - `ksq_max = (2π * alpha)^2`
  - `G = |1/det| / ksq * exp(-ksq * alpha_factor)` where `alpha_factor = 1/(4*alpha^2)`
  - Compare with NEP_Charge's Ewald implementation in `ewald.cuh` / `nep_charge.cu`

### Low Priority — Cleanup

- [ ] **Remove unused `ExpandedBox ebox` member** (nep_efa.cuh:145)
  - Declared but never used in current implementation

- [ ] **Remove unused `efa_para.omega_min`, `efa_para.num_omega`** (nep_efa.cuh:126-127)
  - Set but never used in MD-side kernels

- [ ] **Add documentation for `nep4_efa` / `nep4_zbl_efa` in `doc/nep/`**

- [ ] **Add example input files in `examples/`**

- [ ] **End-to-end training run** to produce a real nep.txt with EFA params, then feed to `gpumd`

---

## Issues

The legacy radial/angular EFA kernels discussed below are no longer on the
active execution path. Their virial concerns are superseded by the ERoPE
pairwise force kernel; runtime and rotational finite-difference checks remain
open because this environment has no CUDA device.

### Issue #1 — Legacy radial EFA virial pattern (superseded)

**Location:** `src/force/nep_efa.cu:391-482`

**Initial concern:**
The kernel accumulates forces to both n1 (direct) and n2 (atomicAdd), but virial only to n1.

**Analysis (corrected):**
This is correct. Each pair (n1,n2) is visited twice in the large-box path:
1. n1 as center: force +f12 to n1, -f12 to n2, virial `-r12*f12` to n1
2. n2 as center: force +f12' to n2, -f12' to n1, virial `r12*f12'` to n2

Total virial = `-r12*f12 + r12*f12' = -r12*(f12-f12')` = standard pair virial. ✓

The pattern differs from original NEP large-box (which uses `f12-f21` difference form and no atomicAdd), but is mathematically equivalent.

**Severity:** None — not a bug.

---

### Issue #2 — Legacy small-box radial virial pattern (superseded)

**Location:** `src/force/nep_efa.cu:488-567`

**Problem:**
The kernel accumulates virial via `atomicAdd(&g_virial[n2+...])` only, matching the original NEP small-box pattern. However, the original NEP small-box stores virial to n2 because it also does `atomicAdd` for forces to both n1 and n2.

**Status:** Appears consistent with original. **Needs verification** that the sign convention of `f12` matches (EFA: `+f12` to n1, `-f12` to n2; original: same).

**Severity:** Low — likely correct, needs confirmation.

---

### Issue #3 — Legacy reciprocal virial mapping (superseded)

**Location:** `src/force/nep_efa.cu:369-377`

**Initial concern:**
Virial index mapping appeared incorrect.

**Analysis (corrected):**
The `temp_virial_sum` layout is `[xx, yy, zz, xy, yz, zx]` (NOT `[xx, yy, zz, xy, xz, yz]` as initially assumed):
```
sum[0]=xx (kx*kx)   sum[3]=xy (kx*ky)   sum[5]=zx (kz*kx)
sum[1]=yy (ky*ky)   sum[4]=yz (ky*kz)
sum[2]=zz (kz*kz)
```

The 9-element virial layout is `[0]=xx [3]=xy [4]=xz / [6]=yx [1]=yy [5]=yz / [7]=zx [8]=zy [2]=zz`.

Code mapping (all correct):
- `[4]=xz ← sum[5]` (zx=xz, symmetric) ✓
- `[5]=yz ← sum[4]` ✓
- `[7]=zx ← sum[5]` ✓
- `[8]=zy ← sum[4]` (yz=zy, symmetric) ✓

**Severity:** None — not a bug. Initial analysis used wrong layout assumption for `temp_virial_sum`.

---

### Issue #4 — Verify Ewald k-space formula

**Location:** `src/force/nep_efa.cu:1090, 1107, 355`

**Formulas used:**
- `ksq_max = (2π * alpha)^2` (line 1090)
- `G = |1/det| / ksq * exp(-ksq / (4*alpha^2))` (line 1107)
- `alpha_k_factor = 2/(4*alpha^2) + 2/ksq` in virial (line 355)

**Need to verify:**
1. The `G(k)` formula matches the EFA energy definition `E = sum_k G(k) |S(k)|^2`
2. The virial `alpha_k_factor` derivation is correct
3. Compare with NEP_Charge's Ewald implementation (`src/force/ewald.cuh`, `src/force/nep_charge.cu`)

**Status:** Formula looks physically reasonable (standard Ewald-like), but needs cross-check against training-side implementation (`src/main_nep/nep_efa.cu:481-554`) — which uses identical formula, so internal consistency is OK. Need external validation.

**Severity:** Medium — affects energy/virial magnitude but internally consistent between training and MD.

---

### Issue #5 — Constructor re-opens nep.txt, may have parsing fragility

**Location:** `src/force/nep_efa.cu:933-988`

**Problem:**
The NEP member constructor reads nep.txt first (consuming the stream). Then NEP_EFA re-opens the file from scratch and re-parses from the beginning to extract EFA parameters. This is fragile:
- If the file format changes, both parsers must be updated in sync
- The skip logic (`for i in 0..5: get_tokens`) assumes exact line count
- The ZBL line skip (`if tokens[0] == "nep4_zbl_efa": tokens = get_tokens(input)`) assumes ZBL line is always present for zbl_efa models

**Status:** Works for current format. Consider refactoring to have NEP_EFA parse everything in one pass.

**Severity:** Low — works now, fragile to format changes.

---

### Issue #6 — No runtime validation yet

**Problem:**
Only compilation has been verified. No actual MD run with a `nep4_efa` potential file has been attempted.

**Risks:**
- Segfault from incorrect buffer sizes
- NaN from incorrect kernel logic
- Incorrect physics (wrong virial/forces)
- `box.cpu_h` layout assumption in `find_k_and_G` may be wrong

**Plan:**
1. Create a minimal `nep4_efa` nep.txt (can use random parameters for smoke test)
2. Set up a simple 8-atom cubic cell
3. Run 1 MD step with `velocity 0` and check energy/forces are finite
4. If finite-difference check is desired, perturb one atom and compare

**Severity:** High — must validate before merging.

---

### Issue #7 — No finite-difference force validation

**Problem:**
Analytic forces have not been validated against numerical energy derivatives.

**Plan:**
1. Run `gpumd` with a fixed configuration, record energy E0
2. Perturb atom i by +dx, record E1
3. Perturb atom i by -dx, record E2
4. Numerical force = -(E1 - E2) / (2*dx)
5. Compare with analytic force from `gpumd`
6. Repeat for multiple atoms and both small/large box paths

**Severity:** High — critical for correctness.

---

### Issue #8 — `find_force_angular_efa_reduce` kernel is dead code

**Location:** `src/force/nep_efa.cu:726-772`

**Problem:**
This kernel was written but is never called. `compute_efa_pass` uses `find_properties_many_body` from the base class instead (lines 1305-1329).

**Fix:** Either:
1. Delete the kernel (recommended — reduces code size)
2. Keep with `#ifdef NEP_EFA_USE_REDUCE_KERNEL` for future benchmarking

**Severity:** Low — cosmetic, causes compiler warning.

---

### Issue #9 — `num_kpoints_max` starts at 1, guaranteed resize on first step

**Location:** `src/force/nep_efa.cu:1007-1008, 1115-1123`

**Problem:**
`efa_para.num_kpoints_max = 1` in constructor. First call to `find_k_and_G` will almost certainly produce `num_kpoints > 1`, triggering `resize` on all k-space buffers.

**Fix:** Set `num_kpoints_max` to a reasonable initial estimate (e.g., 100) or accept the one-time resize cost.

**Severity:** Low — performance, one-time cost.

---

### Issue #10 — Unused struct members

**Location:** `src/force/nep_efa.cuh`

**Members:**
- `ExpandedBox ebox` (line 145) — never used
- `efa_para.omega_min` (line 126) — set but never used
- `efa_para.num_omega` (line 127) — set but never used
- `paramb.efa_l_max` (line 97) — set but never used in MD kernels
- `paramb.efa_num_radial` (line 98) — set but never used
- `paramb.efa_num_channels` (line 99) — set but never used

**Fix:** Remove or mark as future-use.

**Severity:** Low — cosmetic.

---

### Issue #11 — `find_k_and_G` called every step even for fixed box

**Location:** `src/force/nep_efa.cu:1190`

**Problem:**
`compute_efa_pass` calls `find_k_and_G(box.cpu_h)` unconditionally every MD step, even when the box doesn't change (NVE/NVT).

**Fix:** Cache k-mesh, only recompute when box changes. Add a flag like `box_changed_since_last_kmesh`.

**Severity:** Low — performance, not correctness.

---

### Issue #12 — Missing `is_dipole` handling in EFA radial force

**Location:** `src/force/nep_efa.cu:391-482, 488-567`

**Problem:**
Original NEP radial force kernels have an `is_dipole` parameter that changes the virial formula (uses `r12_square * f` instead of `r12 * f`). EFA kernels don't have this parameter.

**Status:** EFA shouldn't need dipole mode (it's not a dipole/polarizability model), so this is likely fine. But verify that the `find_properties_many_body` call in step 9 doesn't assume dipole mode.

**Severity:** Low — likely non-issue.

---

### Issue #13 — Large-box angular force kernel stores partial forces, then uses `find_properties_many_body`

**Location:** `src/force/nep_efa.cu:573-648, 1286-1301, 1317-1329`

**Problem:**
The angular force kernel stores per-pair `f12` to `g_f12x/y/z`, then `find_properties_many_body` reduces them. Need to verify that `find_properties_many_body` expects the same `f12` sign convention and indexing as what EFA stores.

**Verification needed:**
- Check `find_properties_many_body` signature and indexing in `src/force/potential.cuh`
- Confirm `g_f12x[index]` where `index = i1 * N + n1` matches expected layout

**Severity:** Medium — if wrong, forces will be incorrect.

---

## File Inventory

### Modified files
| File | Changes |
|------|---------|
| `src/force/nep.cuh` | `Small_Box_Data` → public; `paramb`/`annmb`/`zbl`/`ebox` → public; added `is_efa` flag |
| `src/force/nep.cu` | Header parsing for `nep4_efa`/`nep4_zbl_efa`; skip EFA params + q_scaler |
| `src/force/force.cu` | Added `#include "nep_efa.cuh"` + dispatch for `nep4_efa`/`nep4_zbl_efa` |
| `src/main_nep/parameters.cuh` | EFA parameter fields + `q_scaler_efa` |
| `src/main_nep/parameters.cu` | EFA parameter calculation |
| `src/main_nep/fitness.cu` | `write_nep_txt` EFA serialization |
| `src/main_nep/snes.cu` | EFA `type_of_variable` mapping |

### New files
| File | Description |
|------|-------------|
| `src/utilities/efa_utilities.cuh` | EFA math primitives |
| `src/main_nep/nep_efa.cuh` | Training-side NEP_EFA class |
| `src/main_nep/nep_efa.cu` | Training-side NEP_EFA implementation |
| `src/force/nep_efa.cuh` | MD-side NEP_EFA class |
| `src/force/nep_efa.cu` | MD-side NEP_EFA implementation |

---

## Next Steps (Priority Order)

1. **Fix Issue #3** (virial index mapping bug) — simple one-line fix
2. **Fix Issue #1** (large-box radial virial) — decide on pattern, implement
3. **Create smoke test** (Issue #6) — generate dummy nep.txt, run 1 step
4. **Finite-difference validation** (Issue #7) — verify forces
5. **Verify `find_properties_many_body` interface** (Issue #13)
6. **Clean up dead code** (Issue #8, #10)
7. **Performance optimizations** (Issue #9, #11)
8. **End-to-end training run**

---

# EFA-TT 低秩扩展分析（基于 Nature Machine Intelligence 2026 文章第四节）

> 来源：https://mp.weixin.qq.com/s/rrzk9YvdaOHRI1uQ9PRpXA
> 文章标题：《当原子看见彼此：EFA 如何让机器学习力场拥有全局视野》
> 第四节标题："我们的探索：EFA-TT —— 用低秩叠加扩展 EFA"
> 性质：文章作者（非原论文 Frank 等人）在复现 EFA 基础上做的自主扩展工作

---

## 一、EFA-TT 核心思想

### 1.1 rank=1 → rank-K 推广

标准 EFA 的注意力核是可分离的（separable），本质上是 rank=1：

```
K(r̂) = φ_q(r̂) ⊗ φ_k(r̂)   （rank=1 外积）
```

EFA-TT 推广为 K 路叠加：

```
K(r̂) = Σ_{k=1}^{K} φ_q^{(k)} ⊗ φ_k^{(k)} · g^{(k)}(r̂)
```

- `φ_q^{(k)}`：由完整 RoPE 特征经 LayerNorm + 线性层得到，query 侧第 k 路因子
- `g^{(k)}`：可学习的方向门控（directional gating），控制每路在不同空间方向上的贡献
- 数学上等价于 Tensor Train / 低秩分解

rank=1 只能捕获"单一方向耦合"，rank-K 能同时建模多个独立的各向异性交互通道。

### 1.2 最关键设计决策：不取范数

作者强调：**不对 RoPE 后的特征做范数压缩（norm compression）**。

原因：
- ERoPE 编码的是复相位信息（旋转/方向），模长和辐角各有物理含义
- 取范数 `|z|` 会丢弃辐角 → 所有 rank 路失去方向区分性 → rank-K 退化为 rank=1
- 花了 K 倍计算量却没获得额外表达力
- 这是等变网络中的经典陷阱：为简化计算破坏等变性，得不偿失

### 1.3 复杂度保持 O(N)

通过交换求和次序（先全局聚合 key 侧因子，再查询），EFA-TT 仍为线性复杂度：

```
标准 EFA:  O(N·L)         — L 个 Lebedev 方向
EFA-TT:    O(N·K·L)       — K 和 L 为常数时仍为 O(N)
```

注意：
- 常数因子增长 K 倍
- N=39 时推理时间仅增加 ~5%（0.00082s vs 0.00078s），说明 EFA 核占比很小
- 大体系上 K 的常数开销才会显著体现

---

## 二、EFA-TT 实验结果

### 2.1 实验设置

- 数据集：N39_D10.npz（39 原子 NaCl 团簇，1500 个结构）
- rank 扫描：rank = 2, 4, 8, 16（两 seed 均值）

### 2.2 结果表格

| 模型 | energy RMSE | forces RMSE | 推理时间 |
|------|-------------|-------------|----------|
| EFA baseline | 6.696 | 1.073 | 0.00078 s |
| EFA-TT rank=2 | 6.615 | 1.071 | 0.00075 s |
| EFA-TT rank=8 | 6.605 | 1.069 | 0.00079 s |
| EFA-TT rank=16 | 6.616 | 1.068 | 0.00082 s |

### 2.3 数据解读

| 指标 | baseline → rank=8 | 相对改善 | 评价 |
|------|-------------------|----------|------|
| 能量 RMSE | 6.696 → 6.605 | -1.4% | 微弱改善 |
| 力 RMSE | 1.073 → 1.069 | -0.4% | 噪声范围内 |
| 推理时间 | 0.78ms → 0.79ms | +1.3% | 可忽略 |

- rank=8 能量 RMSE 最优（169.35 vs 171.68 meV/atom），但改善仅 1-2%
- rank=16 能量误差回升（6.616 > 6.605），暗示过拟合或优化困难
- rank=2 反而比 baseline 更快（0.00075s vs 0.00078s），可能为计时噪声

---

## 三、工程教训（踩坑记录）

### 教训链

```
教训1：取范数 → 丢相位 → rank退化 → 各rank无差别
         ↓ 纠正
教训2：保留相位 → LayerNorm → 线性映射 → 方向门控 → 坐标中心化
         ↓ 验证
教训3：rank扫描恢复分辨度，但对比仍需更严谨
```

### 详细说明

1. **早期"范数因子"版本的失败**：把 RoPE 后的复数特征压缩为范数 `|z|` 再分解 → 各 rank 精度几乎无差别。直接验证了"不能丢相位"的设计原则。

2. **正确做法**：完整 RoPE 特征（保留相位）→ LayerNorm → 带偏置线性映射 → 方向门控 → 图内坐标中心化。其中"图内坐标中心化"确保每个原子在其局部坐标系中处理相对位置，而非绝对坐标。

3. **公平对比仍在进行**：当前结果不够严谨，需补齐多 seed 统计、bias-corrected RMSE，并在 SN₂/cumulene 等长程基准上验证。

### 与 GPUMD EFA 代码的对应

- GPUMD 的 `efa_erope_value(rx, ry, rz, ω, l, m)` 用相对坐标作为输入，与"图内坐标中心化"一致
- GPUMD EFA 的功率谱 `Σ_m |A_lm|²` 是 ANN 输入（描述符层面），不违反"不取范数"原则——该原则针对的是注意力核的分解，不是描述符
- 注意力核仍是 `G(k)|S(k)|²` 的 Ewald 倒空间形式，S(k) 保留了完整相位

---

## 四、EFA-TT 未来展望

1. **相位 + 幅值双路因子**：更结构化地保留 ERoPE 的复数性质，将实部和虚部分别建模
2. **rank 自适应**：用小 rank 蒸馏 full EFA，自动在速度与精度间权衡
3. **大体系 benchmark**：39 原子太小，需在 N≫39 时验证速度优势
4. **与论文场景对齐**：长程/非局域体系（SN₂ 反应、累积烯、电荷转移）才是 EFA 主场

---

## 五、对 GPUMD EFA 实现的启示

### 5.1 低秩扩展方向

如要在 GPUMD 中实现类似 EFA-TT 的低秩扩展：
- 将当前单一 `q_n`（标量注意力权重）扩展为 K 路向量 `q_n^{(k)}`
- 每路有独立的 ANN 和描述符系数
- Ewald 倒空间求和变为 K 个独立结构因子的加权和
- PPPM 路径可自然扩展——PPPM mesh 同时处理 K 路电荷

### 5.2 优先级判断

当前标准 EFA（rank=1）已是合理基线。低秩扩展的优先级应排在基础验证（有限差分力、旋转协变性、NVE 漂移）之后。EFA-TT 在玩具体系上仅 1-2% 改善，不足以 justify 优先实现。

### 5.3 实验启示

- 小体系（<100 原子）上 EFA rank=1 已足够，不应期望低秩扩展有显著收益
- 真正的验证场景：SN₂ 反应、累积烯、电荷转移等长程/非局域体系
- rank 扫描需多 seed + 误差棒才有统计意义

---

## 六、总体评价

### 优点
1. 逻辑闭环完整：动机 → 方法 → 实验 → 踩坑 → 展望
2. 工程教训有实质价值："不取范数"对等变网络社区有参考意义
3. 诚实自省：明确承认实验局限，未过度宣传

### 不足
1. 实验不足以支撑任何结论：39 原子 NaCl 上 1-2% 改善在统计噪声范围内
2. 缺少物理论证：未解释 rank-K 能描述什么 rank=1 不能描述的物理交互
3. rank 扫描不充分：无误差棒，缺中间数据点（rank=4 未列出）
4. 未测核心场景：EFA 价值在于长程交互，但实验避开了这些场景

### 定位
探索性预实验，回答了"能不能做"（可以），没回答"值不值得做"（实验不足无法判断）。
