# SC-IVR / FBTS Implementation Plan

## Document status

Design plan for Method 3 of the semiclassical roadmap. Not yet
implemented. This document specifies the algorithms, data structures,
GPU kernel designs, validation strategy, and estimated effort.

## Prerequisites

- LSC-IVR (Method 1): **complete**. Provides the Wigner sampler, the
  `dump_qct` trajectory output, and the `lsc_ivr.py` post-processor.
- RPMD correlation (Method 2): **planned**. The `dump_centroid` keyword
  and `ensemble rpmd/trpmd` propagation are available; the measurement
  class is not yet tested. RPMD provides the benchmark against which
  SC-IVR results should be validated.

## Background

### Full SC-IVR (Herman–Kluk)

The semiclassical initial value representation expresses the quantum
time-correlation function as a phase-space average over classical
trajectories, each carrying a phase and weight from the van Vleck
determinant:

```
C_AB(t) = (2πℏ)^{-N} ∫ dq dp  [M_t(q,p)]^{1/2}
           × exp(i S_t(q,p)/ℏ - i θ_t)
           × A(q(0), p(0)) B(q(t), p(t))
```

where:
- `M_t` is the van Vleck determinant = det(∂q(t)/∂q(0) + ... )
- `S_t` is the classical action along the trajectory
- `θ_t` is the Maslov phase (sign-tracking of the determinant)
- `N = number_of_atoms` (3N phase-space dimensions)

The key new objects are:
1. The **monodromy (stability) matrix** `M_t`, a `6N × 6N` Jacobian
   that maps initial phase-space perturbations to time-evolved ones.
2. The **classical action** `S_t = ∫_0^t L dt`, accumulated along each
   trajectory.
3. The **Maslov phase** `θ_t`, incremented each time `M_t` changes sign.

### FBTS (Forward–Backward Trajectory Solution)

FBTS pairs a forward trajectory of length `t/2` with a backward
trajectory of length `t/2`, forming the correlation `A(0) B(t/2)
A(t) B(t/2)`. This causes the rapidly oscillating SC-IVR phase to
partially cancel, reducing the sign problem and the number of
trajectories needed for convergence.

In FBTS, the stability matrix only needs to be propagated for the
shorter `t/2` legs, and the combined phase is:

```
ΔS_FB = S_forward - S_backward
```

which is smaller in magnitude than the full `S_t`, improving
convergence.

### Relation to LSC-IVR

LSC-IVR is the **fully linearized** limit where:
- The monodromy matrix is approximated as identity (no stability
  tracking).
- All phases are discarded (real, positive weights only).
- The Wigner distribution replaces the exact quantum density.

SC-IVR/FBTS sit between LSC-IVR and exact quantum dynamics, retaining
partial phase information:

```
LSC-IVR  <  FBTS  <  full SC-IVR
(no phase)   (partial)   (full phase)
```

## Implementation scope

### Phase A: Monodromy matrix infrastructure

The dominant new cost. For a system of `N` atoms, the monodromy matrix
is `6N × 6N`. For each trajectory, we must propagate the matrix elements
alongside the classical trajectory.

#### Data layout

The monodromy matrix has block structure:

```
M = [ ∂q(t)/∂q(0)   ∂q(t)/∂p(0) ]   = [ M_qq  M_qp ]
    [ ∂p(t)/∂q(0)   ∂p(t)/∂p(0) ]     [ M_pq  M_pp ]
```

Each block is `3N × 3N`. For N=2 (OH radical): `6×6` per block,
`12×12` total. For N=10: `60×60`, still tractable. For N=100:
`600×600 = 360K doubles per trajectory` — challenging but feasible
on GPU for a small number of replicas.

**Storage decision**: store `M` as a `GPU_Vector<double>` of size
`36N²` per replica (4 blocks × `(3N)²` elements), laid out as
row-major `3N × 3N` matrices.

#### Propagation

The monodromy matrix satisfies the tangent dynamics equation (TDE):

```
dM/dt = J · ∇²H(q(t)) · M
```

where `J` is the symplectic metric `[[0, I], [-I, 0]]` and `∇²H` is the
Hessian of the Hamiltonian (the force Hessian) evaluated along the
trajectory.

**Key question**: how to obtain the force Hessian `∂²V/∂q_i ∂q_j`
along a trajectory using an NEP potential?

**Options**:

1. **Finite-difference Hessian** (simplest, reuses existing
   `Molecular_Hessian`):
   - At each dump step, compute the Hessian by finite displacement.
   - Cost: `O(N²)` force evaluations per step → expensive.
   - Suitable only for very small systems (N ≤ 10) and short
     trajectories.

2. **Analytic NEP Hessian** (ideal, but not available):
   - NEP's neural network structure permits backpropagation of the
     force Jacobian, but this is a major implementation effort.
   - Could be a separate project; not required for the initial SC-IVR
     implementation.

3. **Semi-analytic approach** (intermediate):
   - Propagate auxiliary "tangent vectors" (columns of `M`) alongside
     the trajectory using finite-difference Jacobian-vector products
     (JVPs).
   - Each tangent vector requires one extra force evaluation per step.
   - Total extra force evaluations per step: `O(N)` (one per column of
     `M`), giving `O(N²)` total cost — same as storing `M` itself.
   - This is the standard approach used in SC-IVR implementations.

**Recommended**: Option 3 (semi-analytic JVP propagation).

#### GPU kernel design for tangent propagation

For each column `k` of the monodromy matrix, we propagate:

```
δq_k(t + dt) = δq_k(t) + δp_k(t)/m * dt
δp_k(t + dt) = δp_k(t) + (∂F/∂q · δq_k) * dt
```

The Jacobian-vector product `∂F/∂q · δq` can be computed by:
- Finite difference: `F(q + ε δq) - F(q)` / `ε` for each column.
- Or: if the NEP compute kernel is modified to also return the
  Jacobian, analytic computation is possible.

**Finite-difference JVP** (per step, per column):
1. Save current positions `q`.
2. Perturb: `q' = q + ε δq_k`.
3. Compute force `F'` at perturbed position.
4. Restore `q`, recompute `F` (or reuse from main trajectory).
5. `JVP = (F' - F) / ε`.

**Cost**: For 3N columns and one force evaluation each, this is `O(N)`
extra force evaluations per step. For N=2 (OH), this is 6 extra force
evaluations per step — negligible overhead. For N=10, ~30 extra —
still cheap on GPU. For N=100, ~300 extra — expensive.

**GPU parallelism**: Each column's JVP is independent, so multiple
columns can be evaluated in parallel by batching the perturbed
positions. The batch QCT framework already supports multi-replica
position layouts; the tangent vectors can be treated as additional
"virtual replicas".

### Phase B: Action and phase accumulation

The classical action `S_t` is accumulated along each trajectory:

```
S_t = ∫_0^t [T(p(τ)) - V(q(τ))] dτ
```

This requires only the kinetic and potential energy per step, both
already available from `dump_qct` thermodynamic output. The action
is a scalar per replica, accumulated in `compute2` alongside the
velocity-Verlet step.

The Maslov phase `θ_t` is incremented each time the determinant of
`M_qq + M_pp + i(M_qp - M_pq)` crosses the negative real axis. The
tracking requires computing the sign of `det(M_qq)` (or the full
complex determinant) at each step.

**Determinant computation**: For small matrices (N ≤ 10), use
`cuSOLVER` (already linked in GPUMD's Makefile). For larger matrices,
use a LU decomposition on GPU.

### Phase C: SC-IVR weight and correlation

The SC-IVR weight per trajectory is:

```
w_i = [det(M_t)]^{1/2} × cos(S_t/ℏ - θ_t)
```

The correlation function is then:

```
C_AB(t) = Σ_i w_i × A(0)_i × B(t)_i  /  Σ_i w_i
```

This is the same structure as LSC-IVR's `lsc_ivr.py`, except:
- The weight is complex (or oscillating real).
- The denominator may be near zero (sign problem).
- More trajectories are needed for convergence.

**Post-processing**: Extend `lsc_ivr.py` (or create `sc_ivr.py`)
to accept complex weights and compute the correlation. The trajectory
format from `dump_qct` already contains all needed per-step data; the
monodromy matrix and action need to be written to a supplementary
output file.

### Phase D: FBTS variant

FBTS modifies the correlation to:

```
C_AB(t) ≈ (2πℏ)^{-N} ∫ dq dp  [M_{t/2}]^{1/2}
           × exp(i ΔS_FB/ℏ - i θ_{t/2})
           × A(q(0)) B(q(t/2)) A(q(t)) B(q(t/2))
```

Implementation differences from full SC-IVR:
1. Shorter propagation legs (t/2 instead of t).
2. Forward-backward pairing of trajectories.
3. Partial phase cancellation → better convergence.
4. The monodromy matrix is propagated only for t/2.

**FBTS weight**:

```
w_i^{FB} = [det(M_{t/2}^{fwd} · M_{t/2}^{bwd})]^{1/2}
            × cos(ΔS_FB/ℏ - θ_{t/2}^{combined})
```

The backward propagation is a time-reversed trajectory: start from
the endpoint of the forward leg, reverse momenta, propagate for t/2,
then reverse momenta again.

## Detailed implementation steps

### Step 1: Tangent vector data structures

Add to `Ensemble_QCT`:

```cpp
// Per-replica monodromy matrix: 4 blocks of 3N × 3N
GPU_Vector<double> monodromy_qq;  // ∂q(t)/∂q(0)
GPU_Vector<double> monodromy_qp;  // ∂q(t)/∂p(0)
GPU_Vector<double> monodromy_pq;  // ∂p(t)/∂q(0)
GPU_Vector<double> monodromy_pp;  // ∂p(t)/∂p(0)

// Per-replica action accumulator
GPU_Vector<double> action;  // S_t per replica

// Per-replica Maslov phase counter
GPU_Vector<int> maslov_count;
```

Initialize `M = I` at t=0:
- `M_qq = I`, `M_pp = I`
- `M_qp = 0`, `M_pq = 0`
- `action = 0`
- `maslov_count = 0`

### Step 2: Tangent propagation kernel

New GPU kernel `propagate_monodromy`:

```cpp
// For each atom i and each column k:
// δq_i^k(t+dt) = δq_i^k(t) + δp_i^k(t)/m_i * dt
// δp_i^k(t+dt) = δp_i^k(t) + (JVP_k)_i * dt
//
// where JVP_k = ∂F/∂q · δq_k  (Jacobian-vector product)
```

The JVP is computed by finite difference using the existing NEP force
evaluator. For a system with `R` replicas and `3N` columns:

1. Create `3N` perturbed position arrays: `q_k' = q + ε δq_k`.
2. Batch-evaluate forces at all perturbed positions (reuse the
   batch NEP framework with `3N * R` virtual replicas).
3. `JVP_k = (F_k' - F) / ε`.

**Memory**: `3N × R × 3N` doubles for perturbed positions.
For N=2, R=32: `6 × 32 × 6 = 1152` doubles — trivial.
For N=10, R=32: `30 × 32 × 30 = 28800` doubles — still fine.

### Step 3: Action accumulation

In `compute2`, after the velocity-Verlet half-step:

```cpp
// Kinetic energy: T = Σ ½ m v²
// Potential energy: V = Σ V_i (already computed)
// action += (T - V) * dt
```

This is a simple reduction; can reuse existing `find_thermo` logic.

### Step 4: Maslov phase tracking

At each step, compute `det(M_qq + M_pp)`:
- For small matrices: `cuSOLVER` LU decomposition.
- Track sign changes.

### Step 5: Output

New keyword: `dump_sc_ivr` (or extend `dump_qct` with an `sc_ivr` flag).

Output files:
- `qct_monodromy.bin`: binary, per-replica, per-dump-step monodromy
  matrix elements. Binary format for efficiency (text would be too
  large for N > 10).
- `qct_action.csv`: `replica, step, time_fs, action_eV_fs, maslov_count`
- The trajectory `qct_trajectory.xyz` remains unchanged.

### Step 6: Post-processing tool

New `tools/qct/sc_ivr.py`:

```python
# Read trajectory + monodromy + action
# Compute complex weight per replica:
#   w = sqrt(det(M)) * exp(i*S/ℏ - i*θ)
# Compute correlation:
#   C_AB(t) = Σ w_i A(0)_i B(t)_i / Σ w_i
# Note: both numerator and denominator are complex.
# Report |C_AB(t)|² or Re[C_AB(t)] depending on the observable.
```

### Step 7: FBTS extension

For FBTS:
1. Run forward trajectory for t/2.
2. Save midpoint state (q, p).
3. Run backward trajectory for t/2 (reverse p, propagate, reverse p).
4. Compute combined monodromy and action.
5. Post-process with the FBTS weight formula.

**Implementation**: The forward-backward loop can be implemented
either:
- (a) As a two-pass GPUMD run with a restart file at the midpoint.
- (b) As a single run with a modified ensemble that reverses momenta
  at t/2.

Option (a) is simpler and reuses existing infrastructure. Option (b)
is more efficient but requires modifying the ensemble.

**Recommended**: Start with option (a) for validation, optimize to
option (b) later if needed.

## Memory and performance estimates

| System | N | Matrix size | Memory/replica | Extra force evals/step |
|--------|---|-------------|-----------------|----------------------|
| OH radical | 2 | 12×12 | 1.2 KB | 6 |
| H₂O | 3 | 18×18 | 2.6 KB | 9 |
| Ethanol | 9 | 54×54 | 23 KB | 27 |
| Small peptide | 50 | 300×300 | 720 KB | 150 |

For OH (N=2) with 32 replicas: total extra force evaluations per step
= 6 × 32 = 192, but each is for only 2 atoms — negligible overhead.
The monodromy matrix operations (12×12) are also trivial.

For ethanol (N=9) with 32 replicas: 27 × 32 = 864 extra force evals
per step for 9 atoms each. The batch NEP can handle this, but the
`3N × R = 27 × 32 = 864` virtual replicas will hit the NEP neighbor
capacity limit. May need to split into batches.

## Validation strategy

1. **Harmonic oscillator**: analytic SC-IVR result is known.
   Compare LSC-IVR, SC-IVR, and exact quantum for a 1D harmonic
   oscillator at various temperatures.

2. **OH radical (NEP89)**: compare SC-IVR spectral density against
   LSC-IVR (already validated at 115 THz) and RPMD (once Method 2
   is implemented). SC-IVR should show sharper peaks and possibly
   additional quantum features.

3. **FBTS convergence**: verify that FBTS converges with fewer
   trajectories than full SC-IVR for the same system.

4. **Cross-check**: at high temperature, SC-IVR should reduce to
   LSC-IVR (classical limit). Verify this by increasing T and
   checking that the phase oscillations vanish.

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sign problem: oscillating weights require many trajectories | High | High | Start with FBTS (partial cancellation); use filtered/ windowed averaging |
| Hessian computation too slow for large N | Medium | High | Start with N ≤ 10; use finite-difference JVP; document the N scaling |
| cuSOLVER determinant for small matrices has overhead | Low | Low | For N ≤ 10, compute determinant on CPU |
| NEP batch capacity insufficient for virtual replicas | Medium | Medium | Split tangent vectors into sub-batches |
| Phase tracking bugs (Maslov index) | Medium | High | Validate against analytic harmonic oscillator first |

## Estimated effort

| Component | Effort | Description |
|-----------|--------|-------------|
| Tangent vector data structures + initialization | Small | Extend `Sampled_Point` and `Ensemble_QCT` |
| JVP computation via batch NEP | Medium | Reuse batch framework; handle perturbation layout |
| Monodromy propagation kernel | Medium | GPU kernel for tangent update; symplectic integrator |
| Action accumulation | Small | Scalar reduction per step |
| Maslov phase tracking | Medium | cuSOLVER determinant; sign tracking |
| Output (binary monodromy + action CSV) | Small | New dump keyword or extend dump_qct |
| Post-processing tool (`sc_ivr.py`) | Medium | Complex weights; convergence diagnostics |
| FBTS two-pass workflow | Small | Restart-based; Python orchestration |
| Validation (harmonic + OH) | Medium | Analytic checks; comparison with LSC-IVR/RPMD |
| **Total** | **~2-3 weeks** | For a single developer with GPU access |

## Decision criteria for proceeding

Before starting SC-IVR implementation, the following should be true:

1. **LSC-IVR validated** on the target system (✅ done for OH).
2. **RPMD correlation available** (Method 2, planned) — provides the
   reference benchmark.
3. **A demonstrated need**: LSC-IVR or RPMD results are insufficient
   for the target application (e.g., missing quantum features, poor
   frequency accuracy, wrong tunneling rates).
4. **Small system size** (N ≤ 10) for the initial implementation.

If all four conditions are met, proceed with FBTS first (lower risk,
better convergence), then full SC-IVR if needed.
