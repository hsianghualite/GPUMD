#!/usr/bin/env bash
# Batch classical MD, QCT, LSC-IVR, QTB, and RPMD Si kappa tests.
# QCT/LSC-IVR use an automatic 3N x 3N Hessian and require explicit opt-in.

set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
METHODS="${METHODS:-all}"
OUT_ROOT="${OUTPUT_ROOT:-$ROOT/tests/gpumd/si_nep4_12x12x12}"
GPUMD="${GPUMD_BIN:-$ROOT/src/gpumd}"
POTENTIAL="${POTENTIAL_SRC:-$ROOT/potentials/nep/Si_2022_NEP4_3body.txt}"
MODEL="${MODEL_SRC:-$ROOT/tests/gpumd/qct_nep89_si/classical_kappa_12x12x12_nep4_3body/model.xyz}"
TEMP="${TEMPERATURE:-300}"
SEED="${SEED:-12345}"
BEADS="${BEADS:-8}"
DT="${TIME_STEP:-1}"
EQUIL="${EQUIL_STEPS:-500000}"
PRODUCTION="${PRODUCTION_STEPS:-10000000}"
HAC_INTERVAL="${HAC_INTERVAL:-20}"
HAC_POINTS="${HAC_POINTS:-50000}"
HAC_OUTPUT="${HAC_OUTPUT_INTERVAL:-10}"
THERMO_INTERVAL="${THERMO_INTERVAL:-1000}"
HESSIAN_DISPLACEMENT="${HESSIAN_DISPLACEMENT:-0.001}"
FORCE="${FORCE:-0}"
DRY_RUN=0
ALLOW_LARGE_HESSIAN="${ALLOW_LARGE_HESSIAN:-0}"

if [[ ! -f "$MODEL" && -f "/home/cuser/gpumd-qct-review/tests/gpumd/qct_nep89_si/classical_kappa_12x12x12_nep4_3body/model.xyz" ]]; then
  MODEL="/home/cuser/gpumd-qct-review/tests/gpumd/qct_nep89_si/classical_kappa_12x12x12_nep4_3body/model.xyz"
elif [[ ! -f "$MODEL" ]]; then
  MODEL="$ROOT/tests/gpumd/qct_nep89_si/classical_kappa_12x12x12/model.xyz"
fi

ARG=""
if (($# > 0)); then ARG="$1"; fi
case "$ARG" in
  --dry-run) DRY_RUN=1 ;;
  --allow-large-hessian) ALLOW_LARGE_HESSIAN=1 ;;
  --force) FORCE=1 ;;
  --help|-h)
    cat <<'EOF'
Use environment variables:
  METHODS=classical,qtb,rpmd ./tools/qct/run_si_nep4_12x12x12.sh
  METHODS=qct,lsc_ivr ALLOW_LARGE_HESSIAN=1 ./tools/qct/run_si_nep4_12x12x12.sh
  ./tools/qct/run_si_nep4_12x12x12.sh --dry-run

For Slurm:
  sbatch --partition=16V100 --gpus-per-node=1 --time=48:00:00 \
    tools/qct/run_si_nep4_12x12x12.sh
EOF
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $ARG" >&2; exit 2 ;;
esac

if [[ "$METHODS" == all ]]; then METHODS=classical,qct,lsc_ivr,qtb,rpmd; fi
[[ -x "$GPUMD" ]] || { echo "GPUMD not found: $GPUMD" >&2; exit 1; }
[[ -f "$POTENTIAL" ]] || { echo "Potential not found: $POTENTIAL" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "Model not found: $MODEL" >&2; exit 1; }
NATOMS="$(awk 'NR == 1 {print $1; exit}' "$MODEL")"
[[ "$NATOMS" == 13824 ]] || { echo "Expected 13824 atoms, got $NATOMS" >&2; exit 1; }

has_method() {
  [[ ",$METHODS," == *",$1,"* ]]
}

write_run_in() {
  local method="$1"
  local dir="$2"
  case "$method" in
    classical)
      printf '%s\n' \
        "potential    $POTENTIAL" "time_step    $DT" "velocity     $TEMP" \
        "ensemble     npt_ber $TEMP $TEMP 100 0 53.4059 2000" \
        "dump_thermo  $THERMO_INTERVAL" "run          $EQUIL" \
        "ensemble     nve" "compute_hac  $HAC_INTERVAL $HAC_POINTS $HAC_OUTPUT" \
        "dump_thermo  $THERMO_INTERVAL" "run          $PRODUCTION" > "$dir/run.in"
      ;;
    qct)
      printf '%s\n' \
        "potential    $POTENTIAL" "time_step    $DT" \
        "ensemble     qct canonical temperature $TEMP seed $SEED replicas 1 hessian_displacement $HESSIAN_DISPLACEMENT zpe yes phase random" \
        "compute_hac  $HAC_INTERVAL $HAC_POINTS $HAC_OUTPUT" \
        "dump_thermo  $THERMO_INTERVAL" "run          $PRODUCTION" > "$dir/run.in"
      ;;
    lsc_ivr)
      printf '%s\n' \
        "potential    $POTENTIAL" "time_step    $DT" \
        "ensemble     lsc_ivr $TEMP seed $SEED replicas 1 hessian_displacement $HESSIAN_DISPLACEMENT anharmonic_reweighting no" \
        "compute_hac  $HAC_INTERVAL $HAC_POINTS $HAC_OUTPUT" \
        "dump_thermo  $THERMO_INTERVAL" "run          $PRODUCTION" > "$dir/run.in"
      ;;
    qtb)
      printf '%s\n' \
        "potential    $POTENTIAL" "time_step    $DT" "velocity     $TEMP" \
        "ensemble     nvt_qtb $TEMP $TEMP 100 f_max 200 N_f 100" \
        "dump_thermo  $THERMO_INTERVAL" "run          $EQUIL" \
        "ensemble     nve" "compute_hac  $HAC_INTERVAL $HAC_POINTS $HAC_OUTPUT" \
        "dump_thermo  $THERMO_INTERVAL" "run          $PRODUCTION" > "$dir/run.in"
      ;;
    rpmd)
      printf '%s\n' \
        "potential    $POTENTIAL" "time_step    $DT" "velocity     $TEMP" \
        "ensemble     pimd $BEADS $TEMP $TEMP 100" \
        "dump_thermo  $THERMO_INTERVAL" "run          $EQUIL" \
        "ensemble     rpmd $BEADS" "compute_hac  $HAC_INTERVAL $HAC_POINTS $HAC_OUTPUT" \
        "dump_thermo  $THERMO_INTERVAL" "run          $PRODUCTION" > "$dir/run.in"
      ;;
    *) echo "Unknown method: $method" >&2; return 2 ;;
  esac
}

analyze() {
  local dir="$1"
  [[ -f "$dir/hac.out" ]] || { echo "WARNING: hac.out missing in $dir"; return 1; }
  python3 - "$dir/hac.out" <<'PY'
import sys
import numpy as np
d = np.loadtxt(sys.argv[1])
if d.ndim == 1:
    d = d.reshape(1, -1)
if d.shape[1] != 11:
    raise SystemExit(f"unexpected HAC columns: {d.shape[1]}")
kx = d[:, 6] + d[:, 7]
ky = d[:, 8] + d[:, 9]
kz = d[:, 10]
ka = (kx + ky + kz) / 3.0
n = max(1, len(ka) // 4)
q = ka[-n:]
print(f"HAC rows={len(d)} time_ps={d[0,0]:.4f}-{d[-1,0]:.4f}")
print(f"kappa_end={ka[-1]:.6f} W/m/K")
print(f"kappa_last_quarter={q.mean():.6f} +/- {q.std():.6f} W/m/K")
print(f"components_last_quarter={kx[-n:].mean():.6f},{ky[-n:].mean():.6f},{kz[-n:].mean():.6f} W/m/K")
PY
}

run_method() {
  local method="$1"
  local dir="$OUT_ROOT/$method"
  echo
  echo "============================================================"
  echo "METHOD: $method"
  echo "OUTPUT: $dir"
  echo "============================================================"
  if [[ ("$method" == qct || "$method" == lsc_ivr) && "$ALLOW_LARGE_HESSIAN" -eq 0 ]]; then
    echo "SKIP: $method requires ALLOW_LARGE_HESSIAN=1 for 13824 atoms."
    return 2
  fi
  mkdir -p "$dir"
  if [[ "$FORCE" -eq 0 && -f "$dir/.completed" ]]; then
    echo "SKIP: already completed; use --force to rerun."
    return 0
  fi
  if [[ "$FORCE" -eq 0 && -f "$dir/gpumd.log" ]]; then
    echo "FAIL: $dir/gpumd.log exists; use FORCE=1 to replace it." >&2
    return 1
  fi
  if [[ "$FORCE" -eq 1 ]]; then
    rm -f "$dir"/{gpumd.log,thermo.out,hac.out,neighbor.out,stress.out}
    rm -f "$dir"/{qct_initial.out,qct_initial.xyz,qct_initial_summary.csv}
    rm -f "$dir"/{qct_hessian.out,qct_eigenvector.out,qct_stationary.xyz}
    rm -f "$dir"/{qct_thermo.csv,qct_trajectory.xyz,.completed}
  fi
  cp "$MODEL" "$dir/model.xyz"
  write_run_in "$method" "$dir"
  echo "Atoms=$NATOMS dt=$DT fs production_steps=$PRODUCTION"
  echo "Potential=$POTENTIAL"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    sed -n '1,120p' "$dir/run.in"
    return 0
  fi
  if ! (
    cd "$dir" &&
    env -u CUDA_VISIBLE_DEVICES \
      -u NVIDIA_VISIBLE_DEVICES \
      LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" \
      "$GPUMD" < run.in > gpumd.log 2>&1
  ); then
    echo "FAIL: $method; see $dir/gpumd.log" >&2
    tail -80 "$dir/gpumd.log" >&2 || true
    return 1
  fi
  analyze "$dir"
  date -Is > "$dir/.completed"
  echo "PASS: $method"
}

FAILED=""
SKIPPED=""
run_selected() {
  local method="$1"
  local rc
  has_method "$method" || return 0
  if run_method "$method"; then
    return 0
  else
    rc="$?"
  fi
  if [[ "$rc" -eq 2 ]]; then SKIPPED="$SKIPPED $method"; else FAILED="$FAILED $method"; fi
}

echo "Si NEP4 three-body 12x12x12 batch"
echo "Methods=$METHODS"
echo "Model=$MODEL"
echo "Potential=$POTENTIAL"
echo "Output=$OUT_ROOT"
if has_method qct || has_method lsc_ivr; then
  echo "WARNING: QCT/LSC-IVR automatic Hessian opt-in=$ALLOW_LARGE_HESSIAN"
fi

run_selected classical
run_selected qct
run_selected lsc_ivr
run_selected qtb
run_selected rpmd

echo
echo "==================== SUMMARY ===================="
if [[ -z "$FAILED" ]]; then echo "Failed: none"; else echo "Failed:$FAILED"; fi
if [[ -z "$SKIPPED" ]]; then echo "Skipped: none"; else echo "Skipped:$SKIPPED"; fi
if [[ -n "$FAILED$SKIPPED" ]]; then exit 1; fi
