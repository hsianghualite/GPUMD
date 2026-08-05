#!/bin/bash
#SBATCH --job-name=efa-fd2
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:30:00

set -eo pipefail

source /etc/profile.d/lmod.sh
module load gpumd/4.8-cuda12.4

export OMP_NUM_THREADS=1
export TMPDIR=$HOME/gpumd-efa-test/tmp
mkdir -p "${TMPDIR}"
GPUMD=~/gpumd-efa-test/src/gpumd
TESTDIR=~/gpumd-efa-test/fd_check2
mkdir -p ${TESTDIR}
cd ${TESTDIR}

# Reuse the same nep.txt (random parameters)
cp ~/gpumd-efa-test/smoke_test/nep.txt .

read_pe() {
  awk '!/^#/ && NF { pe = $3 } END { if (pe == "") exit 1; print pe }' thermo.out
}

# Create a strongly perturbed, asymmetric 8-atom configuration
python3 <<'PYEOF'
import random
random.seed(123)
a = 5.43
# Base positions (diamond cubic)
base = [
    (0, 0, 0),
    (a/4, a/4, 0),
    (a/2, a/2, 0),
    (3*a/4, 3*a/4, 0),
    (0, a/2, a/2),
    (a/4, 3*a/4, a/2),
    (a/2, 0, a/2),
    (3*a/4, a/4, a/2),
]
# Add LARGE perturbations (0.3 A) to break symmetry strongly
perturbed = [(x + random.uniform(-0.3, 0.3),
              y + random.uniform(-0.3, 0.3),
              z + random.uniform(-0.3, 0.3)) for x,y,z in base]

with open('model.xyz', 'w') as f:
    f.write(f"8\n")
    f.write(f"pbc=\"T T T\" Lattice=\"{a} 0 0 0 {a} 0 0 0 {a}\" Properties=species:S:1:pos:R:3:forces:R:3\n")
    for x,y,z in perturbed:
        f.write(f"Si {x:.6f} {y:.6f} {z:.6f} 0.0 0.0 0.0\n")

print("model.xyz written with strong perturbations (0.3 A)")
PYEOF

cat > run.in <<'EOF'
potential   nep.txt
velocity    1
ensemble nve
time_step 0
dump_force 1
dump_thermo 1
run 1
EOF

echo "=== Run 1: strongly perturbed configuration ==="
rm -f thermo.out force.out neighbor.out
${GPUMD} 2>&1 | tail -20
echo ""
echo "=== force.out ==="
cat force.out
echo ""
echo "=== thermo.out ==="
cat thermo.out

# Data columns are T KE PE ..., so PE is column 3. Keep the original force
# before replacing the structure for the +/- displacement runs.
ORIG_PE=$(read_pe)
cp force.out force_orig.out
echo "Original PE = ${ORIG_PE}"

# Finite difference on atom 0, x direction
DX=0.001

python3 <<PYEOF
with open('model.xyz') as f:
    lines = f.readlines()
parts = lines[2].split()
x = float(parts[1]) + ${DX}
lines[2] = f"Si {x:.6f} {parts[2]} {parts[3]} 0.0 0.0 0.0\n"
with open('model.xyz_plus', 'w') as f:
    f.writelines(lines)

with open('model.xyz') as f:
    lines = f.readlines()
parts = lines[2].split()
x = float(parts[1]) - ${DX}
lines[2] = f"Si {x:.6f} {parts[2]} {parts[3]} 0.0 0.0 0.0\n"
with open('model.xyz_minus', 'w') as f:
    f.writelines(lines)
PYEOF

echo ""
echo "=== Run +dx ==="
cp model.xyz_plus model.xyz
rm -f thermo.out force.out neighbor.out
${GPUMD} 2>&1 | tail -5
PE_PLUS=$(read_pe)
echo "PE (+dx) = ${PE_PLUS}"

echo ""
echo "=== Run -dx ==="
cp model.xyz_minus model.xyz
rm -f thermo.out force.out neighbor.out
${GPUMD} 2>&1 | tail -5
PE_MINUS=$(read_pe)
echo "PE (-dx) = ${PE_MINUS}"

# Restore original
cp model.xyz_plus model.xyz.bak 2>/dev/null || true
# Actually restore original (re-generate it)
python3 <<'PYEOF'
import random
random.seed(123)
a = 5.43
base = [
    (0, 0, 0),
    (a/4, a/4, 0),
    (a/2, a/2, 0),
    (3*a/4, 3*a/4, 0),
    (0, a/2, a/2),
    (a/4, 3*a/4, a/2),
    (a/2, 0, a/2),
    (3*a/4, a/4, a/2),
]
perturbed = [(x + random.uniform(-0.3, 0.3),
              y + random.uniform(-0.3, 0.3),
              z + random.uniform(-0.3, 0.3)) for x,y,z in base]

with open('model.xyz', 'w') as f:
    f.write(f"8\n")
    f.write(f"pbc=\"T T T\" Lattice=\"{a} 0 0 0 {a} 0 0 0 {a}\" Properties=species:S:1:pos:R:3:forces:R:3\n")
    for x,y,z in perturbed:
        f.write(f"Si {x:.6f} {y:.6f} {z:.6f} 0.0 0.0 0.0\n")
PYEOF

# Get analytic force on atom 0 from the original configuration.
F_ANA=$(awk 'NR==1{print $1; exit}' force_orig.out)

# Compute numerical force: F_x = -(E(+dx) - E(-dx)) / (2*dx)
python3 <<PYEOF
pe_plus = float("${PE_PLUS}")
pe_minus = float("${PE_MINUS}")
pe_orig = float("${ORIG_PE}")
dx = float("${DX}")
F_num = -(pe_plus - pe_minus) / (2 * dx)
F_ana = float("${F_ANA}")
print(f"")
print(f"=== Finite Difference vs Analytic Force (atom 0, x) ===")
print(f"  PE(-dx) = {pe_minus:.10e}")
print(f"  PE(0)   = {pe_orig:.10e}")
print(f"  PE(+dx) = {pe_plus:.10e}")
print(f"  dx      = {dx:.6e} A")
print(f"  F_numerical  = {F_num:.10e} eV/A")
print(f"  F_analytic   = {F_ana:.10e} eV/A")
if abs(F_num) > 1e-12:
    print(f"  ratio (ana/num) = {F_ana/F_num:.6f}")
print(f"  abs diff        = {abs(F_ana - F_num):.10e} eV/A")
PYEOF

echo ""
echo "=== FD check complete ==="
