import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOLVERS = [
    os.path.join(ROOT, "hybrid_pde", "solvers", "numerical", "colehopf.py"),
    os.path.join(ROOT, "hybrid_pde", "solvers", "numerical", "fdm.py"),
    os.path.join(ROOT, "hybrid_pde", "solvers", "numerical", "spectral.py"),
]

for solver in SOLVERS:
    name = os.path.basename(solver)
    print(f"\n{'='*40}\nRunning {name}\n{'='*40}")
    result = subprocess.run([sys.executable, solver], cwd=ROOT)
    if result.returncode != 0:
        print(f"ERROR: {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print("\nAll datasets generated.")
