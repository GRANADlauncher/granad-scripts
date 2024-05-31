# slurm benchmark

job_script.sh : runs slurm_benchmark.py, adjust the SLURM commands to your needs

slurm_benchmark.py : benchmarks granad

Benchmarks performed:

1. Cutting (cutting finite flake from "infinite" material), should scale linearly
2. Diagonalization (sets up hamiltonian, coulomb, energies, density matrix, ...) should scale cubically.
3. Default Time propagation should scale cubically.
4. Optimized Time propagation (mostly optimized for memory, avoids some quadratic ops) should scale with better prefactor. 

