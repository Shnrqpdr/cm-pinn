"""
Generate dataset for inverse Störmer PINN — equatorial case (Z=0).

Compiles and runs the C solver sv_equatorial.c, converts Cartesian output
to cylindrical coordinates (rho, phi), subsamples, and saves .npz dataset.

Usage:
  python generate_dataset_equatorial.py              # Proton 1 (default)
  python generate_dataset_equatorial.py --proton 2   # Proton 2
"""

import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np

# Physical constants (must match sv_equatorial.c)
M = 1.6726219e-27       # proton mass [kg]
ALPHA1_TRUE = 3.037e3    # magnetic coupling [s^-1]
DT_SOLVER = 0.0001       # solver timestep [s]

# Initial conditions (Figure 5 of report)
PROTONS = {
    1: {
        "rho0": 3.0, "drho0": 10.0, "phi0": 0.0, "dphi0": 10.0,
        "description": "Contained orbit",
    },
    2: {
        "rho0": 3.0, "drho0": 80.0, "phi0": 3 * np.pi / 2, "dphi0": 10.0,
        "description": "High radial velocity",
    },
    3: {
        "rho0": 3.0, "drho0": 100.0, "phi0": np.pi, "dphi0": 10.0,
        "description": "Very high radial velocity",
    },
}


def latin_hypercube_sampling(n_samples, t_min, t_max, seed=42):
    """Generate LHS points in [t_min, t_max]."""
    rng = np.random.default_rng(seed)
    intervals = np.linspace(t_min, t_max, n_samples + 1)
    points = np.array([rng.uniform(intervals[i], intervals[i + 1])
                       for i in range(n_samples)])
    rng.shuffle(points)
    return points


def compile_solver():
    """Compile sv_equatorial.c and return path to executable."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    solver_src = os.path.join(
        script_dir, "..", "..", "simulation",
        "no_constraint_case", "equatorial_case", "sv_equatorial.c"
    )
    solver_src = os.path.normpath(solver_src)

    if not os.path.exists(solver_src):
        print(f"ERROR: Solver source not found at {solver_src}")
        sys.exit(1)

    exe_path = os.path.join(script_dir, "data", "sv_equatorial")
    print(f"  Compiling {solver_src} ...")
    subprocess.run(
        ["gcc", "-O2", "-o", exe_path, solver_src, "-lm"],
        check=True,
    )
    print(f"  Compiled to {exe_path}")
    return exe_path


def run_solver(exe_path, T_final, rho0, drho0, phi0, dphi0):
    """Run the equatorial solver and return parsed (n, x, y) arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        particle_file = os.path.join(tmpdir, "particle.dat")
        potential_file = os.path.join(tmpdir, "potential.dat")

        cmd = [
            exe_path,
            str(T_final), str(rho0), str(drho0),
            str(phi0), str(dphi0),
            particle_file, potential_file,
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Parse particle output: "n x y z" (header + data)
        data = np.loadtxt(particle_file, skiprows=1)
        n_idx = data[:, 0].astype(int)
        x = data[:, 1]
        y = data[:, 2]

    return n_idx, x, y


def generate(proton_id=1, T_final=2.0, n_obs=1000, n_coll=3000):
    """Generate equatorial dataset for the given proton."""
    proton = PROTONS[proton_id]
    rho0 = proton["rho0"]
    drho0 = proton["drho0"]
    phi0 = proton["phi0"]
    dphi0 = proton["dphi0"]

    print(f"\n  Proton {proton_id}: {proton['description']}")
    print(f"  rho0={rho0}, drho0={drho0}, phi0={phi0:.4f}, dphi0={dphi0}")
    print(f"  T_final={T_final} s, dt={DT_SOLVER}")

    # Compile and run solver
    exe_path = compile_solver()
    n_idx, x, y = run_solver(exe_path, T_final, rho0, drho0, phi0, dphi0)

    # Convert to cylindrical coordinates
    rho = np.sqrt(x**2 + y**2)
    phi = np.unwrap(np.arctan2(y, x))
    t = n_idx * DT_SOLVER

    n_total = len(t)
    print(f"  Solver output: {n_total} points, t=[{t[0]:.4f}, {t[-1]:.4f}]")

    # Validate: check energy conservation
    c20 = dphi0 * M * rho0**2 + M * ALPHA1_TRUE / rho0
    drho_num = np.gradient(rho, t)
    V_eff = (c20**2 / (2 * M**2 * rho**2)
             - ALPHA1_TRUE * c20 / (M * rho**3)
             + ALPHA1_TRUE**2 / (2 * rho**4))
    E = 0.5 * drho_num**2 + V_eff
    E_drift = abs(E[-1] - E[0]) / abs(E[0])
    print(f"  Energy drift: {E_drift:.2e} (should be small)")

    # Subsample for observations (equally spaced indices)
    obs_indices = np.linspace(0, n_total - 1, n_obs, dtype=int)
    t_obs = t[obs_indices]
    rho_obs = rho[obs_indices]
    phi_obs = phi[obs_indices]

    # Collocation points (LHS, independent of data)
    t_coll = latin_hypercube_sampling(n_coll, 0.0, T_final, seed=42)
    t_coll = np.sort(t_coll)

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dataset_equatorial_proton{proton_id}.npz")

    np.savez(
        output_path,
        # Physical constants
        M=M, alpha1_true=ALPHA1_TRUE,
        # Initial conditions (known)
        rho0=rho0, drho0=drho0, phi0=phi0, dphi0=dphi0,
        # Solver parameters
        T_final=T_final, dt_solver=DT_SOLVER,
        # Observations (subsampled)
        t_obs=t_obs, rho_obs=rho_obs, phi_obs=phi_obs,
        # Full reference (all solver points)
        t_ref=t, rho_ref=rho, phi_ref=phi,
        # Collocation points
        t_collocation=t_coll,
    )

    print(f"\n  Dataset saved to {output_path}")
    print(f"  Observations: {n_obs} points")
    print(f"  Collocation: {n_coll} points")
    print(f"  Reference: {n_total} points")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate equatorial dataset for inverse Störmer PINN"
    )
    parser.add_argument("--proton", type=int, default=1, choices=PROTONS.keys(),
                        help="Proton number (default: 1)")
    parser.add_argument("--T-final", type=float, default=2.0,
                        help="Simulation time in seconds (default: 2.0)")
    parser.add_argument("--n-obs", type=int, default=1000,
                        help="Number of observation points (default: 1000)")
    parser.add_argument("--n-coll", type=int, default=3000,
                        help="Number of collocation points (default: 3000)")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating equatorial dataset for inverse Störmer PINN")
    print("=" * 60)
    generate(proton_id=args.proton, T_final=args.T_final,
             n_obs=args.n_obs, n_coll=args.n_coll)
