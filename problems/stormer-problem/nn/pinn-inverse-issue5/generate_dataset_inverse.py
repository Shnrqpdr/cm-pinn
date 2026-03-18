"""
Generate dataset for inverse Störmer PINN (Issue #5).

Creates observed trajectory data from the analytical solution with known k,
plus collocation points for ODE evaluation. The same dataset is used for
all experiments (A, B, C) — sparsity and noise are applied at training time.

Supports multiple initial condition cases via --case flag.
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analytical"))
from stormer_sphere_analytical import solve_analytical, compute_constants
from scipy.special import ellipk

# Physical parameters (same as forward PINN)
M, R, k_true = 2.0, 10.0, 0.5

# Initial conditions for each case
CASES = {
    1: {
        "theta0": np.pi / 3, "p_theta0": 0.0,
        "phi0": 0.0, "p_phi0": 0.394,
        "description": "one hemisphere (fig6b, Piña & Cortés 2016)",
    },
    2: {
        "theta0": np.pi / 4, "p_theta0": 0.0,
        "phi0": 0.0, "p_phi0": 0.394,
        "description": "one hemisphere, smaller theta0 (fig6a, Piña & Cortés 2016)",
    },
    3: {
        "theta0": 0.6, "p_theta0": 0.1,
        "phi0": 0.0, "p_phi0": 0.25,
        "description": "one hemisphere, p_theta != 0 (fig7a, Piña & Cortés 2016)",
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


def generate(case_id=1):
    case = CASES[case_id]
    theta0 = case["theta0"]
    p_theta0 = case["p_theta0"]
    phi0 = case["phi0"]
    p_phi0 = case["p_phi0"]

    print(f"  Case {case_id}: {case['description']}")
    print(f"  theta0={theta0:.4f}, p_theta0={p_theta0}, "
          f"phi0={phi0}, p_phi0={p_phi0}")

    params = compute_constants(theta0, p_theta0, phi0, p_phi0, M, R, k_true)

    # Compute period
    kappa = params["kappa"]
    m_param = kappa ** 2
    K_ellip = ellipk(m_param)
    tau_scale = params["tau_scale"]
    B_val = params["B"]
    T_period = (2 * K_ellip / (params["b"] * np.sqrt(B_val))) * tau_scale

    n_periods = 2
    T_final = n_periods * T_period

    # Dense observation data (1000 points)
    n_obs = 1000
    t_obs, theta_obs, phi_obs, _ = solve_analytical(
        theta0, p_theta0, phi0, p_phi0, T_final,
        n_points=n_obs, M=M, R=R, k=k_true,
    )

    # Dense reference for plotting (10000 points)
    t_ref, theta_ref, phi_ref, _ = solve_analytical(
        theta0, p_theta0, phi0, p_phi0, T_final,
        n_points=10000, M=M, R=R, k=k_true,
    )

    # Collocation points (3000, LHS)
    t_coll = latin_hypercube_sampling(3000, 0.0, T_final, seed=42)
    t_coll = np.sort(t_coll)

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dataset_inverse_case{case_id}.npz")

    np.savez(
        output_path,
        # Known parameters
        M=M, R=R, k_true=k_true,
        # Initial conditions (known)
        theta0=theta0, p_theta0=p_theta0, phi0=phi0, p_phi0=p_phi0,
        # Observation data (full — subsample at training time)
        t_obs=t_obs, theta_obs=theta_obs, phi_obs=phi_obs,
        # Collocation points for ODE residual
        t_collocation=t_coll,
        # Dense reference for plotting/validation
        t_reference=t_ref, theta_reference=theta_ref, phi_reference=phi_ref,
        # Time domain
        T_final=T_final, T_period=T_period, n_periods=n_periods,
        # True dimensionless params (for validation only)
        a_true=params["a"], b_true=params["b"],
        A_true=params["A"], B_true=params["B"],
        tau_scale_true=params["tau_scale"], energy_true=params["K"],
    )

    print(f"Dataset saved to {output_path}")
    print(f"  T_final = {T_final:.2f} ({n_periods} periods)")
    print(f"  Observations: {n_obs} points")
    print(f"  Collocation: 3000 points")
    print(f"  Reference: 10000 points")
    print(f"  k_true = {k_true}")
    print(f"  Regime: {params['regime']}")
    print(f"  a = {params['a']:.6f}, b = {params['b']:.6f}")
    print(f"  A = {params['A']:.6f}, B = {params['B']:.6f}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset for inverse Störmer PINN"
    )
    parser.add_argument("--case", type=int, default=1, choices=CASES.keys(),
                        help="Case number (default: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating dataset for inverse Störmer PINN")
    print("=" * 60)
    generate(case_id=args.case)
