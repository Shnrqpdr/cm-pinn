"""
Generate training/validation datasets for the Störmer sphere PINN.

Uses the analytical solution (Jacobi elliptic functions) from Piña & Cortés (2016)
to produce reference data, collocation points, and initial conditions.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analytical"))
from stormer_sphere_analytical import solve_analytical, compute_constants

# Physical parameters (same as sv_sphere.c and the paper)
M, R, k = 2.0, 10.0, 0.5


def latin_hypercube_sampling(n_samples, t_min, t_max, seed=42):
    """Generate LHS points in [t_min, t_max]."""
    rng = np.random.default_rng(seed)
    intervals = np.linspace(t_min, t_max, n_samples + 1)
    points = np.array([rng.uniform(intervals[i], intervals[i + 1]) for i in range(n_samples)])
    rng.shuffle(points)
    return points


def compute_theta_period(params):
    """Compute the period of theta oscillation in physical time."""
    from scipy.special import ellipk

    a, b = params["a"], params["b"]
    A, B = params["A"], params["B"]
    kappa = params["kappa"]
    tau_scale = params["tau_scale"]
    m_param = kappa ** 2

    K_ellip = ellipk(m_param)

    if params["regime"] == "one_hemisphere":
        # dn has period 2K
        T_tau = 2 * K_ellip / (b * np.sqrt(B))
    else:
        # cn has period 4K
        T_tau = 4 * K_ellip / (b * np.sqrt(B - A))

    return T_tau * tau_scale  # physical time


def generate_case(name, theta0, p_theta0, phi0, p_phi0, n_periods=5,
                  n_reference=10000, n_collocation=3000, n_validation=2000,
                  seed=42):
    """Generate a complete dataset for one trajectory case.

    Parameters
    ----------
    name : str
        Case identifier.
    theta0, p_theta0, phi0, p_phi0 : float
        Initial conditions.
    n_periods : int
        Number of theta-oscillation periods to cover.
    n_reference : int
        Number of dense reference points.
    n_collocation : int
        Number of LHS collocation points for PINN training.
    n_validation : int
        Number of uniformly-spaced validation points.
    seed : int
        Random seed for reproducibility.
    """
    # Compute constants and period
    params = compute_constants(theta0, p_theta0, phi0, p_phi0, M, R, k)
    T_period = compute_theta_period(params)
    T_final = n_periods * T_period

    print(f"\nCase: {name}")
    print(f"  Regime: {params['regime']}")
    print(f"  a = {params['a']:.6f}, b = {params['b']:.6f}")
    print(f"  A = {params['A']:.6f}, B = {params['B']:.6f}")
    print(f"  kappa = {params['kappa']:.6f}")
    print(f"  Energy K = {params['K']:.6e}")
    print(f"  Theta period = {T_period:.4f}")
    print(f"  T_final = {T_final:.4f} ({n_periods} periods)")

    # Dense reference solution
    t_ref, theta_ref, phi_ref, _ = solve_analytical(
        theta0, p_theta0, phi0, p_phi0, T_final,
        n_points=n_reference, M=M, R=R, k=k,
    )

    # Collocation points (LHS in (0, T], excluding t=0 which is an IC point)
    t_collocation = latin_hypercube_sampling(n_collocation, 0.0, T_final, seed=seed)
    t_collocation = np.sort(t_collocation)

    # Validation points (uniform, distinct from collocation)
    rng = np.random.default_rng(seed + 1)
    t_validation = np.sort(rng.uniform(0, T_final, n_validation))

    # Interpolate analytical solution at validation points
    theta_validation = np.interp(t_validation, t_ref, theta_ref)
    phi_validation = np.interp(t_validation, t_ref, phi_ref)

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dataset_{name}.npz")

    np.savez(
        output_path,
        # Initial conditions
        theta0=theta0, p_theta0=p_theta0, phi0=phi0, p_phi0=p_phi0,
        # Physical parameters
        M=M, R=R, k=k,
        # Dimensionless parameters
        a=params["a"], b=params["b"], A=params["A"], B=params["B"],
        regime=params["regime"], kappa=params["kappa"],
        energy=params["K"],
        # Time domain
        T_period=T_period, T_final=T_final, n_periods=n_periods,
        # Dense reference (for plotting and fine-grained comparison)
        t_reference=t_ref, theta_reference=theta_ref, phi_reference=phi_ref,
        # Collocation points for PINN training (no labels)
        t_collocation=t_collocation,
        # Validation set (with labels, not used in training)
        t_validation=t_validation,
        theta_validation=theta_validation,
        phi_validation=phi_validation,
    )

    print(f"  Saved to {output_path}")
    print(f"  Reference: {n_reference} pts, Collocation: {n_collocation} pts, "
          f"Validation: {n_validation} pts")

    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Generating datasets for Störmer sphere PINN")
    print("=" * 60)

    # Case 1: One hemisphere (A > 0) — the simplest benchmark
    # Start with 2 periods for PINN feasibility, scale up later
    generate_case(
        name="case1_one_hemisphere",
        theta0=np.pi / 3,
        p_theta0=0.0,
        phi0=0.0,
        p_phi0=0.394,
        n_periods=2,
        n_reference=10000,
        n_collocation=3000,
        n_validation=2000,
    )
