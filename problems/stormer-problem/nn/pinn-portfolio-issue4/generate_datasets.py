"""
Generate training datasets for the Störmer sphere PINN portfolio.

One dataset per paper case, using the analytical solution as reference.
Each dataset is self-contained and compatible with StormerPINNTrainer.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analytical"))
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

    A, B = params["A"], params["B"]
    kappa = params["kappa"]
    b = params["b"]
    tau_scale = params["tau_scale"]
    m_param = kappa ** 2

    K_ellip = ellipk(m_param)

    if params["regime"] == "one_hemisphere":
        T_tau = 2 * K_ellip / (b * np.sqrt(B))
    else:
        T_tau = 4 * K_ellip / (b * np.sqrt(B - A))

    return T_tau * tau_scale


def generate_case(name, theta0, p_theta0, phi0, p_phi0, n_periods=2,
                  n_reference=10000, n_collocation=3000, n_validation=2000,
                  seed=42):
    """Generate a complete dataset for one trajectory case."""
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

    # Collocation points (LHS)
    t_collocation = latin_hypercube_sampling(n_collocation, 0.0, T_final, seed=seed)
    t_collocation = np.sort(t_collocation)

    # Validation points (uniform random)
    rng = np.random.default_rng(seed + 1)
    t_validation = np.sort(rng.uniform(0, T_final, n_validation))
    theta_validation = np.interp(t_validation, t_ref, theta_ref)
    phi_validation = np.interp(t_validation, t_ref, phi_ref)

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dataset_{name}.npz")

    np.savez(
        output_path,
        theta0=theta0, p_theta0=p_theta0, phi0=phi0, p_phi0=p_phi0,
        M=M, R=R, k=k,
        a=params["a"], b=params["b"], A=params["A"], B=params["B"],
        regime=params["regime"], kappa=params["kappa"],
        energy=params["K"],
        T_period=T_period, T_final=T_final, n_periods=n_periods,
        t_reference=t_ref, theta_reference=theta_ref, phi_reference=phi_ref,
        t_collocation=t_collocation,
        t_validation=t_validation,
        theta_validation=theta_validation,
        phi_validation=phi_validation,
    )

    print(f"  Saved to {output_path}")
    return output_path


# All cases from Piña & Cortés (2016)
PAPER_CASES = {
    "fig6a": {
        "theta0": np.pi / 4, "p_theta0": 0.0,
        "phi0": 0.0, "p_phi0": 0.394,
        "label": "Fig 6(a): two hemispheres",
    },
    "fig6b": {
        "theta0": np.pi / 3, "p_theta0": 0.0,
        "phi0": 0.0, "p_phi0": 0.394,
        "label": "Fig 6(b): one hemisphere (Case 1)",
    },
    "fig6c": {
        "theta0": 75 * np.pi / 180, "p_theta0": 0.0,
        "phi0": 0.0, "p_phi0": -0.394,
        "label": "Fig 6(c): loops (p_phi < 0)",
    },
    "fig7a": {
        "theta0": 0.6, "p_theta0": 0.1,
        "phi0": 0.0, "p_phi0": 0.25,
        "label": "Fig 7(a): one hemisphere (p_theta != 0)",
    },
    "fig7c": {
        "theta0": 0.6, "p_theta0": 0.2525,
        "phi0": 0.0, "p_phi0": 0.25,
        "label": "Fig 7(c): crosses equator",
    },
}


if __name__ == "__main__":
    print("=" * 60)
    print("Generating datasets for Störmer sphere PINN portfolio")
    print("=" * 60)

    for name, case in PAPER_CASES.items():
        generate_case(
            name=name,
            theta0=case["theta0"],
            p_theta0=case["p_theta0"],
            phi0=case["phi0"],
            p_phi0=case["p_phi0"],
            n_periods=2,
            n_reference=10000,
            n_collocation=3000,
            n_validation=2000,
        )

    print("\nAll datasets generated.")
