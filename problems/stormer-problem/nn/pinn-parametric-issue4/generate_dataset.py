"""
Generate multi-IC dataset for the Parametric Störmer sphere PINN.

Uses:
  - All 5 cases from Piña & Cortés (2016)
  - Additional sampled ICs for diversity and generalization
  - Held-out validation ICs to test interpolation

Each IC gets 2 periods of theta oscillation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analytical"))
from stormer_sphere_analytical import solve_analytical, compute_constants

# Physical parameters (same as sv_sphere.c and the paper)
M, R, k = 2.0, 10.0, 0.5


def compute_theta_period(params):
    """Compute the period of theta oscillation in physical time."""
    from scipy.special import ellipk

    kappa = params["kappa"]
    b = params["b"]
    A, B = params["A"], params["B"]
    m_param = kappa ** 2

    K_ellip = ellipk(m_param)

    if params["regime"] == "one_hemisphere":
        T_tau = 2 * K_ellip / (b * np.sqrt(B))
    else:
        T_tau = 4 * K_ellip / (b * np.sqrt(B - A))

    return T_tau * params["tau_scale"]


def latin_hypercube_sampling(n_samples, t_min, t_max, seed=42):
    """Generate LHS points in [t_min, t_max]."""
    rng = np.random.default_rng(seed)
    intervals = np.linspace(t_min, t_max, n_samples + 1)
    points = np.array([rng.uniform(intervals[i], intervals[i + 1])
                       for i in range(n_samples)])
    rng.shuffle(points)
    return points


def compute_ic_params(theta0, p_theta0, phi0, p_phi0):
    """Compute all dimensionless params for an IC. Returns None if invalid."""
    try:
        params = compute_constants(theta0, p_theta0, phi0, p_phi0, M, R, k)
    except (ValueError, NotImplementedError):
        return None

    if params["regime"] == "separatrix":
        return None

    K = params["K"]
    L = np.sqrt(2 * M * R**2 * K)
    z0 = np.cos(theta0)
    w0 = -np.sin(theta0) * p_theta0 / L

    T_period = compute_theta_period(params)
    T_final = 2 * T_period  # 2 periods
    tau_final = T_final / params["tau_scale"]

    return {
        "theta0": theta0, "p_theta0": p_theta0,
        "phi0": phi0, "p_phi0": p_phi0,
        "z0": z0, "w0": w0,
        "a": params["a"], "b": params["b"],
        "A": params["A"], "B": params["B"],
        "tau_final": tau_final,
        "tau_scale": params["tau_scale"],
        "T_final": T_final, "T_period": T_period,
        "regime": params["regime"],
    }


def generate_reference(ic, n_ref=2000):
    """Generate analytical reference solution for an IC."""
    t, theta, phi, _ = solve_analytical(
        ic["theta0"], ic["p_theta0"], ic["phi0"], ic["p_phi0"],
        ic["T_final"], n_points=n_ref, M=M, R=R, k=k,
    )
    return t, theta, phi


def sample_ics(n_samples, seed=123):
    """Sample valid ICs from parameter space using LHS."""
    rng = np.random.default_rng(seed)

    # Sample in (theta0, p_theta0, p_phi0) space
    # theta0 in [0.3, 1.5], p_theta0 in [-0.2, 0.2], p_phi0 in [-0.5, 0.5]
    ics = []
    attempts = 0
    max_attempts = n_samples * 20

    # LHS-like: divide each dimension into n_samples intervals
    theta0_vals = np.linspace(0.3, 1.5, n_samples + 2)[1:-1]
    rng.shuffle(theta0_vals)

    p_phi0_vals = np.linspace(-0.45, 0.45, n_samples + 2)[1:-1]
    rng.shuffle(p_phi0_vals)

    p_theta0_vals = np.linspace(-0.15, 0.15, n_samples + 2)[1:-1]
    rng.shuffle(p_theta0_vals)

    for i in range(n_samples):
        # Add small random perturbation
        theta0 = theta0_vals[i] + rng.uniform(-0.05, 0.05)
        theta0 = np.clip(theta0, 0.2, 1.5)
        p_phi0 = p_phi0_vals[i] + rng.uniform(-0.02, 0.02)
        p_theta0 = p_theta0_vals[i] + rng.uniform(-0.02, 0.02)

        ic = compute_ic_params(theta0, p_theta0, 0.0, p_phi0)
        if ic is not None:
            ics.append(ic)

    return ics


def main():
    print("=" * 60)
    print("Generating multi-IC dataset for Parametric Störmer PINN")
    print("=" * 60)

    n_coll_per_ic = 500
    n_ref_per_ic = 2000
    n_periods = 2

    # --- Paper cases (all 5) ---
    paper_cases = [
        {"theta0": np.pi / 4, "p_theta0": 0.0, "phi0": 0.0,
         "p_phi0": 0.394, "label": "fig6a: one hemisphere"},
        {"theta0": np.pi / 3, "p_theta0": 0.0, "phi0": 0.0,
         "p_phi0": 0.394, "label": "fig6b: one hemisphere (Case 1)"},
        {"theta0": 75 * np.pi / 180, "p_theta0": 0.0, "phi0": 0.0,
         "p_phi0": -0.394, "label": "fig6c: loops (p_phi < 0)"},
        {"theta0": 0.6, "p_theta0": 0.1, "phi0": 0.0,
         "p_phi0": 0.25, "label": "fig7a: one hemisphere (p_theta != 0)"},
        {"theta0": 0.6, "p_theta0": 0.2525, "phi0": 0.0,
         "p_phi0": 0.25, "label": "fig7c: crosses equator"},
    ]

    train_ics = []
    train_labels = []

    print("\n--- Paper cases ---")
    for pc in paper_cases:
        ic = compute_ic_params(pc["theta0"], pc["p_theta0"],
                               pc["phi0"], pc["p_phi0"])
        if ic is None:
            print(f"  SKIP: {pc['label']}")
            continue
        # Filter out ICs with extreme parameters
        if abs(ic["a"]) > 3.0 or ic["b"] > 3.0:
            print(f"  SKIP {pc['label']} (extreme params: a={ic['a']:.2f} b={ic['b']:.2f})")
            continue
        ic["label"] = pc["label"]
        train_ics.append(ic)
        train_labels.append(pc["label"])
        print(f"  {pc['label']}: regime={ic['regime']} a={ic['a']:.4f} "
              f"b={ic['b']:.4f} A={ic['A']:.4f} B={ic['B']:.4f}")

    # --- Sampled ICs ---
    print("\n--- Sampled training ICs ---")
    sampled_train = sample_ics(15, seed=42)
    for i, ic in enumerate(sampled_train):
        # Filter out ICs with extreme parameters (|a| or b too large)
        if abs(ic["a"]) > 3.0 or ic["b"] > 3.0:
            print(f"  SKIP sampled_train_{i} (extreme params: a={ic['a']:.2f} b={ic['b']:.2f})")
            continue
        label = f"sampled_train_{i}: theta0={ic['theta0']:.2f} pphi={ic['p_phi0']:.3f}"
        ic["label"] = label
        train_ics.append(ic)
        train_labels.append(label)
        print(f"  {label}: regime={ic['regime']} a={ic['a']:.4f} b={ic['b']:.4f}")

    # --- Validation ICs (held out for generalization test) ---
    print("\n--- Validation ICs (held out) ---")
    val_ics_raw = sample_ics(5, seed=999)
    val_ics = []
    val_labels = []
    for i, ic in enumerate(val_ics_raw):
        if abs(ic["a"]) > 3.0 or ic["b"] > 3.0:
            print(f"  SKIP val_{i} (extreme params: a={ic['a']:.2f} b={ic['b']:.2f})")
            continue
        label = f"val_{i}: theta0={ic['theta0']:.2f} pphi={ic['p_phi0']:.3f}"
        ic["label"] = label
        val_ics.append(ic)
        val_labels.append(label)
        print(f"  {label}: regime={ic['regime']} a={ic['a']:.4f} b={ic['b']:.4f}")

    n_train = len(train_ics)
    n_val = len(val_ics)

    print(f"\nTotal: {n_train} training ICs, {n_val} validation ICs")

    # --- Generate reference solutions ---
    print("\nGenerating analytical reference solutions...")

    train_t_ref = np.zeros((n_train, n_ref_per_ic))
    train_theta_ref = np.zeros((n_train, n_ref_per_ic))
    train_phi_ref = np.zeros((n_train, n_ref_per_ic))

    for i, ic in enumerate(train_ics):
        t, theta, phi = generate_reference(ic, n_ref_per_ic)
        train_t_ref[i] = t
        train_theta_ref[i] = theta
        train_phi_ref[i] = phi

    val_t_ref = np.zeros((n_val, n_ref_per_ic))
    val_theta_ref = np.zeros((n_val, n_ref_per_ic))
    val_phi_ref = np.zeros((n_val, n_ref_per_ic))

    for i, ic in enumerate(val_ics):
        t, theta, phi = generate_reference(ic, n_ref_per_ic)
        val_t_ref[i] = t
        val_theta_ref[i] = theta
        val_phi_ref[i] = phi

    # --- Collocation points ---
    print("Generating collocation points...")

    all_tau_norm = []
    all_ic_idx = []

    for i, ic in enumerate(train_ics):
        # LHS in (0, T_final] then convert to tau_norm
        t_coll = latin_hypercube_sampling(n_coll_per_ic, 0.0, ic["T_final"],
                                          seed=42 + i)
        tau_coll = t_coll / ic["tau_scale"]
        tau_norm_coll = tau_coll / ic["tau_final"]

        all_tau_norm.append(tau_norm_coll)
        all_ic_idx.append(np.full(n_coll_per_ic, i, dtype=int))

    tau_norm_coll = np.concatenate(all_tau_norm)
    ic_idx_coll = np.concatenate(all_ic_idx)

    # --- Pack into arrays ---
    def extract_array(ics_list, key):
        return np.array([ic[key] for ic in ics_list])

    # --- Save ---
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_parametric.npz")

    np.savez(
        output_path,
        # Metadata
        n_ics_train=n_train,
        n_ics_val=n_val,
        n_coll_per_ic=n_coll_per_ic,
        n_ref_per_ic=n_ref_per_ic,
        n_periods=n_periods,
        M=M, R=R, k=k,

        # Training IC params
        train_theta0=extract_array(train_ics, "theta0"),
        train_p_theta0=extract_array(train_ics, "p_theta0"),
        train_phi0=extract_array(train_ics, "phi0"),
        train_p_phi0=extract_array(train_ics, "p_phi0"),
        train_z0=extract_array(train_ics, "z0"),
        train_w0=extract_array(train_ics, "w0"),
        train_a=extract_array(train_ics, "a"),
        train_b=extract_array(train_ics, "b"),
        train_A=extract_array(train_ics, "A"),
        train_B=extract_array(train_ics, "B"),
        train_tau_final=extract_array(train_ics, "tau_final"),
        train_tau_scale=extract_array(train_ics, "tau_scale"),
        train_labels=np.array(train_labels),

        # Collocation
        tau_norm_coll=tau_norm_coll,
        ic_idx_coll=ic_idx_coll,

        # Training reference (plotting)
        train_t_ref=train_t_ref,
        train_theta_ref=train_theta_ref,
        train_phi_ref=train_phi_ref,

        # Validation IC params
        val_theta0=extract_array(val_ics, "theta0"),
        val_p_theta0=extract_array(val_ics, "p_theta0"),
        val_phi0=extract_array(val_ics, "phi0"),
        val_p_phi0=extract_array(val_ics, "p_phi0"),
        val_z0=extract_array(val_ics, "z0"),
        val_w0=extract_array(val_ics, "w0"),
        val_a=extract_array(val_ics, "a"),
        val_b=extract_array(val_ics, "b"),
        val_A=extract_array(val_ics, "A"),
        val_B=extract_array(val_ics, "B"),
        val_tau_final=extract_array(val_ics, "tau_final"),
        val_tau_scale=extract_array(val_ics, "tau_scale"),
        val_labels=np.array(val_labels),

        # Validation reference
        val_t_ref=val_t_ref,
        val_theta_ref=val_theta_ref,
        val_phi_ref=val_phi_ref,
    )

    print(f"\nDataset saved to {output_path}")
    print(f"  Total collocation points: {len(tau_norm_coll)}")


if __name__ == "__main__":
    main()
