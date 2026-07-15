"""
Generate validation figures for the sphere section of the article.

Produces three separate figures, one per case:
  - validation_sphere_banda.pdf: band trajectory in northern hemisphere
  - validation_sphere_equador.pdf: trajectory crossing the equator
  - validation_sphere_loops.pdf: looping trajectory (-k < p_phi < 0)

Each figure: 2 rows x 1 column (3D trajectory on top, trajectory error below).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "text.usetex": False,
})

# Add analytical module to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICAL_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..",
                              "problems", "stormer-problem", "analytical")
sys.path.insert(0, ANALYTICAL_DIR)

from stormer_sphere_analytical import solve_analytical, to_cartesian

# Paths
SV_DIR = os.path.join(ANALYTICAL_DIR, "..", "simulation", "constraint_case", "sphere")
SV_BIN = os.path.join(SV_DIR, "sv")

import subprocess

# Physical parameters
M_val, R_val, k_val = 2.0, 10.0, 0.5
DT = 0.0002  # from sv_sphere.c


def run_stormer_verlet(t_final, theta0, p_theta0, phi0, p_phi0):
    """Run the C Störmer-Verlet integrator and return positions + energy."""
    data_dir = os.path.join(SV_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    particle_file = os.path.join(data_dir, "validate_article.dat")
    phase_file = os.path.join(data_dir, "validate_article_phase.dat")

    cmd = [SV_BIN, str(t_final), str(theta0), str(p_theta0),
           str(phi0), str(p_phi0), particle_file, phase_file]
    subprocess.run(cmd, check=True, capture_output=True)

    # Cartesian positions
    data = np.loadtxt(particle_file, skiprows=1)
    n_steps = data[:, 0].astype(int)
    x_sv, y_sv, z_sv = data[:, 1], data[:, 2], data[:, 3]
    t_sv = n_steps * DT

    # Phase space: theta, p_theta, p_phi
    phase = np.loadtxt(phase_file, skiprows=1)
    theta_sv = phase[:, 1]
    phi_sv = np.zeros_like(theta_sv)
    # Recover phi from cartesian: phi = atan2(y, x)
    phi_sv = np.arctan2(y_sv, x_sv)

    return t_sv, x_sv, y_sv, z_sv, theta_sv, phi_sv


def draw_sphere(ax, R, alpha=0.15):
    """Draw a transparent sphere."""
    N = 200
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones(N), np.cos(v))
    ticks = np.arange(-R_val, R_val + 0.1, 5.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.plot_surface(xs, ys, zs, edgecolor="#a0a0a0", lw=0.1,
                    rstride=20, cstride=20, color="#e0e0e0", alpha=alpha)


def make_figure(case, output_name):
    """Generate a single 2-row x 1-col figure (3D trajectory on top, position error below)."""
    print(f"Processing: {case['name']}...")

    # Störmer-Verlet
    t_sv, x_sv, y_sv, z_sv, theta_sv, phi_sv = run_stormer_verlet(
        case["t_final"], case["theta0"], case["p_theta0"],
        case["phi0"], case["p_phi0"],
    )

    # Analytical solution at the same time points as SV
    n_sv = len(t_sv)
    t_an, theta_an, phi_an, params = solve_analytical(
        case["theta0"], case["p_theta0"], case["phi0"], case["p_phi0"],
        case["t_final"], n_points=n_sv, M=M_val, R=R_val, k=k_val,
    )
    x_an, y_an, z_an = to_cartesian(theta_an, phi_an, R_val)

    # Position error: Euclidean distance / R (dimensionless)
    dist = np.sqrt((x_sv - x_an)**2 + (y_sv - y_an)**2 + (z_sv - z_an)**2)
    dist_rel = dist / R_val

    print(f"  Regime: {params['regime']}")
    print(f"  a = {params['a']:.4f}, b = {params['b']:.4f}")
    print(f"  |d/R| mean={dist_rel.mean():.4e}, max={dist_rel.max():.4e}")

    n_steps_total = len(t_sv)

    fig = plt.figure(figsize=(6, 9))

    # --- Top: 3D trajectory ---
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    draw_sphere(ax, R_val)

    step_sv = max(1, n_steps_total // 12000)
    step_an = max(1, len(t_an) // 12000)

    ax.plot(x_sv[::step_sv], y_sv[::step_sv], z_sv[::step_sv],
            "-", linewidth=1.6, color="blue", label="Störmer-Verlet", alpha=0.8)
    ax.plot(x_an[::step_an], y_an[::step_an], z_an[::step_an],
            "--", linewidth=1.2, color="red", label="Analítico", alpha=0.8)

    ticks = np.arange(-R_val, R_val + 0.1, 5.0)
    ax.set_xlim(-R_val, R_val)
    ax.set_ylim(-R_val, R_val)
    ax.set_zlim(-R_val, R_val)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.view_init(elev=22, azim=-48)
    ax.set_title("Comparação das trajetórias")
    ax.legend(loc="upper right", fontsize=8)

    # --- Bottom: trajectory error ---
    ax2 = fig.add_subplot(2, 1, 2)
    step_e = max(1, n_steps_total // 5000)
    ax2.plot(t_sv[::step_e], dist_rel[::step_e], "k-", linewidth=0.5)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"$\| \Delta\mathbf{r}(t) \| \, / \, R$")
    ax2.set_title(
        f"Erro de posição "
        f"($\\Delta t = {DT}$)",
        fontsize=10
    )
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))

    plt.tight_layout()
    pdf_path = os.path.join(SCRIPT_DIR, f"{output_name}.pdf")
    png_path = os.path.join(SCRIPT_DIR, f"{output_name}.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {pdf_path}")


# Define the three cases
cases = [
    {
        "name": "Banda no hemisfério norte",
        "theta0": np.pi / 3, "p_theta0": 0.0,
        "phi0": 0.0, "p_phi0": 0.394,
        "t_final": 3000.0,
        "output": "validation_sphere_banda",
    },
    {
        "name": "Cruza o equador",
        "theta0": 0.6, "p_theta0": 0.2525,
        "phi0": 0.0, "p_phi0": 0.25,
        "t_final": 3000.0,
        "output": "validation_sphere_equador",
    },
    {
        "name": "Trajetória com laços (loops)",
        "theta0": np.pi / 4, "p_theta0": 0.3,
        "phi0": 0.0, "p_phi0": -0.15,
        "t_final": 10000.0,
        "output": "validation_sphere_loops",
    },
]

for case in cases:
    make_figure(case, case["output"])

print("\nDone.")
