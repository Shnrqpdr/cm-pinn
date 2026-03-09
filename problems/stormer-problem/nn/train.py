"""
Training script for the Störmer sphere PINN (z-formulation).

Phase 1: Adam optimizer with cosine annealing LR
Phase 2: L-BFGS refinement
"""

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_stormer import StormerPINN, StormerPINNTrainer


def train(dataset_path, output_dir="results", n_hidden=4, n_neurons=128,
          n_frequencies=10, adam_epochs=20000, adam_lr=1e-3,
          lbfgs_epochs=5, w_ode=1.0, w_ic=100.0, w_energy=10.0,
          device="cpu"):
    """Full training pipeline: Adam + L-BFGS."""

    os.makedirs(output_dir, exist_ok=True)

    # Use float64 for numerical stability
    torch.set_default_dtype(torch.float64)

    model = StormerPINN(n_hidden=n_hidden, n_neurons=n_neurons,
                        n_frequencies=n_frequencies)
    trainer = StormerPINNTrainer(model, dataset_path, device=device,
                                w_ode=w_ode, w_ic=w_ic, w_energy=w_energy)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Fourier frequencies: {n_frequencies}")
    print(f"Weights: w_ode={w_ode}, w_ic={w_ic}, w_energy={w_energy}")

    history = {"epoch": [], "total": [], "ode": [], "ic": [], "energy": []}

    # ------------------------------------------------------------------
    # Phase 1: Adam
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 1: Adam ({adam_epochs} epochs, lr={adam_lr})")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=adam_epochs, eta_min=1e-6)

    t_start = time.time()
    for epoch in range(1, adam_epochs + 1):
        model.train()
        optimizer.zero_grad()
        total, l_ode, l_ic, l_energy = trainer.total_loss()
        total.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["total"].append(total.item())
            history["ode"].append(l_ode.item())
            history["ic"].append(l_ic.item())
            history["energy"].append(l_energy.item())

            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:6d} | Total: {total.item():.4e} | "
                  f"ODE: {l_ode.item():.4e} | IC: {l_ic.item():.4e} | "
                  f"Energy: {l_energy.item():.4e} | lr: {lr_now:.2e} | "
                  f"t: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Phase 2: L-BFGS
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 2: L-BFGS ({lbfgs_epochs} steps)")
    print(f"{'='*60}")

    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0,
        max_iter=20, max_eval=25,
        tolerance_grad=1e-12, tolerance_change=1e-14,
        history_size=100, line_search_fn="strong_wolfe",
    )

    lbfgs_step = [0]
    best_loss = [float("inf")]

    def closure():
        lbfgs.zero_grad()
        total, l_ode, l_ic, l_energy = trainer.total_loss()

        if torch.isnan(total):
            return torch.tensor(best_loss[0], requires_grad=True)

        total.backward()
        lbfgs_step[0] += 1

        if total.item() < best_loss[0]:
            best_loss[0] = total.item()

        if lbfgs_step[0] % 100 == 0:
            print(f"  L-BFGS step {lbfgs_step[0]:4d} | Total: {total.item():.4e} | "
                  f"ODE: {l_ode.item():.4e} | IC: {l_ic.item():.4e} | "
                  f"Energy: {l_energy.item():.4e}")
        return total

    for _ in range(lbfgs_epochs):
        lbfgs.step(closure)

    # Final loss
    total, l_ode, l_ic, l_energy = trainer.total_loss()
    history["epoch"].append(adam_epochs + lbfgs_step[0])
    history["total"].append(total.item())
    history["ode"].append(l_ode.item())
    history["ic"].append(l_ic.item())
    history["energy"].append(l_energy.item())

    print(f"\nFinal loss: Total={total.item():.4e} | ODE={l_ode.item():.4e} | "
          f"IC={l_ic.item():.4e} | Energy={l_energy.item():.4e}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Validation")
    print(f"{'='*60}")

    val_metrics = trainer.validate()
    for key, val in val_metrics.items():
        print(f"  {key}: {val:.6e}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(output_dir, "pinn_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_hidden": n_hidden,
        "n_neurons": n_neurons,
        "n_frequencies": n_frequencies,
        "tau_final": trainer.tau_final,
        "T_final": trainer.T_final,
        "history": history,
        "val_metrics": val_metrics,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_results(trainer, history, output_dir)

    return model, trainer, history


def _plot_results(trainer, history, output_dir):
    """Generate all result plots."""

    # 1. Loss history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(history["epoch"], history["total"], "k-", label="Total", linewidth=1.5)
    ax.semilogy(history["epoch"], history["ode"], "b--", label="ODE", linewidth=1)
    ax.semilogy(history["epoch"], history["ic"], "r--", label="IC", linewidth=1)
    ax.semilogy(history["epoch"], history["energy"], "g--", label="Energy", linewidth=1)
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss history")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=200)
    plt.close()

    # 2. Predictions
    z_pred, w_pred, theta_pred, phi_pred = trainer.predict(trainer.t_ref)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # theta
    axes[0].plot(trainer.t_ref, np.degrees(trainer.theta_ref), "b-",
                 linewidth=1, label="Analytical")
    axes[0].plot(trainer.t_ref, np.degrees(theta_pred), "r--",
                 linewidth=1, label="PINN")
    axes[0].set_ylabel(r"$\theta$ (deg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("PINN vs Analytical Solution")

    # phi
    axes[1].plot(trainer.t_ref, trainer.phi_ref, "b-", linewidth=1, label="Analytical")
    axes[1].plot(trainer.t_ref, phi_pred, "r--", linewidth=1, label="PINN")
    axes[1].set_ylabel(r"$\varphi$ (rad)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # errors
    theta_err = np.abs(theta_pred - trainer.theta_ref)
    phi_err = np.abs(phi_pred - trainer.phi_ref)
    axes[2].semilogy(trainer.t_ref, np.degrees(theta_err), "b-",
                     linewidth=0.8, label=r"$|\Delta\theta|$ (deg)")
    axes[2].semilogy(trainer.t_ref, phi_err, "r-",
                     linewidth=0.8, label=r"$|\Delta\varphi|$ (rad)")
    axes[2].set_ylabel("Absolute error")
    axes[2].set_xlabel("$t$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pinn_vs_analytical.png"), dpi=200)
    plt.close()

    # 3. Energy integral check: w^2 vs b^2*(B-z^2)*(z^2-A)
    w2 = w_pred**2
    rhs = trainer.b**2 * (trainer.B - z_pred**2) * (z_pred**2 - trainer.A)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trainer.t_ref, np.abs(w2 - rhs), "b-", linewidth=0.8)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|w^2 - b^2(B-z^2)(z^2-A)|$")
    ax.set_title("Energy integral residual")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_conservation.png"), dpi=200)
    plt.close()

    # 4. 3D trajectory
    R = trainer.R
    x_an = R * np.sin(trainer.theta_ref) * np.cos(trainer.phi_ref)
    y_an = R * np.sin(trainer.theta_ref) * np.sin(trainer.phi_ref)
    z_an = R * np.cos(trainer.theta_ref)
    x_pr = R * np.sin(theta_pred) * np.cos(phi_pred)
    y_pr = R * np.sin(theta_pred) * np.sin(phi_pred)
    z_pr = R * np.cos(theta_pred)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(x_an, y_an, z_an, "b-", linewidth=0.8)
    ax1.set_title("Analytical")
    ax1.set_xlim(-R, R); ax1.set_ylim(-R, R); ax1.set_zlim(-R, R)
    ax1.view_init(elev=22, azim=-48)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(x_pr, y_pr, z_pr, "r-", linewidth=0.8)
    ax2.set_title("PINN")
    ax2.set_xlim(-R, R); ax2.set_ylim(-R, R); ax2.set_zlim(-R, R)
    ax2.view_init(elev=22, azim=-48)

    fig.suptitle("3D Trajectory on sphere", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_3d.png"), dpi=200)
    plt.close()

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(__file__), "data",
                           "dataset_case1_one_hemisphere.npz")
    results = os.path.join(os.path.dirname(__file__), "results", "case1")

    train(
        dataset_path=dataset,
        output_dir=results,
        n_hidden=4,
        n_neurons=128,
        n_frequencies=10,
        adam_epochs=20000,
        adam_lr=1e-3,
        lbfgs_epochs=500,
        w_ode=1.0,
        w_ic=100.0,
        w_energy=10.0,
    )
