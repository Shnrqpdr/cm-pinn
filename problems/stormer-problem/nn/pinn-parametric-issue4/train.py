"""
Training script for the Parametric Störmer sphere PINN.

Trains a single network that learns a family of solutions
parameterized by initial conditions (z0, w0, a, b).

Phase 1: Adam optimizer with cosine annealing LR
Phase 2: L-BFGS refinement
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pinn_stormer_parametric import ParametricStormerPINN, ParametricStormerTrainer


def train(dataset_path, output_dir="results", n_hidden=4, n_neurons=128,
          n_frequencies=10, adam_epochs=20000, adam_lr=1e-3,
          lbfgs_epochs=10, w_ode=1.0, w_ic=100.0, w_energy=10.0,
          device="cpu"):
    """Full training pipeline: Adam + L-BFGS."""

    os.makedirs(output_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)

    print("Loading dataset...")
    model = ParametricStormerPINN(n_hidden=n_hidden, n_neurons=n_neurons,
                                   n_frequencies=n_frequencies)
    trainer = ParametricStormerTrainer(model, dataset_path, device=device,
                                       w_ode=w_ode, w_ic=w_ic, w_energy=w_energy)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
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

        if lbfgs_step[0] % 50 == 0:
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
    # Validation (generalization to unseen ICs)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Validation — Held-out ICs (generalization)")
    print(f"{'='*60}")

    val_results = trainer.validate()
    for r in val_results:
        print(f"  {r['label']}")
        print(f"    theta_mae={r['theta_mae']:.4e}  theta_max={r['theta_max']:.4e}  "
              f"phi_mae={r['phi_mae']:.4e}  phi_max={r['phi_max']:.4e}")

    print(f"\n{'='*60}")
    print("Validation — Training ICs (fitting quality)")
    print(f"{'='*60}")

    train_results = trainer.validate_train()
    for r in train_results:
        print(f"  {r['label']}")
        print(f"    theta_mae={r['theta_mae']:.4e}  theta_max={r['theta_max']:.4e}  "
              f"phi_mae={r['phi_mae']:.4e}  phi_max={r['phi_max']:.4e}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(output_dir, "pinn_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_hidden": n_hidden,
        "n_neurons": n_neurons,
        "n_frequencies": n_frequencies,
        "history": history,
        "val_results": val_results,
        "train_results": train_results,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_results(trainer, history, val_results, train_results, output_dir)

    return model, trainer, history


def _plot_results(trainer, history, val_results, train_results, output_dir):
    """Generate all result plots."""

    R = 10.0  # sphere radius

    # 1. Loss history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(history["epoch"], history["total"], "k-", label="Total", linewidth=1.5)
    ax.semilogy(history["epoch"], history["ode"], "b--", label="ODE", linewidth=1)
    ax.semilogy(history["epoch"], history["ic"], "r--", label="IC", linewidth=1)
    ax.semilogy(history["epoch"], history["energy"], "g--", label="Energy", linewidth=1)
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Parametric PINN — Training loss history")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=200)
    plt.close()

    # 2. Per-IC comparison plots (training ICs — paper cases only, first 5)
    n_paper = min(5, trainer.n_train)
    fig, axes = plt.subplots(n_paper, 2, figsize=(14, 4 * n_paper), squeeze=False)

    for i in range(n_paper):
        t_ref = trainer.train_t_ref[i]
        theta_ref = trainer.train_theta_ref[i]
        phi_ref = trainer.train_phi_ref[i]

        z_pred, w_pred, theta_pred, phi_pred = trainer.predict(
            t_ref,
            float(trainer.params_ic[i, 0]), float(trainer.params_ic[i, 1]),
            trainer.train_phi0[i],
            float(trainer.params_ic[i, 2]), float(trainer.params_ic[i, 3]),
            trainer.train_tau_scale[i], trainer.train_tau_final[i],
        )

        # theta
        axes[i, 0].plot(t_ref, np.degrees(theta_ref), "b-", linewidth=1, label="Analytical")
        axes[i, 0].plot(t_ref, np.degrees(theta_pred), "r--", linewidth=1, label="PINN")
        axes[i, 0].set_ylabel(r"$\theta$ (deg)")
        axes[i, 0].set_title(trainer.train_labels[i])
        axes[i, 0].legend(fontsize="small")
        axes[i, 0].grid(True, alpha=0.3)

        # phi
        axes[i, 1].plot(t_ref, phi_ref, "b-", linewidth=1, label="Analytical")
        axes[i, 1].plot(t_ref, phi_pred, "r--", linewidth=1, label="PINN")
        axes[i, 1].set_ylabel(r"$\varphi$ (rad)")
        axes[i, 1].set_title(trainer.train_labels[i])
        axes[i, 1].legend(fontsize="small")
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("$t$")
    axes[-1, 1].set_xlabel("$t$")
    fig.suptitle("Parametric PINN — Training ICs (paper cases)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_ics_comparison.png"), dpi=200)
    plt.close()

    # 3. Validation ICs (generalization)
    n_val = trainer.n_val
    if n_val > 0:
        fig, axes = plt.subplots(n_val, 2, figsize=(14, 4 * n_val), squeeze=False)

        for i in range(n_val):
            t_ref = trainer.val_t_ref[i]
            theta_ref = trainer.val_theta_ref[i]
            phi_ref = trainer.val_phi_ref[i]

            z_pred, w_pred, theta_pred, phi_pred = trainer.predict(
                t_ref,
                trainer.val_z0[i], trainer.val_w0[i],
                trainer.val_phi0[i],
                trainer.val_a[i], trainer.val_b[i],
                trainer.val_tau_scale[i], trainer.val_tau_final[i],
            )

            axes[i, 0].plot(t_ref, np.degrees(theta_ref), "b-", linewidth=1,
                            label="Analytical")
            axes[i, 0].plot(t_ref, np.degrees(theta_pred), "r--", linewidth=1,
                            label="PINN")
            axes[i, 0].set_ylabel(r"$\theta$ (deg)")
            axes[i, 0].set_title(f"Val: {trainer.val_labels[i]}")
            axes[i, 0].legend(fontsize="small")
            axes[i, 0].grid(True, alpha=0.3)

            axes[i, 1].plot(t_ref, phi_ref, "b-", linewidth=1, label="Analytical")
            axes[i, 1].plot(t_ref, phi_pred, "r--", linewidth=1, label="PINN")
            axes[i, 1].set_ylabel(r"$\varphi$ (rad)")
            axes[i, 1].set_title(f"Val: {trainer.val_labels[i]}")
            axes[i, 1].legend(fontsize="small")
            axes[i, 1].grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("$t$")
        axes[-1, 1].set_xlabel("$t$")
        fig.suptitle("Parametric PINN — Validation ICs (unseen)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_ics_comparison.png"), dpi=200)
        plt.close()

    # 4. Error summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training errors
    labels_t = [r["label"][:20] for r in train_results[:n_paper]]
    mae_theta_t = [r["theta_mae"] for r in train_results[:n_paper]]
    mae_phi_t = [r["phi_mae"] for r in train_results[:n_paper]]

    x = np.arange(len(labels_t))
    w = 0.35
    axes[0].bar(x - w / 2, mae_theta_t, w, label=r"$\theta$ MAE", color="steelblue")
    axes[0].bar(x + w / 2, mae_phi_t, w, label=r"$\varphi$ MAE", color="coral")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MAE (rad)")
    axes[0].set_title("Training ICs")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_t, rotation=45, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Validation errors
    if val_results:
        labels_v = [r["label"][:20] for r in val_results]
        mae_theta_v = [r["theta_mae"] for r in val_results]
        mae_phi_v = [r["phi_mae"] for r in val_results]

        x = np.arange(len(labels_v))
        axes[1].bar(x - w / 2, mae_theta_v, w, label=r"$\theta$ MAE", color="steelblue")
        axes[1].bar(x + w / 2, mae_phi_v, w, label=r"$\varphi$ MAE", color="coral")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("MAE (rad)")
        axes[1].set_title("Validation ICs (unseen)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels_v, rotation=45, ha="right", fontsize=8)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_summary.png"), dpi=200)
    plt.close()

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(__file__), "data",
                           "dataset_parametric.npz")
    results = os.path.join(os.path.dirname(__file__), "results")

    train(
        dataset_path=dataset,
        output_dir=results,
        n_hidden=4,
        n_neurons=128,
        n_frequencies=10,
        adam_epochs=20000,
        adam_lr=1e-3,
        lbfgs_epochs=10,
        w_ode=1.0,
        w_ic=100.0,
        w_energy=10.0,
    )
