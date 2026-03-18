"""
Training script for the inverse Störmer sphere PINN.

Identifies the magnetic coupling parameter k from observed trajectory data.

Training strategy (3 phases):
  Phase 0 (Warmup): Train network with data loss only, k frozen.
    Establishes a good trajectory approximation before introducing physics.
  Phase 1 (Adam):   Train network + k with data + ODE loss.
    k has separate (lower) learning rate to prevent overshoot.
  Phase 2 (L-BFGS): Fine-tune all parameters jointly.

Experiments:
  Exp A: Full data, no noise     (python train_inverse.py)
  Exp B: Sparse data (30%)       (python train_inverse.py --fraction 0.3)
  Exp C: Noisy data              (python train_inverse.py --noise 0.01)
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_stormer_inverse import StormerInversePINN, StormerInverseTrainer


def train(dataset_path, output_dir, k_init=1.0,
          n_hidden=4, n_neurons=128, n_frequencies=10,
          warmup_epochs=5000, adam_epochs=20000,
          adam_lr=1e-3, k_lr=1e-3,
          lbfgs_epochs=5,
          w_data=10.0, w_ode=1.0, w_energy=0.0,
          obs_fraction=1.0, noise_std=0.0, device="cpu"):
    """Full training pipeline: Warmup + Adam + L-BFGS."""

    os.makedirs(output_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)

    print(f"\n{'='*60}")
    print("Inverse Störmer PINN — Training")
    print(f"{'='*60}")
    print(f"  k_init = {k_init}")
    print(f"  obs_fraction = {obs_fraction}, noise_std = {noise_std}")
    print(f"  Weights: w_data={w_data}, w_ode={w_ode}, w_energy={w_energy}")
    print(f"  Warmup: {warmup_epochs} epochs (data only, k frozen)")

    model = StormerInversePINN(
        n_hidden=n_hidden, n_neurons=n_neurons,
        n_frequencies=n_frequencies, k_init=k_init,
    )
    trainer = StormerInverseTrainer(
        model, dataset_path, device=device,
        w_data=w_data, w_ode=w_ode, w_energy=w_energy,
        obs_fraction=obs_fraction, noise_std=noise_std,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,} (including k)")

    history = {
        "epoch": [], "total": [], "data": [], "ode": [], "energy": [], "k": []
    }

    # Helper to record history
    def _record(epoch, total, l_data, l_ode, l_energy, verbose=False):
        k_est, k_true, k_err = trainer.get_k_error()
        history["epoch"].append(epoch)
        history["total"].append(total.item())
        history["data"].append(l_data.item())
        history["ode"].append(l_ode.item())
        history["energy"].append(
            l_energy.item() if isinstance(l_energy, torch.Tensor) else 0.0
        )
        history["k"].append(k_est)
        if verbose:
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch:6d} | Total: {total.item():.4e} | "
                f"Data: {l_data.item():.4e} | ODE: {l_ode.item():.4e} | "
                f"k={k_est:.6f} (err={k_err:.4e}) | t={elapsed:.1f}s"
            )

    t_start = time.time()

    # ------------------------------------------------------------------
    # Phase 0: Warmup — data-only, k frozen
    # ------------------------------------------------------------------
    if warmup_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 0: Warmup ({warmup_epochs} epochs, data only, k frozen)")
        print(f"{'='*60}")

        model.k_raw.requires_grad_(False)
        net_params = [p for name, p in model.named_parameters()
                      if name != "k_raw"]
        warmup_opt = torch.optim.Adam(net_params, lr=adam_lr)
        warmup_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            warmup_opt, T_max=warmup_epochs, eta_min=1e-5,
        )

        for epoch in range(1, warmup_epochs + 1):
            model.train()
            warmup_opt.zero_grad()
            l_data = trainer.loss_data()
            l_data.backward()
            torch.nn.utils.clip_grad_norm_(net_params, max_norm=1.0)
            warmup_opt.step()
            warmup_sched.step()

            if epoch % 500 == 0 or epoch == 1:
                zero = torch.tensor(0.0, dtype=torch.float64)
                verbose = (epoch % 2000 == 0 or epoch == 1)
                _record(epoch, l_data, l_data, zero, zero, verbose=verbose)

        model.k_raw.requires_grad_(True)
        print(f"  Warmup done. Data loss: {l_data.item():.4e}")

    # ------------------------------------------------------------------
    # Phase 1: Adam — train network + k jointly
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 1: Adam ({adam_epochs} epochs, net_lr={adam_lr}, k_lr={k_lr})")
    print(f"{'='*60}")

    net_params = [p for name, p in model.named_parameters() if name != "k_raw"]
    optimizer = torch.optim.Adam([
        {"params": net_params, "lr": adam_lr},
        {"params": [model.k_raw], "lr": k_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=adam_epochs, eta_min=1e-6,
    )

    epoch_offset = warmup_epochs
    for epoch in range(1, adam_epochs + 1):
        model.train()
        optimizer.zero_grad()
        total, l_data, l_ode, l_energy = trainer.total_loss()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == 1:
            verbose = (epoch % 2000 == 0 or epoch == 1)
            _record(epoch_offset + epoch, total, l_data, l_ode, l_energy,
                    verbose=verbose)

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
        total, l_data, l_ode, l_energy = trainer.total_loss()
        if torch.isnan(total):
            return torch.tensor(best_loss[0], requires_grad=True)
        total.backward()
        lbfgs_step[0] += 1
        if total.item() < best_loss[0]:
            best_loss[0] = total.item()
        if lbfgs_step[0] % 50 == 0:
            k_est = model.get_k().item()
            print(f"  L-BFGS step {lbfgs_step[0]:4d} | "
                  f"Total: {total.item():.4e} | k={k_est:.6f}")
        return total

    for _ in range(lbfgs_epochs):
        lbfgs.step(closure)

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    total, l_data, l_ode, l_energy = trainer.total_loss()
    k_est, k_true, k_err = trainer.get_k_error()

    _record(epoch_offset + adam_epochs + lbfgs_step[0],
            total, l_data, l_ode, l_energy, verbose=False)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"  Total loss: {total.item():.4e}")
    print(f"  Data loss:  {l_data.item():.4e}")
    print(f"  ODE loss:   {l_ode.item():.4e}")
    print(f"  k_estimated = {k_est:.6f}")
    print(f"  k_true      = {k_true:.6f}")
    print(f"  k_rel_error = {k_err:.6e}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(output_dir, "pinn_inverse_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "k_estimated": k_est,
        "k_true": k_true,
        "k_rel_error": k_err,
        "history": history,
        "config": {
            "k_init": k_init, "n_hidden": n_hidden, "n_neurons": n_neurons,
            "n_frequencies": n_frequencies,
            "warmup_epochs": warmup_epochs,
            "adam_epochs": adam_epochs, "adam_lr": adam_lr, "k_lr": k_lr,
            "w_data": w_data, "w_ode": w_ode, "w_energy": w_energy,
            "obs_fraction": obs_fraction, "noise_std": noise_std,
        },
    }, model_path)
    print(f"\n  Model saved to {model_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_results(trainer, history, output_dir)

    return model, trainer, history


def _plot_results(trainer, history, output_dir):
    """Generate all result plots."""

    # 1. Loss history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(history["epoch"], history["total"], "k-",
                label="Total", linewidth=1.5)
    ax.semilogy(history["epoch"], history["data"], "b--",
                label="Data", linewidth=1)
    ax.semilogy(history["epoch"], history["ode"], "r--",
                label="ODE", linewidth=1)
    if any(e > 0 for e in history["energy"]):
        ax.semilogy(history["epoch"], history["energy"], "g--",
                    label="Energy", linewidth=1)
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss history (Inverse PINN)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=200)
    plt.close()

    # 2. k convergence
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(history["epoch"], history["k"], "b-", linewidth=1.5)
    axes[0].axhline(trainer.k_true, color="r", linestyle="--",
                    label=f"$k_{{true}}$ = {trainer.k_true}")
    axes[0].set_ylabel("$k$ estimated")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Parameter $k$ convergence")

    k_err = [abs(ki - trainer.k_true) / trainer.k_true for ki in history["k"]]
    axes[1].semilogy(history["epoch"], k_err, "b-", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("$|k_{est} - k_{true}| / k_{true}$")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_convergence.png"), dpi=200)
    plt.close()

    # 3. PINN vs true trajectory
    z_pred, w_pred, theta_pred, phi_pred = trainer.predict(trainer.t_ref)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(trainer.t_ref, np.degrees(trainer.theta_ref), "b-",
                 linewidth=1, label="True")
    axes[0].plot(trainer.t_ref, np.degrees(theta_pred), "r--",
                 linewidth=1, label="PINN")
    axes[0].set_ylabel(r"$\theta$ (deg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Inverse PINN: Reconstructed trajectory")

    axes[1].plot(trainer.t_ref, trainer.phi_ref, "b-",
                 linewidth=1, label="True")
    axes[1].plot(trainer.t_ref, phi_pred, "r--",
                 linewidth=1, label="PINN")
    axes[1].set_ylabel(r"$\varphi$ (rad)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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
    plt.savefig(os.path.join(output_dir, "pinn_vs_true.png"), dpi=200)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train inverse Störmer PINN to identify parameter k"
    )
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of observations to use (default: 1.0)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussian noise std on observations (default: 0.0)")
    parser.add_argument("--k-init", type=float, default=1.0,
                        help="Initial guess for k (default: 1.0, true=0.5)")
    parser.add_argument("--warmup-epochs", type=int, default=5000,
                        help="Warmup epochs (data only, k frozen) (default: 5000)")
    parser.add_argument("--adam-epochs", type=int, default=20000,
                        help="Number of Adam epochs (default: 20000)")
    parser.add_argument("--adam-lr", type=float, default=1e-3,
                        help="Learning rate for network (default: 0.001)")
    parser.add_argument("--k-lr", type=float, default=1e-3,
                        help="Learning rate for k parameter (default: 0.001)")
    parser.add_argument("--w-data", type=float, default=10.0,
                        help="Weight for data loss (default: 10.0)")
    parser.add_argument("--w-ode", type=float, default=1.0,
                        help="Weight for ODE loss (default: 1.0)")
    parser.add_argument("--w-energy", type=float, default=0.0,
                        help="Weight for energy loss (default: 0.0)")
    parser.add_argument("--case", type=int, default=1,
                        help="Case number for initial conditions (default: 1)")
    args = parser.parse_args()

    # Determine experiment name and output dir
    if args.noise > 0:
        exp_name = f"expC_noise{args.noise}"
    elif args.fraction < 1.0:
        exp_name = f"expB_{int(args.fraction * 100)}pct"
    else:
        exp_name = "expA"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    case_dir = f"case{args.case}"
    dataset = os.path.join(script_dir, "data", f"dataset_inverse_{case_dir}.npz")
    results = os.path.join(script_dir, "results", case_dir, exp_name)

    train(
        dataset_path=dataset,
        output_dir=results,
        k_init=args.k_init,
        warmup_epochs=args.warmup_epochs,
        adam_epochs=args.adam_epochs,
        adam_lr=args.adam_lr,
        k_lr=args.k_lr,
        w_data=args.w_data,
        w_ode=args.w_ode,
        w_energy=args.w_energy,
        obs_fraction=args.fraction,
        noise_std=args.noise,
    )
