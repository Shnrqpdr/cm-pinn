"""
Training script for the inverse Störmer PINN — 3D case (Phase 2).

Identifies alpha1 from observed (rho, Z, phi) trajectory data.

Training strategy (3 phases):
  Phase 0 (Warmup): Train network with data loss only, alpha1 frozen.
  Phase 1 (Adam):   Train network + alpha1 with data + ODE loss.
  Phase 2 (L-BFGS): Fine-tune all parameters jointly.

Usage:
  python train_inverse_3d.py                        # Exp A
  python train_inverse_3d.py --fraction 0.3         # Exp B 30%
  python train_inverse_3d.py --fraction 0.05        # Exp B 5%
  python train_inverse_3d.py --noise 0.01           # Exp C1
  python train_inverse_3d.py --noise 0.05           # Exp C2
  python train_inverse_3d.py --noise 0.1            # Exp C3
  python train_inverse_3d.py --alpha1-init 1500     # Different initial guess
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_stormer_3d_inverse import Stormer3DInversePINN, Stormer3DInverseTrainer


def train(dataset_path, output_dir, alpha1_init=5000.0,
          n_hidden=4, n_neurons=128, n_frequencies=10,
          warmup_epochs=5000, adam_epochs=20000,
          adam_lr=1e-3, alpha1_lr=1e-2,
          lbfgs_epochs=5,
          w_data=1.0, w_ode=1.0, w_energy=0.0,
          obs_fraction=1.0, noise_std=0.0, device="cpu"):
    """Full training pipeline: Warmup + Adam + L-BFGS."""

    os.makedirs(output_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)

    print(f"\n{'='*60}")
    print("Inverse Störmer PINN — 3D Case")
    print(f"{'='*60}")
    print(f"  alpha1_init = {alpha1_init}")
    print(f"  obs_fraction = {obs_fraction}, noise_std = {noise_std}")
    print(f"  Weights: w_data={w_data}, w_ode={w_ode}, w_energy={w_energy}")

    model = Stormer3DInversePINN(
        mode="3d",
        n_hidden=n_hidden, n_neurons=n_neurons,
        n_frequencies=n_frequencies, alpha1_init=alpha1_init,
    )
    trainer = Stormer3DInverseTrainer(
        model, dataset_path, device=device,
        w_data=w_data, w_ode=w_ode, w_energy=w_energy,
        obs_fraction=obs_fraction, noise_std=noise_std,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,} (including alpha1)")

    history = {
        "epoch": [], "total": [], "data": [], "ode": [],
        "energy": [], "alpha1": [],
    }

    def _record(epoch, total, l_data, l_ode, l_energy, verbose=False):
        a1_est, a1_true, a1_err = trainer.get_alpha1_error()
        history["epoch"].append(epoch)
        history["total"].append(total.item())
        history["data"].append(l_data.item())
        history["ode"].append(l_ode.item())
        history["energy"].append(
            l_energy.item() if isinstance(l_energy, torch.Tensor) else 0.0)
        history["alpha1"].append(a1_est)
        if verbose:
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch:6d} | Total: {total.item():.4e} | "
                f"Data: {l_data.item():.4e} | ODE: {l_ode.item():.4e} | "
                f"a1={a1_est:.4f} (err={a1_err:.4e}) | t={elapsed:.1f}s"
            )

    t_start = time.time()

    # ------------------------------------------------------------------
    # Phase 0: Warmup — data-only, alpha1 frozen
    # ------------------------------------------------------------------
    if warmup_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 0: Warmup ({warmup_epochs} epochs, data only, alpha1 frozen)")
        print(f"{'='*60}")

        model.log_alpha1.requires_grad_(False)
        net_params = [p for name, p in model.named_parameters()
                      if name != "log_alpha1"]
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

        model.log_alpha1.requires_grad_(True)
        print(f"  Warmup done. Data loss: {l_data.item():.4e}")

    # ------------------------------------------------------------------
    # Phase 1: Adam — train network + alpha1 jointly
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 1: Adam ({adam_epochs} epochs, "
          f"net_lr={adam_lr}, alpha1_lr={alpha1_lr})")
    print(f"{'='*60}")

    net_params = [p for name, p in model.named_parameters()
                  if name != "log_alpha1"]
    optimizer = torch.optim.Adam([
        {"params": net_params, "lr": adam_lr},
        {"params": [model.log_alpha1], "lr": alpha1_lr},
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
        torch.nn.utils.clip_grad_norm_(net_params, max_norm=1.0)
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
            a1_est = model.get_alpha1().item()
            print(f"  L-BFGS step {lbfgs_step[0]:4d} | "
                  f"Total: {total.item():.4e} | alpha1={a1_est:.4f}")
        return total

    for _ in range(lbfgs_epochs):
        lbfgs.step(closure)

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    total, l_data, l_ode, l_energy = trainer.total_loss()
    a1_est, a1_true, a1_err = trainer.get_alpha1_error()

    _record(epoch_offset + adam_epochs + lbfgs_step[0],
            total, l_data, l_ode, l_energy, verbose=False)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"  Total loss:      {total.item():.4e}")
    print(f"  Data loss:       {l_data.item():.4e}")
    print(f"  ODE loss:        {l_ode.item():.4e}")
    print(f"  alpha1_estimated = {a1_est:.6f}")
    print(f"  alpha1_true      = {a1_true:.6f}")
    print(f"  alpha1_rel_error = {a1_err:.6e}")
    print(f"  Training time:   {time.time() - t_start:.1f}s")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(output_dir, "pinn_inverse_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "alpha1_estimated": a1_est,
        "alpha1_true": a1_true,
        "alpha1_rel_error": a1_err,
        "history": history,
        "config": {
            "mode": "3d",
            "alpha1_init": alpha1_init,
            "n_hidden": n_hidden, "n_neurons": n_neurons,
            "n_frequencies": n_frequencies,
            "warmup_epochs": warmup_epochs,
            "adam_epochs": adam_epochs, "adam_lr": adam_lr,
            "alpha1_lr": alpha1_lr,
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
    """Generate all result plots for 3D case."""

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
    ax.set_title("Training loss history (Inverse PINN — 3D)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=200)
    plt.close()

    # 2. alpha1 convergence
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(history["epoch"], history["alpha1"], "b-", linewidth=1.5)
    axes[0].axhline(trainer.alpha1_true, color="r", linestyle="--",
                    label=f"$\\alpha_1^{{true}}$ = {trainer.alpha1_true}")
    axes[0].set_ylabel(r"$\alpha_1$ estimated")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(r"Parameter $\alpha_1$ convergence")

    a1_err = [abs(ai - trainer.alpha1_true) / trainer.alpha1_true
              for ai in history["alpha1"]]
    axes[1].semilogy(history["epoch"], a1_err, "b-", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(
        r"$|\alpha_{1,est} - \alpha_{1,true}| / \alpha_{1,true}$")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alpha1_convergence.png"), dpi=200)
    plt.close()

    # 3. PINN vs true trajectory
    pred = trainer.predict(trainer.t_ref)
    rho_pred = pred["rho"]
    Z_pred = pred["Z"]
    phi_pred = pred["phi"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(trainer.t_ref, trainer.rho_ref, "b-",
                 linewidth=1, label="Solver")
    axes[0].plot(trainer.t_ref, rho_pred, "r--",
                 linewidth=1, label="PINN")
    axes[0].set_ylabel(r"$\rho$ [$r_0$]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Inverse PINN: Reconstructed trajectory (3D)")

    axes[1].plot(trainer.t_ref, trainer.Z_ref, "b-",
                 linewidth=1, label="Solver")
    axes[1].plot(trainer.t_ref, Z_pred, "r--",
                 linewidth=1, label="PINN")
    axes[1].set_ylabel(r"$Z$ [$r_0$]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(trainer.t_ref, trainer.phi_ref, "b-",
                 linewidth=1, label="Solver")
    axes[2].plot(trainer.t_ref, phi_pred, "r--",
                 linewidth=1, label="PINN")
    axes[2].set_ylabel(r"$\varphi$ [rad]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    rho_err = np.abs(rho_pred - trainer.rho_ref)
    Z_err = np.abs(Z_pred - trainer.Z_ref)
    phi_err = np.abs(phi_pred - trainer.phi_ref)
    axes[3].semilogy(trainer.t_ref, rho_err, "b-",
                     linewidth=0.8, label=r"$|\Delta\rho|$")
    axes[3].semilogy(trainer.t_ref, Z_err, "g-",
                     linewidth=0.8, label=r"$|\Delta Z|$")
    axes[3].semilogy(trainer.t_ref, phi_err, "r-",
                     linewidth=0.8, label=r"$|\Delta\varphi|$")
    axes[3].set_ylabel("Absolute error")
    axes[3].set_xlabel("$t$ [s]")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pinn_vs_solver.png"), dpi=200)
    plt.close()

    # 4. 2D trajectory (x, y) — projection onto equatorial plane
    x_ref = trainer.rho_ref * np.cos(trainer.phi_ref)
    y_ref = trainer.rho_ref * np.sin(trainer.phi_ref)
    x_pred = rho_pred * np.cos(phi_pred)
    y_pred = rho_pred * np.sin(phi_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_ref, y_ref, "b-", linewidth=0.5, label="Solver", alpha=0.7)
    ax.plot(x_pred, y_pred, "r--", linewidth=0.5, label="PINN", alpha=0.7)
    ax.set_xlabel("$x$ [$r_0$]")
    ax.set_ylabel("$y$ [$r_0$]")
    ax.set_title("2D trajectory (equatorial projection)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_2d.png"), dpi=200)
    plt.close()

    # 5. Energy conservation
    alpha1 = trainer.model.get_alpha1().item()
    c2_val, E0_val, tau_scale_val, _ = trainer._compute_params()
    c2_val = c2_val.item()
    E0_val = E0_val.item()

    R_pred = np.sqrt(rho_pred**2 + Z_pred**2)
    drho_pred = np.gradient(rho_pred, trainer.t_ref)
    dZ_pred = np.gradient(Z_pred, trainer.t_ref)
    H_eff = (0.5 * (drho_pred**2 + dZ_pred**2)
             + c2_val**2 / (2 * trainer.M**2 * rho_pred**2)
             - alpha1 * c2_val / (trainer.M * R_pred**3)
             + alpha1**2 * rho_pred**2 / (2 * R_pred**6))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trainer.t_ref, H_eff, "b-", linewidth=0.8)
    ax.axhline(E0_val, color="r", linestyle="--", alpha=0.7,
               label=f"$H_{{eff,0}}$ = {E0_val:.2f}")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$H_{eff}$")
    ax.set_title("Energy conservation along PINN trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_conservation.png"), dpi=200)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train inverse Störmer PINN — 3D case"
    )
    parser.add_argument("--proton", type=int, default=1,
                        help="Proton number (default: 1)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of observations (default: 1.0)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussian noise std on observations (default: 0.0)")
    parser.add_argument("--alpha1-init", type=float, default=5000.0,
                        help="Initial guess for alpha1 (default: 5000, "
                             "true=3037)")
    parser.add_argument("--warmup-epochs", type=int, default=5000,
                        help="Warmup epochs (default: 5000)")
    parser.add_argument("--adam-epochs", type=int, default=20000,
                        help="Adam epochs (default: 20000)")
    parser.add_argument("--adam-lr", type=float, default=1e-3,
                        help="Network learning rate (default: 0.001)")
    parser.add_argument("--alpha1-lr", type=float, default=1e-2,
                        help="alpha1 learning rate (default: 0.01)")
    parser.add_argument("--w-data", type=float, default=1.0,
                        help="Weight for data loss (default: 1.0)")
    parser.add_argument("--w-ode", type=float, default=1.0,
                        help="Weight for ODE loss (default: 1.0)")
    parser.add_argument("--w-energy", type=float, default=0.0,
                        help="Weight for energy loss (default: 0.0)")
    args = parser.parse_args()

    # Determine experiment name
    if args.noise > 0:
        exp_name = f"expC_noise{args.noise}"
    elif args.fraction < 1.0:
        exp_name = f"expB_{int(args.fraction * 100)}pct"
    else:
        exp_name = "expA"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    proton_dir = f"proton{args.proton}"
    dataset = os.path.join(
        script_dir, "data", f"dataset_3d_{proton_dir}.npz"
    )
    results = os.path.join(
        script_dir, "results", "3d", proton_dir, exp_name
    )

    train(
        dataset_path=dataset,
        output_dir=results,
        alpha1_init=args.alpha1_init,
        warmup_epochs=args.warmup_epochs,
        adam_epochs=args.adam_epochs,
        adam_lr=args.adam_lr,
        alpha1_lr=args.alpha1_lr,
        w_data=args.w_data,
        w_ode=args.w_ode,
        w_energy=args.w_energy,
        obs_fraction=args.fraction,
        noise_std=args.noise,
    )
