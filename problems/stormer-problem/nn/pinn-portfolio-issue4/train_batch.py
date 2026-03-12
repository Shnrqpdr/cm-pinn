"""
Batch training script for the Störmer sphere PINN portfolio.

Trains one independent PINN per paper case, then generates a combined summary.
Each PINN uses the validated architecture from Case 1 (z-formulation, Fourier features).

Usage:
    python train_batch.py                   # Train all cases
    python train_batch.py fig6a fig7c       # Train specific cases only
"""

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pinn_stormer import StormerPINN, StormerPINNTrainer

from generate_datasets import PAPER_CASES


# Default hyperparameters (validated on Case 1)
DEFAULTS = {
    "n_hidden": 4,
    "n_neurons": 128,
    "n_frequencies": 10,
    "adam_epochs": 20000,
    "adam_lr": 1e-3,
    "lbfgs_epochs": 5,
    "w_ode": 1.0,
    "w_ic": 100.0,
    "w_energy": 10.0,
}


def train_single(case_name, dataset_path, output_dir, hparams=None, device="cpu"):
    """Train a single PINN for one case. Returns metrics dict."""
    hp = {**DEFAULTS, **(hparams or {})}
    os.makedirs(output_dir, exist_ok=True)

    torch.set_default_dtype(torch.float64)

    model = StormerPINN(
        n_hidden=hp["n_hidden"],
        n_neurons=hp["n_neurons"],
        n_frequencies=hp["n_frequencies"],
    )
    trainer = StormerPINNTrainer(
        model, dataset_path, device=device,
        w_ode=hp["w_ode"], w_ic=hp["w_ic"], w_energy=hp["w_energy"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    history = {"epoch": [], "total": [], "ode": [], "ic": [], "energy": []}

    # Phase 1: Adam
    print(f"  Phase 1: Adam ({hp['adam_epochs']} epochs)")
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["adam_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["adam_epochs"], eta_min=1e-6)

    t_start = time.time()
    for epoch in range(1, hp["adam_epochs"] + 1):
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

        if epoch % 5000 == 0:
            elapsed = time.time() - t_start
            print(f"    Epoch {epoch:6d} | Total: {total.item():.4e} | "
                  f"ODE: {l_ode.item():.4e} | IC: {l_ic.item():.4e} | "
                  f"Energy: {l_energy.item():.4e} | t: {elapsed:.1f}s")

    adam_time = time.time() - t_start

    # Phase 2: L-BFGS
    print(f"  Phase 2: L-BFGS ({hp['lbfgs_epochs']} steps)")
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
        return total

    t_lbfgs_start = time.time()
    for _ in range(hp["lbfgs_epochs"]):
        lbfgs.step(closure)
    lbfgs_time = time.time() - t_lbfgs_start

    # Final loss
    total, l_ode, l_ic, l_energy = trainer.total_loss()
    history["epoch"].append(hp["adam_epochs"] + lbfgs_step[0])
    history["total"].append(total.item())
    history["ode"].append(l_ode.item())
    history["ic"].append(l_ic.item())
    history["energy"].append(l_energy.item())

    print(f"  Final: Total={total.item():.4e} | ODE={l_ode.item():.4e} | "
          f"IC={l_ic.item():.4e} | Energy={l_energy.item():.4e}")

    # Validation
    val_metrics = trainer.validate()
    for key, val in val_metrics.items():
        print(f"    {key}: {val:.6e}")

    # Save model
    model_path = os.path.join(output_dir, "pinn_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "hparams": hp,
        "tau_final": trainer.tau_final,
        "T_final": trainer.T_final,
        "history": history,
        "val_metrics": val_metrics,
    }, model_path)

    # Plots
    _plot_case_results(trainer, history, output_dir, case_name)

    total_time = adam_time + lbfgs_time

    return {
        "case": case_name,
        "label": PAPER_CASES[case_name]["label"],
        "final_loss": total.item(),
        "loss_ode": l_ode.item(),
        "loss_ic": l_ic.item(),
        "loss_energy": l_energy.item(),
        "val_metrics": val_metrics,
        "n_params": n_params,
        "train_time_s": total_time,
    }


def _plot_case_results(trainer, history, output_dir, case_name):
    """Generate per-case plots."""
    label = PAPER_CASES[case_name]["label"]

    # Loss history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(history["epoch"], history["total"], "k-", label="Total", linewidth=1.5)
    ax.semilogy(history["epoch"], history["ode"], "b--", label="ODE", linewidth=1)
    ax.semilogy(history["epoch"], history["ic"], "r--", label="IC", linewidth=1)
    ax.semilogy(history["epoch"], history["energy"], "g--", label="Energy", linewidth=1)
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"{label} — Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=200)
    plt.close()

    # PINN vs Analytical
    z_pred, w_pred, theta_pred, phi_pred = trainer.predict(trainer.t_ref)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(trainer.t_ref, np.degrees(trainer.theta_ref), "b-", linewidth=1, label="Analytical")
    axes[0, 0].plot(trainer.t_ref, np.degrees(theta_pred), "r--", linewidth=1, label="PINN")
    axes[0, 0].set_ylabel(r"$\theta$ (deg)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title(f"{label}")

    axes[0, 1].plot(trainer.t_ref, trainer.phi_ref, "b-", linewidth=1, label="Analytical")
    axes[0, 1].plot(trainer.t_ref, phi_pred, "r--", linewidth=1, label="PINN")
    axes[0, 1].set_ylabel(r"$\varphi$ (rad)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    theta_err = np.abs(theta_pred - trainer.theta_ref)
    phi_err = np.abs(phi_pred - trainer.phi_ref)

    axes[1, 0].semilogy(trainer.t_ref, theta_err, "b-", linewidth=0.8)
    axes[1, 0].set_ylabel(r"$|\Delta\theta|$ (rad)")
    axes[1, 0].set_xlabel("$t$")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Absolute errors")

    axes[1, 1].semilogy(trainer.t_ref, phi_err, "r-", linewidth=0.8)
    axes[1, 1].set_ylabel(r"$|\Delta\varphi|$ (rad)")
    axes[1, 1].set_xlabel("$t$")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pinn_vs_analytical.png"), dpi=200)
    plt.close()

    # Energy conservation
    w2 = w_pred**2
    rhs = trainer.b**2 * (trainer.B - z_pred**2) * (z_pred**2 - trainer.A)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trainer.t_ref, np.abs(w2 - rhs), "b-", linewidth=0.8)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|w^2 - b^2(B-z^2)(z^2-A)|$")
    ax.set_title(f"{label} — Energy integral residual")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_conservation.png"), dpi=200)
    plt.close()


def plot_summary(all_results, summary_dir):
    """Generate combined summary plots across all cases."""
    os.makedirs(summary_dir, exist_ok=True)

    cases = [r["case"] for r in all_results]
    labels = [r["label"] for r in all_results]
    theta_maes = [r["val_metrics"]["theta_mae"] for r in all_results]
    phi_maes = [r["val_metrics"]["phi_mae"] for r in all_results]
    theta_maxs = [r["val_metrics"]["theta_max"] for r in all_results]
    phi_maxs = [r["val_metrics"]["phi_max"] for r in all_results]
    total_losses = [r["final_loss"] for r in all_results]
    train_times = [r["train_time_s"] for r in all_results]

    x = np.arange(len(cases))
    width = 0.35

    # MAE comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, theta_maes, width, label=r"$\theta$ MAE", color="steelblue")
    bars2 = ax.bar(x + width/2, phi_maes, width, label=r"$\varphi$ MAE", color="coral")
    ax.set_yscale("log")
    ax.set_ylabel("MAE (rad)")
    ax.set_title("Portfolio PINN — Validation MAE per case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "portfolio_mae_comparison.png"), dpi=200)
    plt.close()

    # Max error comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, theta_maxs, width, label=r"$\theta$ max error", color="steelblue")
    bars2 = ax.bar(x + width/2, phi_maxs, width, label=r"$\varphi$ max error", color="coral")
    ax.set_yscale("log")
    ax.set_ylabel("Max absolute error (rad)")
    ax.set_title("Portfolio PINN — Max error per case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "portfolio_maxerr_comparison.png"), dpi=200)
    plt.close()

    # Final loss comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, total_losses, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("Final total loss")
    ax.set_title("Portfolio PINN — Final loss per case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "portfolio_loss_comparison.png"), dpi=200)
    plt.close()

    # Print summary table
    print("\n" + "=" * 90)
    print("PORTFOLIO SUMMARY")
    print("=" * 90)
    print(f"{'Case':<10} {'Label':<35} {'Total Loss':>12} {'theta MAE':>12} "
          f"{'phi MAE':>12} {'Time (s)':>10}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['case']:<10} {r['label']:<35} {r['final_loss']:>12.4e} "
              f"{r['val_metrics']['theta_mae']:>12.4e} "
              f"{r['val_metrics']['phi_mae']:>12.4e} "
              f"{r['train_time_s']:>10.1f}")
    print("=" * 90)

    total_time = sum(r["train_time_s"] for r in all_results)
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results as JSON
    results_json = []
    for r in all_results:
        rj = {**r}
        rj["val_metrics"] = {k: float(v) for k, v in r["val_metrics"].items()}
        results_json.append(rj)

    with open(os.path.join(summary_dir, "portfolio_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {summary_dir}/portfolio_results.json")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    results_base = os.path.join(base_dir, "results")
    summary_dir = os.path.join(results_base, "summary")

    # Parse optional case names from command line
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        for s in selected:
            if s not in PAPER_CASES:
                print(f"Unknown case '{s}'. Available: {list(PAPER_CASES.keys())}")
                sys.exit(1)
    else:
        selected = list(PAPER_CASES.keys())

    print("=" * 60)
    print(f"PINN Portfolio Training — {len(selected)} cases")
    print("=" * 60)
    print(f"Cases: {selected}")
    print(f"Hyperparameters: {DEFAULTS}")

    all_results = []
    for case_name in selected:
        print(f"\n{'='*60}")
        print(f"Training: {case_name} — {PAPER_CASES[case_name]['label']}")
        print(f"{'='*60}")

        dataset_path = os.path.join(data_dir, f"dataset_{case_name}.npz")
        if not os.path.exists(dataset_path):
            print(f"  Dataset not found: {dataset_path}")
            print(f"  Run generate_datasets.py first!")
            sys.exit(1)

        output_dir = os.path.join(results_base, case_name)
        result = train_single(case_name, dataset_path, output_dir)
        all_results.append(result)

    plot_summary(all_results, summary_dir)
