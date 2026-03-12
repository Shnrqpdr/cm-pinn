"""
Parametric Physics-Informed Neural Network for the Störmer problem on a sphere.

Learns a family of solutions parameterized by initial conditions.

Network input:  (tau_norm, z0, w0, a, b)
  - tau_norm: normalized dimensionless time in [0, 1]
  - z0, w0: initial conditions (z = cos(theta), w = dz/dtau)
  - a, b: dimensionless parameters (a = p_phi/L, b = k/L)

Network output: (z, w, delta_phi)
  - delta_phi = phi - phi0 (phi0 is added post-hoc)

ODE system (all dimensionless, O(1) coefficients):
  dz/dtau   = w
  dw/dtau   = b^2 * z * (A + B - 2*z^2)
  dphi/dtau = a / (1 - z^2) + b

where A, B are derived from a, b:
  disc = 1 - 4*a*b
  A = (2*b*(a+b) - 1 - sqrt(disc)) / (2*b^2)
  B = (2*b*(a+b) - 1 + sqrt(disc)) / (2*b^2)

Energy integral: w^2 = b^2 * (B - z^2) * (z^2 - A)

Based on Piña & Cortés, Eur. J. Phys. 37 (2016) 065009.
"""

import torch
import torch.nn as nn
import numpy as np

from pinn_stormer import FourierFeatures


class ParametricStormerPINN(nn.Module):
    """Parametric MLP for the Störmer problem (z-formulation).

    Input: (tau_norm, params) where params = (z0, w0, a, b)
    Output: (z, w, delta_phi)
    """

    def __init__(self, n_hidden=5, n_neurons=256, n_frequencies=10):
        super().__init__()
        self.fourier = FourierFeatures(n_frequencies=n_frequencies)
        # Fourier features on tau_norm (21) + 4 parameters (z0, w0, a, b)
        input_dim = self.fourier.output_dim + 4

        layers = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))  # z, w, delta_phi
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tau_norm, params):
        """
        tau_norm: (N, 1) — normalized time, requires_grad for autograd
        params:   (N, 4) — [z0, w0, a, b] per sample
        """
        features = self.fourier(tau_norm)
        x = torch.cat([features, params], dim=1)
        return self.net(x)


class ParametricStormerTrainer:
    """Training handler for the parametric Störmer PINN."""

    def __init__(self, model, dataset_path, device="cpu",
                 w_ode=1.0, w_ic=100.0, w_energy=10.0):
        self.model = model.to(device)
        self.device = device
        self.w_ode = w_ode
        self.w_ic = w_ic
        self.w_energy = w_energy

        data = np.load(dataset_path, allow_pickle=True)

        n_train = int(data["n_ics_train"])
        n_val = int(data["n_ics_val"])
        n_coll_per_ic = int(data["n_coll_per_ic"])
        n_ref = int(data["n_ref_per_ic"])
        self.n_train = n_train
        self.n_val = n_val

        # --- Per-IC training arrays (n_train,) ---
        z0_all = data["train_z0"]
        w0_all = data["train_w0"]
        a_all = data["train_a"]
        b_all = data["train_b"]
        A_all = data["train_A"]
        B_all = data["train_B"]
        tau_final_all = data["train_tau_final"]
        tau_scale_all = data["train_tau_scale"]
        self.train_labels = list(data["train_labels"])
        self.train_phi0 = data["train_phi0"]
        self.train_tau_scale = tau_scale_all
        self.train_tau_final = tau_final_all

        # --- Collocation data (flattened) ---
        tau_norm_coll = data["tau_norm_coll"]       # (n_train * n_coll,)
        ic_idx_coll = data["ic_idx_coll"].astype(int)  # (n_train * n_coll,)

        # Expand per-IC params to per-collocation-point
        z0_coll = z0_all[ic_idx_coll]
        w0_coll = w0_all[ic_idx_coll]
        a_coll = a_all[ic_idx_coll]
        b_coll = b_all[ic_idx_coll]
        A_coll = A_all[ic_idx_coll]
        B_coll = B_all[ic_idx_coll]
        tau_final_coll = tau_final_all[ic_idx_coll]

        # Collocation tensors
        self.tau_coll = torch.tensor(
            tau_norm_coll, dtype=torch.float64, device=device
        ).unsqueeze(1).requires_grad_(True)

        self.params_coll = torch.tensor(
            np.stack([z0_coll, w0_coll, a_coll, b_coll], axis=1),
            dtype=torch.float64, device=device
        )
        self.a_coll = torch.tensor(a_coll, dtype=torch.float64, device=device).unsqueeze(1)
        self.b_coll = torch.tensor(b_coll, dtype=torch.float64, device=device).unsqueeze(1)
        self.A_coll = torch.tensor(A_coll, dtype=torch.float64, device=device).unsqueeze(1)
        self.B_coll = torch.tensor(B_coll, dtype=torch.float64, device=device).unsqueeze(1)
        self.tau_final_coll = torch.tensor(
            tau_final_coll, dtype=torch.float64, device=device
        ).unsqueeze(1)

        # --- IC points (one per training IC) ---
        self.tau_ic = torch.zeros(
            n_train, 1, dtype=torch.float64, device=device, requires_grad=True
        )
        self.params_ic = torch.tensor(
            np.stack([z0_all, w0_all, a_all, b_all], axis=1),
            dtype=torch.float64, device=device
        )
        self.z0_ic = torch.tensor(z0_all, dtype=torch.float64, device=device)
        self.w0_ic = torch.tensor(w0_all, dtype=torch.float64, device=device)

        # --- Reference data for training ICs (plotting) ---
        self.train_t_ref = data["train_t_ref"]         # (n_train, n_ref)
        self.train_theta_ref = data["train_theta_ref"]  # (n_train, n_ref)
        self.train_phi_ref = data["train_phi_ref"]      # (n_train, n_ref)

        # --- Validation IC data ---
        self.val_z0 = data["val_z0"]
        self.val_w0 = data["val_w0"]
        self.val_a = data["val_a"]
        self.val_b = data["val_b"]
        self.val_A = data["val_A"]
        self.val_B = data["val_B"]
        self.val_phi0 = data["val_phi0"]
        self.val_tau_final = data["val_tau_final"]
        self.val_tau_scale = data["val_tau_scale"]
        self.val_labels = list(data["val_labels"])
        self.val_t_ref = data["val_t_ref"]           # (n_val, n_ref)
        self.val_theta_ref = data["val_theta_ref"]
        self.val_phi_ref = data["val_phi_ref"]

        # IC index per collocation point (for per-IC loss averaging)
        self.ic_idx_coll = torch.tensor(ic_idx_coll, dtype=torch.long, device=device)
        self.n_coll_per_ic = n_coll_per_ic

        print(f"  Training ICs: {n_train}, Validation ICs: {n_val}")
        print(f"  Collocation per IC: {n_coll_per_ic}, total: {len(tau_norm_coll)}")
        for i in range(n_train):
            print(f"    Train {i}: {self.train_labels[i]}  "
                  f"a={a_all[i]:.4f} b={b_all[i]:.4f} A={A_all[i]:.4f} B={B_all[i]:.4f}")

    def _compute_derivatives(self, tau_norm, params, tau_final):
        """Compute outputs and d/dtau via autograd, with per-sample tau_final scaling."""
        out = self.model(tau_norm, params)
        z = out[:, 0:1]
        w = out[:, 1:2]
        dphi = out[:, 2:3]

        dz_dn = torch.autograd.grad(z, tau_norm, torch.ones_like(z),
                                     create_graph=True, retain_graph=True)[0]
        dw_dn = torch.autograd.grad(w, tau_norm, torch.ones_like(w),
                                     create_graph=True, retain_graph=True)[0]
        ddphi_dn = torch.autograd.grad(dphi, tau_norm, torch.ones_like(dphi),
                                        create_graph=True, retain_graph=True)[0]

        # d/dtau = d/d(tau_norm) / tau_final  (element-wise, per sample)
        dz_dtau = dz_dn / tau_final
        dw_dtau = dw_dn / tau_final
        ddphi_dtau = ddphi_dn / tau_final

        return z, w, dphi, dz_dtau, dw_dtau, ddphi_dtau

    def _per_ic_mean(self, loss_per_point):
        """Average loss per IC first, then average across ICs.

        This prevents ICs with large parameter values (high b) from
        dominating the total loss.
        """
        # loss_per_point: (N_total, 1)
        loss_flat = loss_per_point.squeeze(1)
        # Reshape to (n_train, n_coll_per_ic) — points are stored contiguously per IC
        loss_per_ic = loss_flat.view(self.n_train, self.n_coll_per_ic).mean(dim=1)
        return loss_per_ic.mean()

    def loss_ode(self):
        """ODE residual loss with per-sample parameters, averaged per IC."""
        z, w, dphi, dz_dtau, dw_dtau, ddphi_dtau = \
            self._compute_derivatives(self.tau_coll, self.params_coll, self.tau_final_coll)

        a = self.a_coll
        b = self.b_coll
        A = self.A_coll
        B = self.B_coll

        r1 = dz_dtau - w
        r2 = dw_dtau - b**2 * z * (A + B - 2 * z**2)
        r3 = ddphi_dtau - (a / (1 - z**2 + 1e-8) + b)

        return self._per_ic_mean(r1**2 + r2**2 + r3**2)

    def loss_ic(self):
        """Initial conditions loss (one evaluation per training IC)."""
        out = self.model(self.tau_ic, self.params_ic)
        z_pred = out[:, 0]
        w_pred = out[:, 1]
        dphi_pred = out[:, 2]

        return torch.mean(
            (z_pred - self.z0_ic)**2
            + (w_pred - self.w0_ic)**2
            + dphi_pred**2   # delta_phi(0) = 0 always
        )

    def loss_energy(self):
        """Energy integral: w^2 = b^2*(B-z^2)*(z^2-A), averaged per IC."""
        out = self.model(self.tau_coll, self.params_coll)
        z = out[:, 0:1]
        w = out[:, 1:2]

        lhs = w**2
        rhs = self.b_coll**2 * (self.B_coll - z**2) * (z**2 - self.A_coll)

        return self._per_ic_mean((lhs - rhs)**2)

    def total_loss(self):
        l_ode = self.loss_ode()
        l_ic = self.loss_ic()
        l_energy = self.loss_energy()
        total = self.w_ode * l_ode + self.w_ic * l_ic + self.w_energy * l_energy
        return total, l_ode, l_ic, l_energy

    def predict(self, t_physical, z0, w0, phi0, a, b, tau_scale, tau_final):
        """Predict trajectory for a single IC given physical time array."""
        self.model.eval()
        tau = t_physical / tau_scale
        tau_norm = tau / tau_final

        N = len(t_physical)
        tau_t = torch.tensor(
            tau_norm, dtype=torch.float64, device=self.device
        ).unsqueeze(1)
        params = torch.tensor(
            np.tile([z0, w0, a, b], (N, 1)),
            dtype=torch.float64, device=self.device
        )

        with torch.no_grad():
            out = self.model(tau_t, params)

        z = out[:, 0].cpu().numpy()
        w = out[:, 1].cpu().numpy()
        delta_phi = out[:, 2].cpu().numpy()

        theta = np.arccos(np.clip(z, -1, 1))
        phi = delta_phi + phi0

        return z, w, theta, phi

    def validate(self):
        """Validate on held-out ICs (generalization test)."""
        results = []
        for i in range(self.n_val):
            z_pred, w_pred, theta_pred, phi_pred = self.predict(
                self.val_t_ref[i],
                self.val_z0[i], self.val_w0[i], self.val_phi0[i],
                self.val_a[i], self.val_b[i],
                self.val_tau_scale[i], self.val_tau_final[i],
            )

            theta_ref = self.val_theta_ref[i]
            phi_ref = self.val_phi_ref[i]

            theta_err = np.abs(theta_pred - theta_ref)
            phi_err = np.abs(phi_pred - phi_ref)

            phi_denom = max(np.sqrt(np.mean(phi_ref**2)), 1e-10)

            results.append({
                "label": self.val_labels[i],
                "theta_mae": theta_err.mean(),
                "theta_max": theta_err.max(),
                "phi_mae": phi_err.mean(),
                "phi_max": phi_err.max(),
                "theta_rel_l2": (np.sqrt(np.mean((theta_pred - theta_ref)**2))
                                 / np.sqrt(np.mean(theta_ref**2))),
                "phi_rel_l2": (np.sqrt(np.mean((phi_pred - phi_ref)**2))
                               / phi_denom),
            })

        return results

    def validate_train(self):
        """Validate on training ICs (fitting quality)."""
        results = []
        for i in range(self.n_train):
            z_pred, w_pred, theta_pred, phi_pred = self.predict(
                self.train_t_ref[i],
                float(self.params_ic[i, 0]), float(self.params_ic[i, 1]),
                self.train_phi0[i],
                float(self.params_ic[i, 2]), float(self.params_ic[i, 3]),
                self.train_tau_scale[i], self.train_tau_final[i],
            )

            theta_ref = self.train_theta_ref[i]
            phi_ref = self.train_phi_ref[i]

            theta_err = np.abs(theta_pred - theta_ref)
            phi_err = np.abs(phi_pred - phi_ref)

            phi_denom = max(np.sqrt(np.mean(phi_ref**2)), 1e-10)

            results.append({
                "label": self.train_labels[i],
                "theta_mae": theta_err.mean(),
                "theta_max": theta_err.max(),
                "phi_mae": phi_err.mean(),
                "phi_max": phi_err.max(),
                "theta_rel_l2": (np.sqrt(np.mean((theta_pred - theta_ref)**2))
                                 / np.sqrt(np.mean(theta_ref**2))),
                "phi_rel_l2": (np.sqrt(np.mean((phi_pred - phi_ref)**2))
                               / phi_denom),
            })

        return results
