"""
Standalone PINN for the inverse Störmer problem on a sphere.

Identifies the magnetic coupling parameter k from observed trajectory data.
The network predicts (z, w, phi) and k is a trainable parameter.
All dimensionless parameters (a, b, A, B) are recomputed as differentiable
functions of k at each training step.

Completely independent from the forward PINN implementation.

ODE system (dimensionless):
  dz/dtau   = w
  dw/dtau   = b(k)^2 * z * (A(k) + B(k) - 2*z^2)
  dphi/dtau = a(k) / (1 - z^2) + b(k)

The network input is t_norm = t / T_obs (physical time normalized to [0,1]).
Chain rule: d/dtau = (tau_scale / T_obs) * d/dt_norm
         => d/dt_norm = S * f(z,w,phi)  where S = T_obs / tau_scale(k)

Based on Piña & Cortés, Eur. J. Phys. 37 (2016) 065009.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FourierFeatures(nn.Module):
    """Map scalar input to Fourier features: [t, sin(2*pi*f_i*t), cos(2*pi*f_i*t)]."""

    def __init__(self, n_frequencies=10):
        super().__init__()
        freqs = torch.linspace(1.0, float(n_frequencies), n_frequencies)
        self.register_buffer("freqs", freqs)
        self.output_dim = 1 + 2 * n_frequencies

    def forward(self, t):
        angles = 2 * math.pi * t * self.freqs
        return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=1)


class StormerInversePINN(nn.Module):
    """Neural network with trainable k for the inverse Störmer problem."""

    def __init__(self, n_hidden=4, n_neurons=128, n_frequencies=10, k_init=1.0):
        super().__init__()
        self.fourier = FourierFeatures(n_frequencies=n_frequencies)
        input_dim = self.fourier.output_dim

        layers = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))  # z, w, phi
        self.net = nn.Sequential(*layers)

        # Trainable k via softplus: k = softplus(k_raw) ensures k > 0
        k_raw_init = math.log(math.exp(k_init) - 1)  # inverse softplus
        self.k_raw = nn.Parameter(torch.tensor(k_raw_init, dtype=torch.float64))

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t_norm):
        features = self.fourier(t_norm)
        return self.net(features)

    def get_k(self):
        """Return current k value (always positive via softplus)."""
        return F.softplus(self.k_raw)


class StormerInverseTrainer:
    """Handles loss computation for the inverse Störmer PINN.

    Loss = w_data * L_data + w_ode * L_ode [+ w_energy * L_energy]

    L_data: MSE between network predictions and observed (z, phi) at data points.
    L_ode:  Residual of the 3 ODEs at collocation points, with k-dependent coefficients.
    L_energy: Energy integral w^2 = b^2*(B-z^2)*(z^2-A) (optional regularizer).
    """

    def __init__(self, model, dataset_path, device="cpu",
                 w_data=1.0, w_ode=1.0, w_energy=0.0,
                 obs_fraction=1.0, noise_std=0.0, seed=42):
        self.model = model.to(device)
        self.device = device
        self.w_data = w_data
        self.w_ode = w_ode
        self.w_energy = w_energy

        # Load dataset
        data = np.load(dataset_path)
        self.M = float(data["M"])
        self.R = float(data["R"])
        self.MR2 = self.M * self.R ** 2
        self.k_true = float(data["k_true"])

        # Known initial conditions and conserved quantities
        self.theta0 = float(data["theta0"])
        self.p_theta0 = float(data["p_theta0"])
        self.phi0 = float(data["phi0"])
        self.p_phi0 = float(data["p_phi0"])
        self.sin2_theta0 = math.sin(self.theta0) ** 2

        # Observation time window (fixed)
        self.T_obs = float(data["T_final"])

        # --- Observation data: apply sparsity and noise ---
        t_obs_full = data["t_obs"]
        theta_obs_full = data["theta_obs"]
        phi_obs_full = data["phi_obs"]

        rng = np.random.default_rng(seed)
        n_full = len(t_obs_full)

        if obs_fraction < 1.0:
            n_keep = max(2, int(n_full * obs_fraction))
            mid_indices = np.sort(
                rng.choice(np.arange(1, n_full - 1), n_keep - 2, replace=False)
            )
            indices = np.concatenate([[0], mid_indices, [n_full - 1]])
        else:
            indices = np.arange(n_full)

        t_obs = t_obs_full[indices]
        theta_obs = theta_obs_full[indices].copy()
        phi_obs = phi_obs_full[indices].copy()

        if noise_std > 0:
            theta_obs += rng.normal(0, noise_std, len(theta_obs))
            phi_obs += rng.normal(0, noise_std, len(phi_obs))

        # Convert to z = cos(theta), normalize time to [0, 1]
        z_obs = np.cos(theta_obs)
        t_obs_norm = t_obs / self.T_obs

        self.t_obs_norm = torch.tensor(
            t_obs_norm, dtype=torch.float64, device=device
        ).unsqueeze(1)
        self.z_obs = torch.tensor(z_obs, dtype=torch.float64, device=device)
        self.phi_obs = torch.tensor(phi_obs, dtype=torch.float64, device=device)

        # --- Collocation points (for ODE residual, no labels) ---
        t_coll = data["t_collocation"]
        t_coll_norm = t_coll / self.T_obs
        self.t_coll_norm = torch.tensor(
            t_coll_norm, dtype=torch.float64, device=device
        ).unsqueeze(1).requires_grad_(True)

        # --- Reference data (for plotting only) ---
        self.t_ref = data["t_reference"]
        self.theta_ref = data["theta_reference"]
        self.phi_ref = data["phi_reference"]

        print(f"  Observations: {len(t_obs)} points "
              f"(fraction={obs_fraction}, noise_std={noise_std})")
        print(f"  Collocation: {len(t_coll)} points")
        print(f"  T_obs = {self.T_obs:.2f}")
        print(f"  k_true = {self.k_true}, k_init = {model.get_k().item():.4f}")

    def _compute_params(self):
        """Recompute dimensionless parameters as differentiable functions of k.

        Returns (a, b, A, B, scale) where scale = T_obs / tau_scale(k).
        All outputs are torch tensors connected to k's computation graph.
        """
        k = self.model.get_k()

        # Energy at t=0: K = [p_theta0^2 + (p_phi0 + k*sin^2(theta0))^2 / sin^2(theta0)] / (2*M*R^2)
        eta0 = self.p_phi0 + k * self.sin2_theta0
        K = (self.p_theta0 ** 2 + eta0 ** 2 / self.sin2_theta0) / (2 * self.MR2)

        L = torch.sqrt(2 * self.MR2 * K)
        a = self.p_phi0 / L
        b = k / L

        # Roots of the quartic potential
        disc = 1 - 4 * a * b
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=1e-12))
        numer = 2 * b * (a + b) - 1
        A_val = (numer - sqrt_disc) / (2 * b ** 2)
        B_val = (numer + sqrt_disc) / (2 * b ** 2)

        # Time scale conversion: S = T_obs / tau_scale
        tau_scale = torch.sqrt(self.MR2 / (2 * K))
        scale = self.T_obs / tau_scale

        return a, b, A_val, B_val, scale

    def _compute_derivatives(self, t_norm):
        """Compute network outputs and d/dt_norm via autograd."""
        out = self.model(t_norm)
        z = out[:, 0:1]
        w = out[:, 1:2]
        phi = out[:, 2:3]

        ones_z = torch.ones_like(z)
        ones_w = torch.ones_like(w)
        ones_phi = torch.ones_like(phi)

        dz = torch.autograd.grad(z, t_norm, ones_z,
                                 create_graph=True, retain_graph=True)[0]
        dw = torch.autograd.grad(w, t_norm, ones_w,
                                 create_graph=True, retain_graph=True)[0]
        dphi = torch.autograd.grad(phi, t_norm, ones_phi,
                                   create_graph=True, retain_graph=True)[0]

        return z, w, phi, dz, dw, dphi

    def loss_data(self):
        """Data loss: MSE between predictions and observations in (z, phi)."""
        out = self.model(self.t_obs_norm)
        z_pred = out[:, 0]
        phi_pred = out[:, 2]

        return torch.mean((z_pred - self.z_obs) ** 2
                          + (phi_pred - self.phi_obs) ** 2)

    def loss_ode(self):
        """ODE residual loss with k-dependent coefficients.

        In t_norm coordinates:
          dz/dt_norm   = S * w
          dw/dt_norm   = S * b^2 * z * (A + B - 2*z^2)
          dphi/dt_norm = S * (a / (1 - z^2 + eps) + b)

        where S = T_obs / tau_scale(k).
        """
        a, b, A_val, B_val, scale = self._compute_params()
        z, w, phi, dz_dt, dw_dt, dphi_dt = \
            self._compute_derivatives(self.t_coll_norm)

        r1 = dz_dt - scale * w
        r2 = dw_dt - scale * b ** 2 * z * (A_val + B_val - 2 * z ** 2)
        r3 = dphi_dt - scale * (a / (1 - z ** 2 + 1e-8) + b)

        return torch.mean(r1 ** 2 + r2 ** 2 + r3 ** 2)

    def loss_energy(self):
        """Energy integral: w^2 = b^2 * (B - z^2) * (z^2 - A)."""
        _, b, A_val, B_val, _ = self._compute_params()
        out = self.model(self.t_coll_norm)
        z = out[:, 0:1]
        w = out[:, 1:2]

        lhs = w ** 2
        rhs = b ** 2 * (B_val - z ** 2) * (z ** 2 - A_val)

        return torch.mean((lhs - rhs) ** 2)

    def total_loss(self):
        """Compute total weighted loss."""
        l_data = self.loss_data()
        l_ode = self.loss_ode()
        l_energy = self.loss_energy() if self.w_energy > 0 else \
            torch.tensor(0.0, dtype=torch.float64, device=self.device)

        total = self.w_data * l_data + self.w_ode * l_ode \
            + self.w_energy * l_energy
        return total, l_data, l_ode, l_energy

    def predict(self, t_physical):
        """Predict z(t), w(t), theta(t), phi(t) given physical time array."""
        self.model.eval()
        t_norm = torch.tensor(
            t_physical / self.T_obs, dtype=torch.float64, device=self.device
        ).unsqueeze(1)
        with torch.no_grad():
            out = self.model(t_norm)
        z = out[:, 0].cpu().numpy()
        w = out[:, 1].cpu().numpy()
        phi = out[:, 2].cpu().numpy()
        theta = np.arccos(np.clip(z, -1, 1))
        return z, w, theta, phi

    def get_k_error(self):
        """Return (k_estimated, k_true, relative_error)."""
        k_est = self.model.get_k().item()
        rel_err = abs(k_est - self.k_true) / self.k_true
        return k_est, self.k_true, rel_err
