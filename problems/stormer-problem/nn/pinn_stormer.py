"""
Physics-Informed Neural Network for the Störmer problem on a sphere.

Uses the dimensionless z = cos(theta) formulation to avoid sin^3 singularity.

Network input:  tau_norm (dimensionless time, normalized to [0, 1])
Network output: z(tau), w(tau) = dz/dtau, phi(tau)

ODE system (all dimensionless, O(1) coefficients):
  dz/dtau   = w
  dw/dtau   = b^2 * z * (A + B - 2*z^2)
  dphi/dtau = a / (1 - z^2) + b

Energy integral: w^2 = b^2 * (B - z^2) * (z^2 - A)

Based on Piña & Cortés, Eur. J. Phys. 37 (2016) 065009.
"""

import torch
import torch.nn as nn
import numpy as np


class FourierFeatures(nn.Module):
    """Map scalar input to Fourier features: [t, sin(2*pi*f_i*t), cos(2*pi*f_i*t)]."""

    def __init__(self, n_frequencies=10, max_freq=None):
        super().__init__()
        if max_freq is None:
            max_freq = float(n_frequencies)
        freqs = torch.linspace(1.0, max_freq, n_frequencies)
        self.register_buffer("freqs", freqs)
        self.output_dim = 1 + 2 * n_frequencies

    def forward(self, t):
        angles = 2 * np.pi * t * self.freqs
        return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=1)


class StormerPINN(nn.Module):
    """Fourier-featured MLP for the Störmer problem (z-formulation)."""

    def __init__(self, n_hidden=4, n_neurons=128, n_frequencies=10):
        super().__init__()
        self.fourier = FourierFeatures(n_frequencies=n_frequencies)
        input_dim = self.fourier.output_dim

        layers = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))  # z, w, phi
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tau_norm):
        features = self.fourier(tau_norm)
        return self.net(features)


class StormerPINNTrainer:
    """Handles loss computation and training for the Störmer PINN (z-formulation)."""

    def __init__(self, model, dataset_path, device="cpu",
                 w_ode=1.0, w_ic=100.0, w_energy=10.0):
        self.model = model.to(device)
        self.device = device
        self.w_ode = w_ode
        self.w_ic = w_ic
        self.w_energy = w_energy

        # Load dataset
        data = np.load(dataset_path)
        self.M = float(data["M"])
        self.R = float(data["R"])
        self.k = float(data["k"])
        self.MR2 = self.M * self.R ** 2

        # Dimensionless parameters
        self.a = float(data["a"])
        self.b = float(data["b"])
        self.A = float(data["A"])
        self.B = float(data["B"])

        # Physical initial conditions
        theta0 = float(data["theta0"])
        p_theta0 = float(data["p_theta0"])
        self.phi0 = float(data["phi0"])

        # Energy
        K = float(data["energy"])
        L = np.sqrt(2 * self.MR2 * K)  # characteristic momentum

        # Dimensionless initial conditions
        self.z0 = np.cos(theta0)
        self.w0 = -np.sin(theta0) * p_theta0 / L  # dz/dtau at tau=0
        # phi0 stays the same

        # Time scales
        # tau = t * sqrt(2K/(MR^2)), so tau_final = T_final / tau_scale
        self.tau_scale = np.sqrt(self.MR2 / (2 * K))
        self.T_final = float(data["T_final"])
        self.tau_final = self.T_final / self.tau_scale

        # Collocation points (normalized to [0, 1] in tau space)
        t_coll = data["t_collocation"]
        tau_coll = t_coll / self.tau_scale
        self.tau_coll_norm = torch.tensor(
            tau_coll / self.tau_final, dtype=torch.float64, device=device
        ).unsqueeze(1).requires_grad_(True)

        # IC point
        self.tau_ic = torch.zeros(1, 1, dtype=torch.float64, device=device,
                                  requires_grad=True)

        # Validation data (converted to z)
        self.t_val = data["t_validation"]
        self.z_val = np.cos(data["theta_validation"])
        self.phi_val = data["phi_validation"]

        # Reference data (dense)
        self.t_ref = data["t_reference"]
        self.theta_ref = data["theta_reference"]
        self.z_ref = np.cos(data["theta_reference"])
        self.phi_ref = data["phi_reference"]

        print(f"  Dimensionless params: a={self.a:.6f}, b={self.b:.6f}")
        print(f"  Roots: A={self.A:.6f}, B={self.B:.6f}")
        print(f"  z0={self.z0:.6f}, w0={self.w0:.6f}")
        print(f"  tau_final={self.tau_final:.4f} (T_final={self.T_final:.2f})")

    def _compute_derivatives(self, tau_norm):
        """Compute outputs and d/dtau via autograd."""
        out = self.model(tau_norm)
        z = out[:, 0:1]
        w = out[:, 1:2]
        phi = out[:, 2:3]

        # d/d(tau_norm) then chain rule: d/dtau = d/d(tau_norm) / tau_final
        dz = torch.autograd.grad(z, tau_norm, torch.ones_like(z),
                                 create_graph=True, retain_graph=True)[0]
        dw = torch.autograd.grad(w, tau_norm, torch.ones_like(w),
                                 create_graph=True, retain_graph=True)[0]
        dphi = torch.autograd.grad(phi, tau_norm, torch.ones_like(phi),
                                   create_graph=True, retain_graph=True)[0]

        scale = 1.0 / self.tau_final
        return z, w, phi, dz * scale, dw * scale, dphi * scale

    def loss_ode(self):
        """ODE residual loss (dimensionless, O(1) coefficients).

        dz/dtau   = w
        dw/dtau   = b^2 * z * (A + B - 2*z^2)
        dphi/dtau = a / (1 - z^2) + b
        """
        z, w, phi, dz_dtau, dw_dtau, dphi_dtau = \
            self._compute_derivatives(self.tau_coll_norm)

        b, a = self.b, self.a
        A, B = self.A, self.B

        r1 = dz_dtau - w
        r2 = dw_dtau - b**2 * z * (A + B - 2 * z**2)
        r3 = dphi_dtau - (a / (1 - z**2 + 1e-8) + b)

        return torch.mean(r1**2 + r2**2 + r3**2)

    def loss_ic(self):
        """Initial conditions loss."""
        out = self.model(self.tau_ic)
        z_pred = out[:, 0]
        w_pred = out[:, 1]
        phi_pred = out[:, 2]

        return ((z_pred - self.z0)**2
                + (w_pred - self.w0)**2
                + (phi_pred - self.phi0)**2).mean()

    def loss_energy(self):
        """Energy integral: w^2 should equal b^2*(B-z^2)*(z^2-A)."""
        out = self.model(self.tau_coll_norm)
        z = out[:, 0:1]
        w = out[:, 1:2]

        lhs = w**2
        rhs = self.b**2 * (self.B - z**2) * (z**2 - self.A)

        return torch.mean((lhs - rhs)**2)

    def total_loss(self):
        l_ode = self.loss_ode()
        l_ic = self.loss_ic()
        l_energy = self.loss_energy()
        total = self.w_ode * l_ode + self.w_ic * l_ic + self.w_energy * l_energy
        return total, l_ode, l_ic, l_energy

    def predict(self, t_physical):
        """Predict z(t), w(t), phi(t) given physical time array."""
        self.model.eval()
        tau = t_physical / self.tau_scale
        tau_norm = torch.tensor(
            tau / self.tau_final, dtype=torch.float64, device=self.device
        ).unsqueeze(1)
        with torch.no_grad():
            out = self.model(tau_norm)
        z = out[:, 0].cpu().numpy()
        w = out[:, 1].cpu().numpy()
        phi = out[:, 2].cpu().numpy()
        theta = np.arccos(np.clip(z, -1, 1))
        return z, w, theta, phi

    def validate(self):
        """Compute validation errors against analytical solution."""
        z_pred, w_pred, theta_pred, phi_pred = self.predict(self.t_val)
        theta_val = np.arccos(np.clip(self.z_val, -1, 1))

        theta_err = np.abs(theta_pred - theta_val)
        phi_err = np.abs(phi_pred - self.phi_val)

        theta_l2 = np.sqrt(np.mean((theta_pred - theta_val)**2)) / \
                   np.sqrt(np.mean(theta_val**2))
        phi_l2 = np.sqrt(np.mean((phi_pred - self.phi_val)**2)) / \
                 np.sqrt(np.mean(self.phi_val**2))

        return {
            "theta_mae": theta_err.mean(),
            "theta_max": theta_err.max(),
            "phi_mae": phi_err.mean(),
            "phi_max": phi_err.max(),
            "theta_rel_l2": theta_l2,
            "phi_rel_l2": phi_l2,
        }
