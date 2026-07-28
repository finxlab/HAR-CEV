"""HAR-CEV core models.

Implements the models used in the paper:
  * HARAttentiveCEV       -- proposed model (stabilized one-step Euler drift)
  * HARAttentiveCEVExp    -- exponential drift discretization (robustness)
  * HARAttentiveCEVFixedAlpha -- ablation with alpha fixed at 0.5
  * HARAttentiveCEVAntiMR -- ablation: drift direction reversed
  * GRUWithLag  (No-SDE)  -- ablation: same encoder/ShapeNet, free location-scale

Shared blocks:
  * SDEContextEncoder -- 2-layer GRU over the 22-day RV window -> c_t (64)
  * HARShiftExtractor -- 7 hand-crafted regime-shift statistics f_t
  * SDEShapeNet       -- c_t -> 7 monotone standardized quantile residuals

All models output 7 predictive quantiles of next-day annualized realized
variance at tau = (5, 10, 25, 50, 75, 90, 95)%.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
D_SHIFT = 7


def quantile_loss(v_q, y_target, quantiles=None):
    """Mean pinball (quantile regression) loss. v_q: (B, 7), y: (B,)."""
    if quantiles is None:
        quantiles = QUANTILES
    q = torch.tensor(quantiles, dtype=v_q.dtype, device=v_q.device)
    e = y_target.unsqueeze(1) - v_q
    return torch.where(e >= 0, q * e, (q - 1) * e).mean()


def pinball_crps(v_q, y_target, quantiles=None):
    """Mean pinball loss; twice this equals the reported quantile score.

    With the median plus central 50/80/90% interval quantiles,
    (2/Q) * mean pinball is exactly the weighted interval score (WIS),
    a quantile-based approximation of the CRPS.
    """
    if quantiles is None:
        quantiles = QUANTILES
    q = torch.tensor(quantiles, dtype=v_q.dtype, device=v_q.device)
    e = y_target.unsqueeze(1) - v_q
    pb = torch.where(e >= 0, q * e, (q - 1) * e)
    return pb.mean(dim=1).mean()


def _center_eps_median(eps_q):
    """Force the standardized median innovation to zero: q50 == mu_t."""
    return eps_q - eps_q[:, 3:4]


class SDEContextEncoder(nn.Module):
    """GRU: normalized RV sequence -> context vector c_t."""

    def __init__(self, look_back=22, d_ctx=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=d_ctx,
                          num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return out[:, -1, :]


class SDEShapeNet(nn.Module):
    """c_t -> 7 monotone standardized quantile residuals [e05..e95].

    Monotonicity by construction: median + softplus-positive inter-quantile
    gaps. Initialized at the N(0,1) quantiles.
    """

    def __init__(self, d_ctx=64, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_ctx, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, 7),
        )

        def sp_inv(x):
            return float(np.log(np.expm1(x)))

        nn.init.zeros_(self.net[-1].weight)
        self.net[-1].bias.data = torch.tensor([
            0.0,                  # median residual = 0
            sp_inv(0.6745),       # e50 -> e25 gap
            sp_inv(0.6071),       # e25 -> e10 gap
            sp_inv(0.3633),       # e10 -> e05 gap
            sp_inv(0.6745),       # e50 -> e75 gap
            sp_inv(0.6071),       # e75 -> e90 gap
            sp_inv(0.3633),       # e90 -> e95 gap
        ])

    def forward(self, ctx):
        raw = self.net(ctx)
        med = raw[:, 0]
        g_l3 = F.softplus(raw[:, 1])
        g_l2 = F.softplus(raw[:, 2])
        g_l1 = F.softplus(raw[:, 3])
        g_u1 = F.softplus(raw[:, 4])
        g_u2 = F.softplus(raw[:, 5])
        g_u3 = F.softplus(raw[:, 6])
        q25 = med - g_l3
        q10 = q25 - g_l2
        q05 = q10 - g_l1
        q75 = med + g_u1
        q90 = q75 + g_u2
        q95 = q90 + g_u3
        return torch.stack([q05, q10, q25, med, q75, q90, q95], dim=1)


class HARShiftExtractor(nn.Module):
    """7 HAR-scale regime-shift statistics f_t (lookback window only).

    x = rv / rv_mean (normalized space); training-window statistics are
    stored in the same space.
    """

    def __init__(self, train_mean=1.0, train_std=0.5, rv_mean=1.0, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer(
            "mean_n", torch.tensor(train_mean / rv_mean, dtype=torch.float32))
        self.register_buffer(
            "std_n", torch.tensor(max(train_std / rv_mean, eps),
                                  dtype=torch.float32))

    def forward(self, x):
        v_last = x[:, -1]
        v_prev = x[:, -2] if x.shape[1] >= 2 else v_last

        level_z_d = (v_last - self.mean_n) / self.std_n
        level_z_w = (x[:, -5:].mean(1) - self.mean_n) / self.std_n
        level_z_m = (x.mean(1) - self.mean_n) / self.std_n
        jump_abs = (torch.log(v_last + self.eps)
                    - torch.log(v_prev + self.eps)).abs()
        std_w = x[:, -5:].std(dim=1).clamp(min=self.eps) / self.std_n
        std_m = x.std(dim=1).clamp(min=self.eps) / self.std_n
        std_ratio = x[:, -5:].std(dim=1).clamp(min=self.eps) / (
            x.std(dim=1).clamp(min=self.eps) + self.eps)

        return torch.stack(
            [level_z_d, level_z_w, level_z_m, jump_abs, std_w, std_m,
             std_ratio], dim=1)


class HARAttentiveCEV(nn.Module):
    """HAR-attentive CEV forecaster (the proposed model).

    Local equilibrium (HAR anchor):
        theta_t = w1*v_1 + w5*v_5 + w22*v_22,  w = softmax(MLP([c_t; f_t]))
    State-amplified mean reversion:
        kappa_t = kappa_base,t * exp(eta_t * max(log(v_1/theta_t), 0))
    Stabilized one-step Euler drift and CEV diffusion:
        mu_t = v_1 + min(kappa_t*dt, 1) * (theta_t - v_1)
        s_t  = sigma_t * v_1^alpha_t * sqrt(dt)
    Predictive quantiles:
        q_tau = max(mu_t + s_t * eps_tau, 1e-6),  eps from ShapeNet(c_t)
    """

    def __init__(self, look_back=22, d_ctx=64,
                 kappa_init=48.0, sigma_init=9.7,
                 kappa_min=2.0, theta_min=0.005, sigma_min=0.5,
                 alpha_init=0.634, alpha_min=0.1, alpha_max=1.5,
                 dt=1 / 252, rv_mean=1.0, train_mean=1.0, train_std=0.5,
                 discretization="euler"):
        super().__init__()
        self.dt = dt
        self.rv_mean = rv_mean
        self.discretization = discretization
        self.kappa_min = kappa_min
        self.theta_min = theta_min
        self.sigma_min = sigma_min
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.encoder = SDEContextEncoder(look_back, d_ctx)
        self.shift_extractor = HARShiftExtractor(train_mean, train_std,
                                                 rv_mean)
        self.shapenet = SDEShapeNet(d_ctx)

        self.param_head = nn.Sequential(
            nn.Linear(d_ctx + D_SHIFT, 64), nn.Tanh(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.param_head[-1].weight)
        alpha_raw = float(np.log((alpha_init - alpha_min)
                                 / (alpha_max - alpha_init)))
        self.param_head[-1].bias.data = torch.tensor([
            float(np.log(np.expm1(max(kappa_init - kappa_min, 0.01)))),
            float(np.log(np.expm1(0.10))),          # eta init
            float(np.log(np.expm1(max(sigma_init - sigma_min, 0.01)))),
            alpha_raw,
        ])

        self.har_attn = nn.Sequential(
            nn.Linear(d_ctx + D_SHIFT, 32), nn.Tanh(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.har_attn[-1].weight)
        # HAR-style start: daily persistence dominates.
        self.har_attn[-1].bias.data = torch.tensor([1.5, 0.5, 0.0])

    def _har_levels(self, x):
        v_1 = (x[:, -1] * self.rv_mean).clamp(min=1e-6)
        v_5 = (x[:, -5:].mean(dim=1) * self.rv_mean).clamp(min=1e-6)
        v_22 = (x.mean(dim=1) * self.rv_mean).clamp(min=1e-6)
        return torch.stack([v_1, v_5, v_22], dim=1), v_1

    def _alpha(self, raw_col):
        return self.alpha_min + (self.alpha_max - self.alpha_min) \
            * torch.sigmoid(raw_col)

    def _drift_fraction(self, kappa):
        if self.discretization == "euler":
            beta = kappa * self.dt
        elif self.discretization == "exp":
            beta = 1.0 - torch.exp(-kappa * self.dt)
        else:
            raise ValueError(f"Unknown discretization: {self.discretization}")
        return beta.clamp(min=0.0, max=1.0)

    def _compute(self, x):
        c_t = self.encoder(x)
        f_t = self.shift_extractor(x)
        h = torch.cat([c_t, f_t], dim=-1)

        raw = self.param_head(h)
        kappa_base = F.softplus(raw[:, 0]) + self.kappa_min
        eta = F.softplus(raw[:, 1])
        sigma = F.softplus(raw[:, 2]) + self.sigma_min
        alpha = self._alpha(raw[:, 3])

        theta_basis, v_1 = self._har_levels(x)
        weights = torch.softmax(self.har_attn(h), dim=-1)
        theta = (weights * theta_basis).sum(dim=1).clamp(min=self.theta_min)

        gap = F.relu(torch.log(v_1.clamp(min=1e-6) / theta.clamp(min=1e-6)))
        kappa = kappa_base * torch.exp(eta * gap)

        beta = self._drift_fraction(kappa)
        mu_t = (v_1 + beta * (theta - v_1)).clamp(min=1e-6)
        s_t = (sigma * v_1.pow(alpha) * (self.dt ** 0.5)).clamp(min=1e-6)

        eps_q = _center_eps_median(self.shapenet(c_t))
        v_q = (mu_t.unsqueeze(1) + s_t.unsqueeze(1) * eps_q).clamp(min=1e-6)
        return v_q, eps_q, mu_t, s_t, kappa, theta, sigma, alpha, beta, \
            weights, f_t

    def forward(self, x):
        v_q, _, mu_t, s_t, kappa, theta, sigma, alpha, beta, weights, f_t = \
            self._compute(x)
        return v_q, mu_t, s_t, kappa, theta, sigma, alpha, beta, weights, f_t

    def forward_eps(self, x):
        v_q, eps_q, mu_t, s_t, *_ = self._compute(x)
        return v_q, eps_q, mu_t, s_t


class HARAttentiveCEVExp(HARAttentiveCEV):
    """HAR-attentive CEV with exponential drift discretization."""

    def __init__(self, **kwargs):
        kwargs["discretization"] = "exp"
        super().__init__(**kwargs)


class HARAttentiveCEVFixedAlpha(HARAttentiveCEV):
    """Ablation: elasticity frozen at alpha = 0.5 (CIR/Heston diffusion).

    Everything else -- architecture, sigma initialization and bounds,
    optimizer schedule -- is identical to HARAttentiveCEV, so the
    comparison isolates the diffusion exponent alone.
    """

    def __init__(self, fixed_alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.fixed_alpha = float(fixed_alpha)

    def _alpha(self, raw_col):
        return torch.full_like(raw_col, self.fixed_alpha)


class HARAttentiveCEVAntiMR(HARAttentiveCEV):
    """Ablation: identical architecture, drift direction reversed.

        standard: mu_t = v_1 + beta_t * (theta_t - v_1)
        anti-MR : mu_t = v_1 - beta_t * (theta_t - v_1)
    """

    def _compute(self, x):
        v_q, eps_q, _, s_t, kappa, theta, sigma, alpha, beta, weights, f_t = \
            super()._compute(x)
        del v_q
        v_1 = (x[:, -1] * self.rv_mean).clamp(min=1e-6)
        mu_t = (v_1 - beta * (theta - v_1)).clamp(min=1e-6)
        v_q = (mu_t.unsqueeze(1) + s_t.unsqueeze(1) * eps_q).clamp(min=1e-6)
        return v_q, eps_q, mu_t, s_t, kappa, theta, sigma, alpha, beta, \
            weights, f_t


class GRUWithLag(nn.Module):
    """No-SDE ablation: same encoder and ShapeNet, free location and scale.

        mu_t = v_1 * softplus(head([c_t; v_1]))   (free scaling, no drift)
        s_t  = softplus(head([c_t; v_1]))          (free scale, no CEV)
    """

    def __init__(self, look_back=22, d_ctx=64, rv_mean=1.0, **kwargs):
        super().__init__()
        self.rv_mean = rv_mean
        self.encoder = SDEContextEncoder(look_back, d_ctx)
        self.shapenet = SDEShapeNet(d_ctx)
        self.head = nn.Linear(d_ctx + 1, 2)
        nn.init.zeros_(self.head.weight)
        # softplus(0.541) ~= 1.0 -> mu ~= v_1;  softplus(-1.5) ~= 0.20
        self.head.bias.data = torch.tensor([0.541, -1.5])

    def _compute(self, x):
        c_t = self.encoder(x)
        raw = self.head(torch.cat([c_t, x[:, -1:]], dim=-1))
        v_1 = x[:, -1] * self.rv_mean
        mu_t = (v_1 * F.softplus(raw[:, 0])).clamp(min=1e-6)
        s_t = F.softplus(raw[:, 1]).clamp(min=1e-6)
        eps_q = _center_eps_median(self.shapenet(c_t))
        return mu_t, s_t, eps_q

    def forward(self, x):
        mu_t, s_t, eps_q = self._compute(x)
        v_q = (mu_t.unsqueeze(1) + s_t.unsqueeze(1) * eps_q).clamp(min=1e-6)
        return v_q, mu_t, s_t

    def forward_eps(self, x):
        mu_t, s_t, eps_q = self._compute(x)
        v_q = (mu_t.unsqueeze(1) + s_t.unsqueeze(1) * eps_q).clamp(min=1e-6)
        return v_q, eps_q, mu_t, s_t

