"""Benchmark quantile forecasters: HAR, DLinear, LSTM, TCN, N-BEATS,
PatchTST, iTransformer.

Every model maps a normalized 22-day RV window to 7 monotone quantiles of
next-day annualized RV via a shared quantile head (median + softplus gaps),
so all baselines face the identical output parameterization as HAR-CEV's
ShapeNet.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sp_inv(x):
    return float(np.log(np.expm1(max(x, 1e-6))))


def _init_quantile_bias(bias_param, rv_mean):
    """Initialize quantile head bias: median ~= rv_mean, N(0,1)-scaled gaps."""
    bias_param.data = torch.tensor([
        float(np.log(max(rv_mean, 1e-6))),   # exp -> q50 ~= rv_mean
        _sp_inv(rv_mean * 0.6745),           # q50 -> q25
        _sp_inv(rv_mean * 0.6071),           # q25 -> q10
        _sp_inv(rv_mean * 0.3633),           # q10 -> q05
        _sp_inv(rv_mean * 0.6745),           # q50 -> q75
        _sp_inv(rv_mean * 0.6071),           # q75 -> q90
        _sp_inv(rv_mean * 0.3633),           # q90 -> q95
    ])


def _decode_quantiles(raw):
    """raw (B, 7) = [log_med, 3 lower gaps, 3 upper gaps] -> monotone
    quantiles in absolute RV space."""
    med = torch.exp(raw[:, 0]).clamp(min=1e-6)
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
    return torch.stack([q05, q10, q25, med, q75, q90, q95],
                       dim=1).clamp(min=1e-6)


def _make_head(d_in, d_hidden, rv_mean):
    head = nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.Tanh(),
        nn.Linear(d_hidden, 7),
    )
    nn.init.zeros_(head[-1].weight)
    _init_quantile_bias(head[-1].bias, rv_mean)
    return head


class HARQuantile(nn.Module):
    """HAR with quantile output: [v_1, v_5bar, v_22bar] -> linear -> 7q."""

    def __init__(self, look_back=22, rv_mean=0.15):
        super().__init__()
        self.linear = nn.Linear(3, 7)
        nn.init.zeros_(self.linear.weight)
        _init_quantile_bias(self.linear.bias, rv_mean)

    def forward(self, x):
        feat = torch.stack([x[:, -1], x[:, -5:].mean(dim=1), x.mean(dim=1)],
                           dim=1)
        return (_decode_quantiles(self.linear(feat)),)


class DLinearQuantile(nn.Module):
    """DLinear: moving-average trend/seasonal decomposition, linear heads."""

    def __init__(self, look_back=22, rv_mean=0.15, kernel_size=5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=pad)
        self.linear_trend = nn.Linear(look_back, 7)
        self.linear_seasonal = nn.Linear(look_back, 7)
        for lin in (self.linear_trend, self.linear_seasonal):
            nn.init.zeros_(lin.weight)
        _init_quantile_bias(self.linear_trend.bias, rv_mean)
        nn.init.zeros_(self.linear_seasonal.bias)

    def forward(self, x):
        trend = self.avg(x.unsqueeze(1)).squeeze(1)
        raw = self.linear_trend(trend) + self.linear_seasonal(x - trend)
        return (_decode_quantiles(raw),)


class LSTMQuantile(nn.Module):
    """2-layer LSTM encoder -> quantile head."""

    def __init__(self, look_back=22, d_ctx=64, rv_mean=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=d_ctx, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = _make_head(d_ctx, d_ctx, rv_mean)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return (_decode_quantiles(self.head(out[:, -1, :])),)


class _CausalConv1d(nn.Module):
    """Dilated causal convolution: left-pad then trim the right overhang."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.pad, dilation=dilation)

    def forward(self, x):                    # (B, C, T)
        out = self.conv(x)
        return F.relu(out[:, :, :-self.pad].contiguous()
                      if self.pad > 0 else out)


class TCNQuantile(nn.Module):
    """Stack of exponentially dilated causal convolutions -> quantile head."""

    def __init__(self, look_back=22, d_ctx=64, rv_mean=0.15,
                 kernel_size=2, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        layers, in_ch = [], 1
        for d in dilations:
            layers.append(_CausalConv1d(in_ch, d_ctx, kernel_size, d))
            in_ch = d_ctx
        self.tcn = nn.Sequential(*layers)
        self.head = _make_head(d_ctx, d_ctx, rv_mean)

    def forward(self, x):
        h = self.tcn(x.unsqueeze(1))         # (B, d_ctx, T)
        return (_decode_quantiles(self.head(h[:, :, -1])),)


class _NBEATSBlock(nn.Module):
    def __init__(self, look_back, n_hidden, n_layers):
        super().__init__()
        layers = [nn.Linear(look_back, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.bc_proj = nn.Linear(n_hidden, look_back)
        self.fc_proj = nn.Linear(n_hidden, 7)

    def forward(self, x):
        h = self.fc(x)
        return self.bc_proj(h), self.fc_proj(h)


class NBEATSQuantile(nn.Module):
    """N-BEATS generic stack with doubly residual quantile forecasting."""

    def __init__(self, look_back=22, n_blocks=3, n_hidden=128, n_layers=4,
                 rv_mean=0.15):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_NBEATSBlock(look_back, n_hidden, n_layers)
             for _ in range(n_blocks)])
        for blk in self.blocks:
            nn.init.zeros_(blk.fc_proj.weight)
        _init_quantile_bias(self.blocks[0].fc_proj.bias, rv_mean)
        for blk in self.blocks[1:]:
            nn.init.zeros_(blk.fc_proj.bias)

    def forward(self, x):
        residual = x
        raw_sum = torch.zeros(x.size(0), 7, device=x.device)
        for blk in self.blocks:
            backcast, fc = blk(residual)
            residual = residual - backcast
            raw_sum = raw_sum + fc
        return (_decode_quantiles(raw_sum),)


class _TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(x2))
        return self.norm2(x + self.drop(self.ff(x)))


class PatchTSTQuantile(nn.Module):
    """PatchTST for 1-step quantile forecasting (patch 4, stride 2)."""

    def __init__(self, look_back=22, d_model=32, n_heads=4, n_layers=2,
                 patch_len=4, stride=2, rv_mean=0.15):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (look_back - patch_len) // stride + 1
        self.patch_emb = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.encoder = nn.Sequential(
            *[_TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.head = _make_head(n_patches * d_model, d_model * 2, rv_mean)

    def forward(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len,
                           step=self.stride)
        z = self.patch_emb(patches) + self.pos_emb
        z = self.encoder(z).flatten(1)
        return (_decode_quantiles(self.head(z)),)


class iTransformerQuantile(nn.Module):
    """iTransformer: inverted tokens, no positional encoding, mean pool."""

    def __init__(self, look_back=22, d_model=32, n_heads=4, n_layers=2,
                 rv_mean=0.15):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.encoder = nn.Sequential(
            *[_TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.head = _make_head(d_model, d_model * 2, rv_mean)

    def forward(self, x):
        z = self.encoder(self.proj(x.unsqueeze(-1)))
        return (_decode_quantiles(self.head(z.mean(dim=1))),)


ALL_BENCHMARK_MODELS = ["HAR", "DLinear", "LSTM", "TCN", "NBEATS",
                        "PatchTST", "iTransformer"]


def build_model(model_name, look_back, rv_mean):
    cls = {
        "HAR": HARQuantile,
        "DLinear": DLinearQuantile,
        "LSTM": LSTMQuantile,
        "TCN": TCNQuantile,
        "NBEATS": NBEATSQuantile,
        "PatchTST": PatchTSTQuantile,
        "iTransformer": iTransformerQuantile,
    }[model_name]
    if model_name in ("HAR", "DLinear"):
        return cls(look_back, rv_mean)
    return cls(look_back, rv_mean=rv_mean)
