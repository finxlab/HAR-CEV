"""Fast, data-free sanity check for every neural model."""

import torch

from baselines import ALL_BENCHMARK_MODELS, build_model
from models import (
    GRUWithLag,
    HARAttentiveCEV,
    HARAttentiveCEVAntiMR,
    HARAttentiveCEVExp,
    HARAttentiveCEVFixedAlpha,
    quantile_loss,
)


def check(name, model, x):
    model.eval()
    with torch.no_grad():
        result = model(x)
        q = result[0] if isinstance(result, tuple) else result
    assert q.shape == (len(x), 7), f"{name}: unexpected shape {q.shape}"
    assert torch.isfinite(q).all(), f"{name}: non-finite output"
    assert (q > 0).all(), f"{name}: non-positive volatility"
    assert (q[:, 1:] >= q[:, :-1]).all(), f"{name}: crossing quantiles"
    loss = quantile_loss(q, torch.full((len(x),), 0.15))
    assert torch.isfinite(loss), f"{name}: non-finite loss"
    print(f"[ OK ] {name}")


def main():
    torch.manual_seed(42)
    x = torch.exp(0.1 * torch.randn(4, 22))
    rv_mean = 0.15
    models = {
        "HARCEV_Euler": HARAttentiveCEV(rv_mean=rv_mean),
        "HARCEV_FixedA": HARAttentiveCEVFixedAlpha(rv_mean=rv_mean),
        "HARCEV_Exp": HARAttentiveCEVExp(rv_mean=rv_mean),
        "AntiMR": HARAttentiveCEVAntiMR(rv_mean=rv_mean),
        "GRULag": GRUWithLag(rv_mean=rv_mean),
    }
    for name in ALL_BENCHMARK_MODELS:
        models[name] = build_model(name, look_back=22, rv_mean=rv_mean)
    for name, model in models.items():
        check(name, model, x)
    print(f"All {len(models)} model checks passed.")


if __name__ == "__main__":
    main()

