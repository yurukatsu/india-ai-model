"""Masked (denoising) autoencoder for imputing missing ranked features.

Training: observed cells are hidden at random (``mask_frac``) and replaced by the fill
value; the network reconstructs the full row and the loss is taken on the hidden cells
only, so the model learns "predict a feature from the other features" rather than the
identity. Cells that are missing in the data never enter the loss.

Transform: rows pass through the trained network and only the originally missing
cells are replaced by the reconstruction; observed values are untouched.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


class MaskedAEImputer:
    def __init__(self, params: dict | None = None):
        self.p = {
            "hidden": [64, 32],
            "dropout": 0.1,
            "mask_frac": 0.25,
            "fill": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
            "max_epochs": 50,
            "patience": 5,
            "holdout_frac": 0.1,
            "seed": 0,
            "device": "cpu",
            **(params or {}),
        }
        self.net: nn.Module | None = None
        self.info_: dict = {}

    def _build(self, n: int) -> nn.Module:
        h = list(self.p["hidden"])
        dims = [n] + h + h[-2::-1] + [n]  # symmetric: n -> 64 -> 32 -> 64 -> n
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [
                    nn.LayerNorm(dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(float(self.p["dropout"])),
                ]
        return nn.Sequential(*layers)

    def _masked_mse(self, net, x, obs, rng_t) -> torch.Tensor:
        hide = obs & (
            torch.rand(x.shape, generator=rng_t, device=x.device) < float(self.p["mask_frac"])
        )
        x_in = torch.where(hide, torch.full_like(x, float(self.p["fill"])), x)
        out = net(x_in)
        n = hide.sum()
        return ((out - x) ** 2)[hide].sum() / n.clamp(min=1)

    def fit(self, X: np.ndarray, observed: np.ndarray) -> MaskedAEImputer:
        """X: (n, f) float32 with missing cells already filled; observed: (n, f) bool."""
        p = self.p
        torch.manual_seed(int(p["seed"]))
        rng = np.random.default_rng(int(p["seed"]))
        dev = torch.device(p["device"])
        n = len(X)
        perm = rng.permutation(n)
        n_ho = int(n * float(p["holdout_frac"]))
        ho, tr = perm[:n_ho], perm[n_ho:]
        Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
        Ot = torch.from_numpy(observed).to(dev)
        net = self._build(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(
            net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        gen = torch.Generator(device=dev).manual_seed(int(p["seed"]))
        gen_ho = torch.Generator(device=dev)
        bs = int(p["batch_size"])

        def holdout_loss() -> float:
            net.eval()
            gen_ho.manual_seed(123)  # same hidden cells every evaluation
            with torch.no_grad():
                return float(self._masked_mse(net, Xt[ho], Ot[ho], gen_ho))

        # reference: error of the constant fill on the same hidden cells
        gen_ho.manual_seed(123)
        hide = Ot[ho] & (
            torch.rand(Xt[ho].shape, generator=gen_ho, device=dev) < float(p["mask_frac"])
        )
        base = float(((Xt[ho] - float(p["fill"])) ** 2)[hide].mean())

        best, best_state, bad, best_epoch = np.inf, copy.deepcopy(net.state_dict()), 0, 0
        for epoch in range(1, int(p["max_epochs"]) + 1):
            net.train()
            order = rng.permutation(tr)
            for b in range(0, len(order), bs):
                idx = torch.from_numpy(order[b : b + bs]).to(dev)
                loss = self._masked_mse(net, Xt[idx], Ot[idx], gen)
                opt.zero_grad()
                loss.backward()
                opt.step()
            lh = holdout_loss()
            if lh < best - 1e-6:
                best, best_state, bad, best_epoch = lh, copy.deepcopy(net.state_dict()), 0, epoch
            else:
                bad += 1
                if bad >= int(p["patience"]):
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.info_ = {
            "ae_epoch": best_epoch,
            "ae_holdout_mse": round(best, 5),
            "fill_holdout_mse": round(base, 5),
            "ae_r2_vs_fill": round(1 - best / base, 3),
        }
        return self

    def transform(self, X: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """Replace the missing cells of X by the reconstruction; returns a new array."""
        dev = torch.device(self.p["device"])
        out = X.astype(np.float32).copy()
        with torch.no_grad():
            rec = self.net(torch.from_numpy(out).to(dev)).cpu().numpy()
        rec = np.clip(rec, 0.0, 1.0)
        miss = ~observed
        out[miss] = rec[miss]
        return out
