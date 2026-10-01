"""Small MLP trained with a per-date IC loss on date batches, seed-ensembled.

Loss for a batch of dates D:  -sum_d w_d * IC_d  (+ mse_weight * MSE(z(pred_d), y_d))
with IC_d the Pearson correlation of the prediction and the Gaussian-ranked target
within date d, w_d = sqrt(N_d) / sum sqrt(N).  Early stopping on the mean per-date
Spearman IC of the validation set. Predictions of the seed ensemble are averaged after
z-scoring each member (the IC loss leaves the output scale undetermined).

params:
  arch: mlp | attn        attn = MLP encoder + 1 block of self-attention across the
                          stocks of the same date (set attention, no positional encoding)
  attn_heads: 2
  attn_dropout: 0.2
  hidden: [64, 32]        layer widths (attn: hidden[0] = encoder/attention width,
                          hidden[1:] = head widths)
  dropout: 0.2
  input_noise: 0.05       Gaussian noise std added to (standardised) inputs in training
  activation: gelu | relu
  lr: 1e-3
  weight_decay: 1e-4
  dates_per_batch: 16
  max_epochs: 200
  patience: 20            epochs without val-IC improvement (ignored without val)
  epochs_no_val: 30       epochs to train when there is no validation set
  mse_weight: 0.0
  seeds: 5                number of ensemble members (seeds 0..seeds-1) or a list
  eps: 1e-6
  device: cpu
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from india_ai_model.common.evaluate import mean_ic_fast

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def _mlp_layers(n_in: int, hidden: list[int], dropout: float, activation: str) -> list[nn.Module]:
    act = _ACT[activation]
    layers: list[nn.Module] = []
    d = n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), act(), nn.Dropout(dropout)]
        d = h
    return layers


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float, activation: str):
        super().__init__()
        self.net = nn.Sequential(
            *_mlp_layers(n_in, hidden, dropout, activation), nn.Linear(hidden[-1], 1)
        )

    def forward(self, x: torch.Tensor, groups: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CrossSectionAttn(nn.Module):
    """Per-stock encoder -> self-attention over the stocks of one date -> head.

    ``groups`` lists the row positions of each date in ``x``; attention never crosses dates.
    """

    def __init__(
        self,
        n_in: int,
        hidden: list[int],
        dropout: float,
        activation: str,
        *,
        heads: int,
        attn_dropout: float,
    ):
        super().__init__()
        d = hidden[0]
        self.enc = nn.Sequential(*_mlp_layers(n_in, [d], dropout, activation))
        self.attn = nn.MultiheadAttention(d, heads, dropout=attn_dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d), _ACT[activation](), nn.Dropout(dropout), nn.Linear(d, d)
        )
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            *_mlp_layers(d, hidden[1:], dropout, activation), nn.Linear(hidden[-1], 1)
        )

    def forward(self, x: torch.Tensor, groups: list[torch.Tensor] | None = None) -> torch.Tensor:
        h = self.enc(x)
        out = torch.empty_like(h)
        groups = groups if groups is not None else [torch.arange(len(x), device=x.device)]
        for g in groups:
            z = h[g].unsqueeze(0)  # (1, N_d, d)
            a, _ = self.attn(z, z, z, need_weights=False)
            z = self.ln1(z + self.drop(a))
            z = self.ln2(z + self.drop(self.ffn(z)))
            out[g] = z.squeeze(0)
        return self.head(out).squeeze(-1)


def build_net(n_in: int, p: dict) -> nn.Module:
    hidden, dropout, act = list(p["hidden"]), float(p["dropout"]), p["activation"]
    if p.get("arch", "mlp") == "mlp":
        return MLP(n_in, hidden, dropout, act)
    if p["arch"] == "attn":
        return CrossSectionAttn(
            n_in,
            hidden,
            dropout,
            act,
            heads=int(p.get("attn_heads", 2)),
            attn_dropout=float(p.get("attn_dropout", 0.2)),
        )
    raise ValueError(p["arch"])


def _date_groups(d: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(d, kind="stable")
    _, starts, counts = np.unique(d[order], return_index=True, return_counts=True)
    return [order[s : s + c] for s, c in zip(starts, counts, strict=True)]


def ic_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    groups: list[torch.Tensor],
    *,
    eps: float,
    mse_weight: float,
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted mean over dates of -IC_d (+ mse term).

    Without ``w``: date weight = sqrt(N_d), names equally weighted.
    With ``w`` (per-row sample weights): date weight = sum of w in the date, and the
    within-date correlation is weighted by w normalised to sum 1.
    """
    if w is None:
        dw = torch.tensor([np.sqrt(len(g)) for g in groups], dtype=pred.dtype)
    else:
        dw = torch.stack([w[g].sum() for g in groups])
    dw = dw / dw.sum()
    total = pred.new_zeros(())
    for wi, g in zip(dw, groups, strict=True):
        p = pred[g]
        t = y[g]
        if w is None:
            v = torch.full_like(p, 1.0 / len(g))
        else:
            v = w[g] / w[g].sum()
        pc = p - (v * p).sum()
        tc = t - (v * t).sum()
        p_std = torch.sqrt((v * pc * pc).sum() + eps)
        t_std = torch.sqrt((v * tc * tc).sum() + eps)
        ic = (v * pc * tc).sum() / (p_std * t_std)
        loss = -ic
        if mse_weight > 0:
            loss = loss + mse_weight * (v * (pc / p_std - tc / t_std) ** 2).sum()
        total = total + wi * loss
    return total


class Model:
    def __init__(self, params: dict, feature_names: list[str]):
        self.p = {
            "hidden": [64, 32],
            "dropout": 0.2,
            "input_noise": 0.05,
            "activation": "gelu",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dates_per_batch": 16,
            "max_epochs": 200,
            "patience": 20,
            "epochs_no_val": 30,
            "mse_weight": 0.0,
            "seeds": 5,
            "eps": 1e-6,
            "device": "cpu",
            **params,
        }
        self.feature_names = feature_names
        self.nets: list[MLP] = []
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------- helpers
    def _seeds(self) -> list[int]:
        s = self.p["seeds"]
        return list(range(int(s))) if isinstance(s, int) else [int(x) for x in s]

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)

    def _train_one(self, seed: int, X_tr, y_tr, d_tr, X_va, y_va, d_va, w_tr=None):
        p = self.p
        torch.manual_seed(seed)
        dev = torch.device(p["device"])
        net = build_net(X_tr.shape[1], p).to(dev)
        opt = torch.optim.AdamW(
            net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        Xt = torch.from_numpy(X_tr).to(dev)
        yt = torch.from_numpy(y_tr).to(dev)
        wt = (
            torch.from_numpy(np.asarray(w_tr, dtype=np.float32)).to(dev)
            if w_tr is not None
            else None
        )
        groups = [torch.from_numpy(g).to(dev) for g in _date_groups(d_tr)]
        has_val = X_va is not None and len(X_va) > 0
        Xv = torch.from_numpy(X_va).to(dev) if has_val else None
        gv = [torch.from_numpy(g).to(dev) for g in _date_groups(d_va)] if has_val else None
        rng = np.random.default_rng(seed)
        n_epochs = int(p["max_epochs"]) if has_val else int(p["epochs_no_val"])
        dpb = int(p["dates_per_batch"])
        noise = float(p["input_noise"])

        best_ic, best_state, best_epoch, bad = -np.inf, copy.deepcopy(net.state_dict()), 0, 0
        for epoch in range(1, n_epochs + 1):
            net.train()
            order = rng.permutation(len(groups))
            for b in range(0, len(order), dpb):
                bg = [groups[i] for i in order[b : b + dpb]]
                idx = torch.cat(bg)
                x = Xt[idx]
                if noise > 0:
                    x = x + noise * torch.randn_like(x)
                # remap groups to positions inside the batch
                pos, cur = [], 0
                for g in bg:
                    pos.append(torch.arange(cur, cur + len(g), device=dev))
                    cur += len(g)
                pred_all = net(x, pos)
                loss = ic_loss(
                    pred_all,
                    yt[idx],
                    pos,
                    eps=float(p["eps"]),
                    mse_weight=float(p["mse_weight"]),
                    w=None if wt is None else wt[idx],
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            if has_val:
                net.eval()
                with torch.no_grad():
                    pv = net(Xv, gv).cpu().numpy()
                ic = mean_ic_fast(pv, y_va, d_va)
                if ic > best_ic:
                    best_ic, best_state, best_epoch, bad = (
                        ic,
                        copy.deepcopy(net.state_dict()),
                        epoch,
                        0,
                    )
                else:
                    bad += 1
                    if bad >= int(p["patience"]):
                        break
            else:
                best_epoch = epoch
        if has_val:
            net.load_state_dict(best_state)
        net.eval()
        return net, {
            "seed": seed,
            "best_epoch": best_epoch,
            "val_ic": None if not has_val else round(float(best_ic), 4),
        }

    # ----------------------------------------------------------------- api
    def fit(self, X_tr, y_tr, d_tr, X_va, y_va, d_va, w_tr=None) -> None:
        self.mu = X_tr.mean(axis=0)
        self.sd = X_tr.std(axis=0) + 1e-6
        Xs_tr = self._standardise(X_tr)
        Xs_va = self._standardise(X_va) if X_va is not None and len(X_va) else None
        self.nets, self.history = [], []
        for seed in self._seeds():
            net, h = self._train_one(seed, Xs_tr, y_tr, d_tr, Xs_va, y_va, d_va, w_tr)
            self.nets.append(net)
            self.history.append(h)

    def predict(self, X, dates=None) -> np.ndarray:
        dev = torch.device(self.p["device"])
        Xs = torch.from_numpy(self._standardise(X)).to(dev)
        groups = (
            [torch.from_numpy(g).to(dev) for g in _date_groups(dates)]
            if dates is not None
            else None
        )
        outs = []
        with torch.no_grad():
            for net in self.nets:
                o = net(Xs, groups).cpu().numpy().astype(np.float64)
                outs.append((o - o.mean()) / (o.std() + 1e-12))
        return np.mean(outs, axis=0)

    def info(self) -> dict:
        ics = [h["val_ic"] for h in self.history if h["val_ic"] is not None]
        return {
            "epochs": [h["best_epoch"] for h in self.history],
            "val_ic_members": ics,
            "val_ic_mean": round(float(np.mean(ics)), 4) if ics else None,
        }
