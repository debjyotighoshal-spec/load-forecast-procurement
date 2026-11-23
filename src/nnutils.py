"""Windowing, recurrent model and training loop shared by the RNN, LSTM and
quantile-LSTM scripts."""
import copy

import numpy as np
import torch
from torch import nn

from common import set_seed

torch.set_num_threads(4)


def make_windows(z, cal, lag, target_mask):
    """X[i] = z[t-lag:t], C[i] = calendar(t), Y[i] = z[t] for every t in mask.

    The window for day t only contains days strictly before t, so every
    prediction is one step ahead. Windows may reach back into an earlier
    split (e.g. a validation target uses the last training days as input),
    which is how the model would be used in production.
    """
    idx = np.where(target_mask)[0]
    idx = idx[idx >= lag]
    X = np.stack([z[t - lag:t] for t in idx])[..., None]
    C = cal[idx] if cal is not None else np.zeros((len(idx), 0))
    Y = z[idx]
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(C, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32)), idx


class RecurrentForecaster(nn.Module):
    def __init__(self, cell="lstm", hidden=32, layers=1, n_cal=0, dropout=0.1):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn(input_size=1, hidden_size=hidden, num_layers=layers,
                       batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + n_cal, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x, c):
        h, _ = self.rnn(x)
        return self.head(torch.cat([h[:, -1, :], c], dim=1)).squeeze(-1)


def pinball(pred, y, tau):
    e = y - pred
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))


def train_model(cfg, train, val, loss="mse", tau=None, max_epochs=200,
                patience=20, seed=0, verbose=False):
    """Adam + early stopping on the validation loss; returns the best model."""
    set_seed(seed)
    Xtr, Ctr, Ytr = train
    Xva, Cva, Yva = val
    model = RecurrentForecaster(cfg["cell"], cfg["hidden"], cfg["layers"],
                                n_cal=Ctr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = (lambda p, y: torch.mean((p - y) ** 2)) if loss == "mse" else \
        (lambda p, y: pinball(p, y, tau))
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = np.inf, None, 0, []
    n = len(Ytr)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            l = lossf(model(Xtr[b], Ctr[b]), Ytr[b])
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xva, Cva), Yva))
            tl = float(lossf(model(Xtr, Ctr), Ytr))
        hist.append((tl, vl))
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and ep % 20 == 0:
            print(ep, tl, vl)
    model.load_state_dict(best_state)
    return model, best, hist


def predict(model, data):
    X, C, _ = data
    model.eval()
    with torch.no_grad():
        return model(X, C).numpy()
