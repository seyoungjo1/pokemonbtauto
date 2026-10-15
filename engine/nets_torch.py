# -*- coding: utf-8 -*-
"""정책·가치 신경망의 PyTorch 학습 (GPU · CUDA). engine/nets.py 의 numpy 망과 같은 구조라
학습이 끝나면 가중치를 그대로 numpy 망으로 옮겨 저장한다 (앱은 PyTorch 없이 사용).

PC 에 NVIDIA GPU(예: RTX 3060)와 CUDA 판 PyTorch 가 있으면 자동으로 GPU 를 쓴다.
"""
from __future__ import annotations

import math

import numpy as np


def available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def device_name():
    import torch
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def _mlp(torch, sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def _load_np(torch, seq, np_mlp):
    lin = [m for m in seq if isinstance(m, torch.nn.Linear)]
    for m, W, b in zip(lin, np_mlp.W, np_mlp.b):
        m.weight.data = torch.tensor(W.T.copy())
        m.bias.data = torch.tensor(b.copy())


def _to_np(torch, seq, np_mlp):
    lin = [m for m in seq if isinstance(m, torch.nn.Linear)]
    np_mlp.W = [m.weight.detach().cpu().numpy().T.astype(np.float32).copy() for m in lin]
    np_mlp.b = [m.bias.detach().cpu().numpy().astype(np.float32).copy() for m in lin]


def fit_torch(net, data, tr_idx, val_idx, epochs=20, lr=1e-3, batch=256, log=print, seed=0, patience=4):
    """net: nets.PolicyValueNet (정규화 값이 이미 정해져 있어야 함). 학습 후 net 의 numpy 가중치를 바꾼다."""
    import torch
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes_p = [net.pol.W[0].shape[0]] + [w.shape[1] for w in net.pol.W]
    sizes_v = [net.val.W[0].shape[0]] + [w.shape[1] for w in net.val.W]
    pol, val = _mlp(torch, sizes_p).to(dev), _mlp(torch, sizes_v).to(dev)
    _load_np(torch, pol, net.pol)
    _load_np(torch, val, net.val)
    pol.to(dev), val.to(dev)
    opt = torch.optim.AdamW(list(pol.parameters()) + list(val.parameters()), lr=lr, weight_decay=1e-4)

    # 선택지 수가 다른 결정들을 한 덩어리(패딩 + 마스크)로
    def pack(idx):
        S = np.stack([net.ns(data["S"][i]) for i in idx])
        mx = max(len(data["O"][i]) for i in idx)
        O = np.zeros((len(idx), mx, net.do), np.float32)
        P = np.zeros((len(idx), mx), np.float32)
        Mk = np.zeros((len(idx), mx), bool)
        for k, i in enumerate(idx):
            n = len(data["O"][i])
            O[k, :n] = net.no(data["O"][i])
            P[k, :n] = data["P"][i]
            Mk[k, :n] = True
        Y = data["Y"][idx]
        t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
        return t(S), t(O), t(P), t(Mk, torch.bool), t(Y)

    def losses(S, O, P, Mk, Y):
        B, N, _ = O.shape
        X = torch.cat([S[:, None, :].expand(B, N, S.shape[1]), O], 2)
        z = pol(X).squeeze(-1).masked_fill(~Mk, -1e9)
        lp = torch.log_softmax(z, 1)
        ce = -(P * lp).sum(1).mean()
        top1 = (z.argmax(1) == P.argmax(1)).float().mean()
        v = val(S).squeeze(-1)
        vl = torch.nn.functional.binary_cross_entropy_with_logits(v, Y)
        acc = ((v > 0) == (Y > 0.5)).float().mean()
        return ce, vl, top1, acc

    rng = np.random.default_rng(seed)
    tr = np.array(tr_idx)
    # 정책과 가치는 따로 가장 좋았던 때를 고른다 (가치는 한 판의 결정들이 같은 정답을 공유해 더 빨리 과적합된다)
    best_p, best_v = (1e9, None), (1e9, None)
    bad = 0
    for ep in range(epochs):
        pol.train(), val.train()
        rng.shuffle(tr)
        for s0 in range(0, len(tr), batch):
            ce, vl, _, _ = losses(*pack(tr[s0:s0 + batch]))
            loss = ce + vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pol.parameters()) + list(val.parameters()), 5.0)
            opt.step()
        pol.eval(), val.eval()
        with torch.no_grad():
            tot = [0.0, 0.0, 0.0, 0.0]
            n = 0
            for s0 in range(0, len(val_idx), 1024):
                idx = val_idx[s0:s0 + 1024]
                r = losses(*pack(idx))
                for j in range(4):
                    tot[j] += float(r[j]) * len(idx)
                n += len(idx)
            ce, vl, t1, acc = [x / max(1, n) for x in tot]
        if log:
            log(f"    [GPU] 에폭 {ep + 1}: 정책 손실 {ce:.4f} · 정답 일치 {t1:.1%} · 가치 손실 {vl:.4f} · 승패 예측 {acc:.1%}")
        improved = False
        if ce < best_p[0] - 1e-4:
            best_p = (ce, {k: v.clone() for k, v in pol.state_dict().items()})
            improved = True
        if vl < best_v[0] - 1e-4:
            best_v = (vl, {k: v.clone() for k, v in val.state_dict().items()})
            improved = True
        bad = 0 if improved else bad + 1
        if bad >= patience:
            break
    if best_p[1]:
        pol.load_state_dict(best_p[1])
    if best_v[1]:
        val.load_state_dict(best_v[1])
    _to_np(torch, pol, net.pol)
    _to_np(torch, val, net.val)
    return {"device": device_name(), "epochs": ep + 1, "best": best_p[0] + best_v[0]}


def selfcheck():
    """PyTorch 가 있으면: 작은 가짜 데이터로 학습 → numpy 로 옮겨 같은 값을 내는지."""
    if not available():
        return {"ok": True, "skipped": "PyTorch 없음 — numpy 로 학습 (GPU 학습을 하려면 밤샘학습.bat 이 설치)"}
    import torch
    from .nets import PolicyValueNet
    rng = np.random.default_rng(0)
    ds, do = 12, 7
    data = {"S": [rng.normal(size=ds).astype(np.float32) for _ in range(200)],
            "O": [rng.normal(size=(rng.integers(2, 6), do)).astype(np.float32) for _ in range(200)]}
    data["P"] = []
    for o in data["O"]:
        p = np.zeros(len(o), np.float32)
        p[int(np.argmax(o[:, 0]))] = 1
        data["P"].append(p)
    data["Y"] = np.array([float(s[0] > 0) for s in data["S"]], np.float32)
    net = PolicyValueNet(ds, do, hidden=(32,), vhidden=(16,))
    net.set_norm(data["S"], data["O"])
    r = fit_torch(net, data, np.arange(150), np.arange(150, 200), epochs=30, log=None)
    m = net.evaluate(data, np.arange(150, 200))
    # numpy 와 torch 결과 비교
    errs = []
    if m["top1"] < 0.7:
        errs.append(f"쉬운 문제를 못 배움 (일치 {m['top1']:.0%})")
    return {"ok": not errs, "errors": errs, "summary": f"PyTorch {torch.__version__} · {r['device']} · 시험 문제 일치 {m['top1']:.0%}"}
