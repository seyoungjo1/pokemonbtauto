# -*- coding: utf-8 -*-
"""토큰 배틀망의 PyTorch(GPU) 학습 경로 — engine/battlenet.py(numpy, engine/tinytf.py) 와 수식이 1:1 로 같다.

  * torch 는 함수 안에서만 import 한다 (샌드박스처럼 torch 가 없어도 모듈 import 는 된다).
  * 가중치는 numpy 망(tinytf 파라미터 배열) ↔ torch 모듈 사이를 양방향으로 복사한다
    (tinytf Linear 는 W[d_in, d_out] · y = x @ W + b, torch Linear 는 weight[d_out, d_in] → 전치).
  * fit_torch: numpy 망을 warm start 로 받아 AdamW 로 학습하고, 가장 좋은 에폭의 가중치를 numpy 망에 되돌려 쓴다.
  * parity_check: 같은 배치를 numpy 와 torch 로 돌려 정책 로짓·가치의 최대 절대 오차를 보고한다.

공개 API: available, device_name, build_model, to_tensors, load_from_numpy, export_to_numpy, fit_torch, parity_check
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .battlenet import N_MON, pack_batch

_CLS = None  # torch 모듈 클래스 캐시 (torch 를 늦게 import 하려고 클래스를 함수 안에서 만든다)


def available() -> bool:
    """torch 를 import 할 수 있으면 True."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def device_name() -> str:
    if not available():
        return "none (torch 없음)"
    import torch
    return f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- torch 모듈 (tinytf 거울)
def _classes():
    """tinytf 와 같은 구조의 torch 모듈 클래스들을 만들어 dict 로 돌려준다 (한 번만 생성)."""
    global _CLS
    if _CLS is not None:
        return _CLS
    import torch
    nn = torch.nn

    class MHA(nn.Module):
        """tinytf.MultiHeadSelfAttention: qkv 한 Linear(d,3d) → 마지막 축 3등분 → 헤드 분할, key_pad 는 softmax 전 -1e9."""

        def __init__(self, d, h, dropout):
            super().__init__()
            self.d, self.h, self.dh = d, h, d // h
            self.qkv, self.out, self.drop = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Dropout(dropout)

        def forward(self, x, key_pad=None):
            B, T, _ = x.shape
            q, k, v = (a.reshape(B, T, self.h, self.dh).transpose(1, 2) for a in self.qkv(x).chunk(3, -1))
            s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.dh))
            if key_pad is not None:
                s = s.masked_fill(key_pad[:, None, None, :], -1e9)
            p = self.drop(torch.softmax(s, -1))
            return self.out((p @ v).transpose(1, 2).reshape(B, T, self.d))

    class FFN(nn.Module):
        def __init__(self, d, d_ff, dropout):
            super().__init__()
            self.l1, self.act, self.drop, self.l2 = nn.Linear(d, d_ff), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(d_ff, d)

        def forward(self, x):
            return self.l2(self.drop(self.act(self.l1(x))))

    class Block(nn.Module):
        """pre-LN: x + Drop(MHA(LN1(x))); x + Drop(FFN(LN2(x)))."""

        def __init__(self, d, h, d_ff, dropout):
            super().__init__()
            self.ln1, self.attn, self.drop1 = nn.LayerNorm(d, eps=1e-5), MHA(d, h, dropout), nn.Dropout(dropout)
            self.ln2, self.ffn, self.drop2 = nn.LayerNorm(d, eps=1e-5), FFN(d, d_ff, dropout), nn.Dropout(dropout)

        def forward(self, x, key_pad=None):
            x = x + self.drop1(self.attn(self.ln1(x), key_pad))
            return x + self.drop2(self.ffn(self.ln2(x)))

    class TokenProjector(nn.Module):
        """타입별 Linear + 타입 임베딩, 0번 위치에 학습 CLS. 반환 (X [B,1+Σn,d], key_pad [B,T] True=패딩)."""

        def __init__(self, types, d_types, d):
            super().__init__()
            self.types, self.d = list(types), d
            self.proj = nn.ModuleDict({t: nn.Linear(int(d_types[t]), d) for t in self.types})
            self.type_emb = nn.Parameter(torch.zeros(len(self.types), d))
            self.cls = nn.Parameter(torch.zeros(d))

        def forward(self, inputs, masks=None):
            B = next(iter(inputs.values())).shape[0]
            parts, pads = [self.cls.expand(B, 1, self.d)], [torch.zeros(B, 1, dtype=torch.bool, device=self.cls.device)]
            for i, t in enumerate(self.types):
                x = inputs[t]
                parts.append(self.proj[t](x) + self.type_emb[i])
                m = masks.get(t) if masks is not None else None
                pads.append(~m if m is not None else torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device))
            return torch.cat(parts, 1), torch.cat(pads, 1)

    class Encoder(nn.Module):
        """TokenProjector → n_layers × Block → LayerNorm. 반환 (CLS [B,d], CLS 제외 마스크 평균 [B,d], H [B,T,d])."""

        def __init__(self, cfg):
            super().__init__()
            d, dff, p = cfg["d_model"], cfg.get("d_ff") or 4 * cfg["d_model"], cfg.get("dropout", 0.0)
            self.tok = TokenProjector(cfg["types"], cfg["d_types"], d)
            self.blocks = nn.ModuleList([Block(d, cfg["n_heads"], dff, p) for _ in range(cfg["n_layers"])])
            self.ln_f = nn.LayerNorm(d, eps=1e-5)

        def forward(self, inputs, masks=None):
            X, key_pad = self.tok(inputs, masks)
            for blk in self.blocks:
                X = blk(X, key_pad)
            H = self.ln_f(X)
            valid = (~key_pad).to(H.dtype)
            valid[:, 0] = 0.0
            cnt = valid.sum(1, keepdim=True).clamp(min=1.0)
            return H[:, 0], (H * valid[:, :, None]).sum(1) / cnt, H

    def mlp(sizes):
        layers = []
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.GELU(approximate="tanh"))
        return nn.Sequential(*layers)

    class TorchBattleNet(nn.Module):
        """BattleNet.forward 와 같은 순전파. 반환 (정책 로짓 z [B,K] (패딩 −1e9), 가치 로짓 [B] — σ 전)."""

        def __init__(self, set_cfg, bat_cfg, pol_sizes, val_sizes):
            super().__init__()
            self.set_enc, self.bat_enc = Encoder(set_cfg), Encoder(bat_cfg)
            self.pol, self.val = mlp(pol_sizes), mlp(val_sizes)

        def forward(self, b):
            B = b["actor"].shape[0]
            cls_s, _, _ = self.set_enc(b["set_in"], b["set_mask"])                          # [B*12, d_set]
            setvec = cls_s.reshape(B, N_MON, cls_s.shape[1]) * b["mon_mask"][:, :, None].to(cls_s.dtype)
            bat_in = {"mon": torch.cat([setvec, b["dyn"]], 2), "global": b["global"][:, None, :], "hist": b["hist"]}
            bat_mask = {"mon": b["mon_mask"], "global": None, "hist": b["hist_mask"]}
            cls_b, _, H = self.bat_enc(bat_in, bat_mask)                                     # H [B, 1+12+1+8, d_b]
            K = b["options"].shape[1]
            bi = torch.arange(B, device=H.device).repeat_interleave(K)
            actor = H[bi, 1 + b["actor"][bi]]
            tgt = b["opt_target"].reshape(-1)
            target = H[bi, 1 + tgt.clamp(0, N_MON - 1)] * (tgt >= 0).to(H.dtype)[:, None]
            X = torch.cat([cls_b.repeat_interleave(K, 0), actor, target,
                           b["opt_move"].reshape(B * K, -1), b["options"].reshape(B * K, -1)], 1)
            z = self.pol(X)[:, 0].reshape(B, K)
            z = torch.where(b["opt_mask"], z, torch.full_like(z, -1e9))
            return z, self.val(cls_b)[:, 0]

    _CLS = {"MHA": MHA, "FFN": FFN, "Block": Block, "TokenProjector": TokenProjector, "Encoder": Encoder,
            "TorchBattleNet": TorchBattleNet, "mlp": mlp}
    return _CLS


def build_model(net_np):
    """numpy BattleNet 의 cfg(set_enc.cfg · bat_enc.cfg · MLP 층 모양)로 같은 구조의 torch 모델을 만든다 (가중치는 초기값)."""
    C = _classes()
    pol_sizes = [net_np.pol.layers[0].W.shape[0]] + [l.W.shape[1] for l in net_np.pol.layers]
    val_sizes = [net_np.val.layers[0].W.shape[0]] + [l.W.shape[1] for l in net_np.val.layers]
    return C["TorchBattleNet"](net_np.set_enc.cfg, net_np.bat_enc.cfg, pol_sizes, val_sizes)


def to_tensors(batch: dict, device):
    """battlenet.pack_batch 결과(numpy) → 같은 키 구조의 torch 텐서 (bool→bool, int64→long, float32→float)."""
    import torch

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)
    out = {k: ({kk: t(vv) for kk, vv in v.items()} if isinstance(v, dict) else t(v)) for k, v in batch.items()}
    return out


# ----------------------------------------------------------------------------- 가중치 복사
def _pairs(tm, net_np):
    """(torch 파라미터, numpy 배열, 전치 여부) 목록 — load/export 가 같은 대응표를 쓴다."""
    import torch
    out = []

    def lin(tl, nl):  # tinytf.Linear: W[d_in,d_out], b[d_out]
        out.append((tl.weight, nl.W, True))
        out.append((tl.bias, nl.b, False))

    def ln(tl, nl):   # tinytf.LayerNorm: g, b
        out.append((tl.weight, nl.g, False))
        out.append((tl.bias, nl.b, False))

    for te, ne in ((tm.set_enc, net_np.set_enc), (tm.bat_enc, net_np.bat_enc)):
        for t in ne.tok.types:
            lin(te.tok.proj[t], ne.tok.proj[t])
        out.append((te.tok.type_emb, ne.tok.type_emb, False))
        out.append((te.tok.cls, ne.tok.cls, False))
        for tb, nb in zip(te.blocks, ne.blocks):
            ln(tb.ln1, nb.ln1)
            lin(tb.attn.qkv, nb.attn.qkv)
            lin(tb.attn.out, nb.attn.out)
            ln(tb.ln2, nb.ln2)
            lin(tb.ffn.l1, nb.ffn.l1)
            lin(tb.ffn.l2, nb.ffn.l2)
        ln(te.ln_f, ne.ln_f)
    for tmlp, nmlp in ((tm.pol, net_np.pol), (tm.val, net_np.val)):
        for tl, nl in zip([m for m in tmlp if isinstance(m, torch.nn.Linear)], nmlp.layers):
            lin(tl, nl)
    assert len(out) == len(net_np.params()), f"파라미터 수 불일치 torch {len(out)} != numpy {len(net_np.params())}"
    return out


def load_from_numpy(tm, net_np):
    """numpy BattleNet 의 가중치를 torch 모델에 복사한다 (in-place)."""
    import torch
    with torch.no_grad():
        for tp, arr, tr in _pairs(tm, net_np):
            a = arr.T if tr else arr
            assert tuple(tp.shape) == a.shape, f"모양 불일치 {tuple(tp.shape)} != {a.shape}"
            tp.copy_(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)))
    return tm


def export_to_numpy(tm, net_np):
    """torch 모델의 가중치를 numpy BattleNet 의 배열에 되돌려 쓴다 (배열 객체를 바꾸지 않고 값만 대입)."""
    for tp, arr, tr in _pairs(tm, net_np):
        a = tp.detach().cpu().float().numpy()
        arr[...] = (a.T if tr else a).astype(arr.dtype)
    return net_np


# ----------------------------------------------------------------------------- 학습 · 평가
def _targets(data, idx, K):
    """정답 분포 P [B,K](패딩 0) 와 승패 Y [B]."""
    P = np.zeros((len(idx), K), np.float32)
    for k, i in enumerate(idx):
        P[k, :len(data["P"][i])] = data["P"][i]
    return P, data["Y"][idx].astype(np.float32)


def _evaluate(model, data, idx, d_mv, dev, batch=64) -> dict:
    """battlenet.evaluate 와 같은 지표 (policyCE·top1·valueLoss·valueAcc·n) 를 torch 순전파로 계산."""
    import torch
    model.eval()
    ce = top1 = vl = acc = 0.0
    idx = np.asarray(idx)
    with torch.no_grad():
        for s0 in range(0, len(idx), batch):
            ii = idx[s0:s0 + batch]
            b = pack_batch([data["encs"][i] for i in ii], d_mv)
            z, vz = model(to_tensors(b, dev))
            z = z.float().cpu().numpy()
            v = torch.sigmoid(vz.float().clamp(-30, 30)).cpu().numpy()
            zz = z - z.max(1, keepdims=True)
            p = np.exp(zz) * b["opt_mask"]
            p = p / p.sum(1, keepdims=True)
            for k, i in enumerate(ii):
                P = data["P"][i]
                pk = p[k, :len(P)]
                ce += -float(np.sum(P * np.log(np.clip(pk, 1e-8, 1))))
                top1 += float(np.argmax(pk) == np.argmax(P))
                y = data["Y"][i]
                vl += -float(y * math.log(max(v[k], 1e-6)) + (1 - y) * math.log(max(1 - v[k], 1e-6)))
                acc += float((v[k] > 0.5) == (y > 0.5))
    n = max(1, len(idx))
    return {"policyCE": ce / n, "top1": top1 / n, "valueLoss": vl / n, "valueAcc": acc / n, "n": int(n)}


def fit_torch(net_np, data, tr_idx, va_idx, epochs=20, lr=1e-3, batch=128, log=print, seed=0, val_weight=0.5,
              patience=4, device=None, amp=False) -> dict:
    """numpy BattleNet 을 warm start 로 torch 에서 학습하고, 가장 좋은 에폭의 가중치를 net_np 에 되돌려 쓴다.

    손실 = listwise 정책 CE(정답 분포 data['P'], opt_mask 패딩) + val_weight × 가치 BCE (battlenet.fit 과 동일),
    AdamW(wd 1e-4) · 기울기 노름 클립 5 · 검증 점수(정책 CE + val_weight × 가치 손실) 로 조기 종료.
    amp=True 면 CUDA 에서 bfloat16 autocast (기본 꺼짐). 돌려줌: battlenet.evaluate 지표 + device · epochs · bestEpoch."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_from_numpy(build_model(net_np).to(dev), net_np)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = bool(amp and dev.type == "cuda")
    tr = np.array(tr_idx)
    best, bad, best_ep, ran = (1e9, None), 0, 0, 0
    t0 = time.time()
    for ep in range(epochs):
        ran = ep + 1
        model.train()
        rng.shuffle(tr)
        tot = n = 0.0
        for s0 in range(0, len(tr), batch):
            idx = tr[s0:s0 + batch]
            b = pack_batch([data["encs"][i] for i in idx], net_np.d_mv)
            P, Y = _targets(data, idx, b["opt_mask"].shape[1])
            bt = to_tensors(b, dev)
            Pt, Yt = torch.from_numpy(P).to(dev), torch.from_numpy(Y).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                z, vz = model(bt)
            logp = torch.log_softmax(z.float(), -1)
            ce = -torch.where(bt["opt_mask"], Pt * logp, torch.zeros_like(logp)).sum(1).mean()   # 패딩 칸은 0 (0×−1e9 방지)
            vl = torch.nn.functional.binary_cross_entropy_with_logits(vz.float(), Yt)
            loss = ce + val_weight * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * len(idx)
            n += len(idx)
        m = _evaluate(model, data, va_idx, net_np.d_mv, dev)
        if log:
            log(f"    에폭 {ep + 1}: 학습 손실 {tot / max(1, n):.4f} · 검증 정책 손실 {m['policyCE']:.4f} · 정답 일치 {m['top1']:.1%}"
                f" · 승패 예측 {m['valueAcc']:.1%} · {time.time() - t0:.0f}초 [{dev}]")
        score = m["policyCE"] + val_weight * m["valueLoss"]
        if score < best[0] - 1e-4:
            best, bad, best_ep = (score, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})), 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
    export_to_numpy(model, net_np)
    out = _evaluate(model, data, va_idx, net_np.d_mv, dev)
    out.update({"device": str(dev), "epochs": ran, "bestEpoch": best_ep})
    return out


# ----------------------------------------------------------------------------- 동등성 검사
def _sample_encs(n=3):
    """battlenet.selfcheck 처럼 첫 자기대전 기록에서 결정 n개를 토큰화. (encs, vocab) 또는 None."""
    import glob
    from .dex import Dex
    from .selfplay import GAMES_DIR, iter_games
    from .statetok import StateTokenizer
    from .tokens import Vocab
    dex = Dex()
    vb = Vocab(dex)
    st = StateTokenizer(dex, vb)
    files = sorted(glob.glob(str(GAMES_DIR / "**" / "games_singles_*.jsonl.gz"), recursive=True))[:1]
    encs = []
    for g in iter_games(files):
        for d in g["decisions"]:
            if len(d.get("options") or []) >= 2:
                e = st.encode(d["view"], d["req"], g["teams"][d["side"]], d["options"])
                if e is not None:
                    encs.append(e)
            if len(encs) >= n:
                break
        break
    return (encs, vb) if len(encs) >= 2 else None


def parity_check(net_np=None, encs=None, device="cpu") -> dict:
    """numpy BattleNet → torch 로 가중치를 복사해 같은 packed 배치를 돌리고 정책 로짓(합법 칸)·가치의 최대 절대 오차를 보고한다.
    net_np 가 None 이면 battlenet.selfcheck 과 같은 작은 망을, encs 가 None 이면 첫 자기대전 기록의 결정 3개를 쓴다.
    export_to_numpy 왕복 뒤 numpy 순전파가 그대로인지도 본다. 기본 device='cpu' (CUDA TF32 오차를 피한다)."""
    if not available():
        return {"ok": False, "skipped": "torch 없음"}
    import torch
    from .battlenet import BattleNet
    if net_np is None or encs is None:
        s = _sample_encs(3)
        if s is None:
            return {"ok": True, "skipped": "기록 없음"}
        encs = encs or s[0]
        net_np = net_np or BattleNet(s[1].dims, d_set=16, d_battle=24, set_layers=1, battle_layers=1, heads=2, dropout=0.0)
    b = pack_batch(encs, net_np.d_mv)
    z_np, v_np = net_np.forward(b, train=False)
    dev = torch.device(device)
    model = load_from_numpy(build_model(net_np).to(dev), net_np).eval()
    with torch.no_grad():
        z_t, vz = model(to_tensors(b, dev))
    z_t = z_t.float().cpu().numpy()
    v_t = torch.sigmoid(vz.float().clamp(-30, 30)).cpu().numpy()
    m = b["opt_mask"]
    dz = float(np.abs(z_np[m] - z_t[m]).max())
    dv = float(np.abs(v_np - v_t).max())
    pad_ok = bool(np.all(z_t[~m] <= -1e8)) if (~m).any() else True
    export_to_numpy(model, net_np)                      # 왕복: torch → numpy 뒤 numpy 순전파가 같아야 함
    z2, v2 = net_np.forward(b, train=False)
    dr = max(float(np.abs(z2[m] - z_np[m]).max()), float(np.abs(v2 - v_np).max()))
    n_params = int(sum(p.numel() for p in model.parameters()))
    return {"ok": bool(dz < 1e-3 and dv < 1e-4 and dr < 1e-5 and pad_ok), "maxDiffPolicy": dz, "maxDiffValue": dv,
            "roundTrip": dr, "paddingMasked": pad_ok, "nParams": n_params, "B": len(encs), "K": int(m.shape[1]), "device": str(dev),
            "summary": f"numpy↔torch 정책 로짓 최대 오차 {dz:.2e} · 가치 {dv:.2e} · 왕복 {dr:.2e} ({n_params} 파라미터, {dev})"}


if __name__ == "__main__":
    print({"available": available(), "device": device_name()})
    print(parity_check())
