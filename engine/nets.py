# -*- coding: utf-8 -*-
"""정책·가치 신경망 (알파고 제로 방식의 손실).

  정책: 결정마다 '모든 선택지'에 점수를 매기고 softmax → 정답 분포(선생님·수읽기)와의 교차엔트로피
  가치: 상태 → 이길 확률, 정답은 그 판의 실제 승패 (이진 교차엔트로피)

구조
  점수(선택지 i) = MLP_p([상태 변수, 선택지 i 변수])      (같은 망을 선택지마다 공유)
  승률          = sigmoid(MLP_v(상태 변수))

학습 장치: PyTorch 가 있으면 GPU(CUDA)·CPU, 없으면 numpy.  저장은 항상 같은 JSON(가중치 목록)이라
앱은 numpy 만으로 불러 쓴다.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import features as F
from .dex import Dex
from .policies import HeuristicPolicy

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


# ======================================================================== 데이터
def build_listwise(games, dex: Dex, game_type: str, max_games=None, log=print, target="probs"):
    """판 기록 → 결정마다 (상태 벡터, [선택지 벡터...], 정답 분포, 승패, 판 번호).

    target: 'probs' = 기록된 정책 분포(선생님·수읽기), 없으면 고른 행동 하나에 1.
    """
    st, op = F.active_groups({"groups": {}})
    calc = HeuristicPolicy(dex)
    S, O, P, Y, G, K = [], [], [], [], [], []
    n = 0
    t0 = time.time()
    for g in games:
        if g.get("gameType") != game_type or g.get("winner") not in ("p1", "p2"):
            continue
        n += 1
        if max_games and n > max_games:
            break
        calc.beliefs = {}
        for d in g["decisions"]:
            opts = d.get("options") or []
            if len(opts) < 2 or d.get("chosen") is None:
                continue
            x = F.FCtx(dex, d["view"], d["req"], g["teams"][d["side"]], calc)
            S.append(np.array(F.state_vector(x, st), dtype=np.float32))
            O.append(np.array([F.option_vector(x, o, op) for o in opts], dtype=np.float32))
            pr = d.get("probs")
            if target == "probs" and pr and len(pr) == len(opts):
                p = np.array(pr, dtype=np.float32)
            else:
                p = np.zeros(len(opts), dtype=np.float32)
                p[d["chosen"]] = 1.0
            P.append(p / p.sum())
            Y.append(1.0 if g["winner"] == d["side"] else 0.0)
            G.append(n)
            K.append({"preview": 0, "force": 1, "turn": 2}[x.kind])
        if n % 200 == 0:
            log(f"  변수 계산 {n}판 · {len(S)}개 결정 · {time.time() - t0:.0f}초")
    meta = {"state_groups": [(g.key, g.size) for g in st], "option_groups": [(g.key, g.size) for g in op],
            "games": n, "gameType": game_type}
    return {"S": S, "O": O, "P": P, "Y": np.array(Y, dtype=np.float32), "G": np.array(G), "K": np.array(K), "meta": meta}


def split(data, val=0.15, seed=0):
    games = np.unique(data["G"])
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    vs = set(games[:max(1, int(len(games) * val))].tolist())
    va = np.array([g in vs for g in data["G"]])
    return np.where(~va)[0], np.where(va)[0]


# ======================================================================== numpy 신경망
class NpMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, math.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]

    def forward(self, X, keep=False):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            acts.append(h)
        return (h, acts) if keep else h

    def backward(self, acts, gout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = gout
        for li in range(len(self.W) - 1, -1, -1):
            gW[li] = acts[li].T @ g
            gb[li] = g.sum(0)
            if li > 0:
                g = (g @ self.W[li].T) * (acts[li] > 0)
        return gW, gb

    def params(self):
        return self.W + self.b

    def to_json(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_json(cls, d):
        m = cls.__new__(cls)
        m.W = [np.array(w, dtype=np.float32) for w in d["W"]]
        m.b = [np.array(b, dtype=np.float32) for b in d["b"]]
        return m


class Adam:
    def __init__(self, params, lr=1e-3, wd=1e-5):
        self.p, self.lr, self.wd = params, lr, wd
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            g = g + self.wd * p
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class PolicyValueNet:
    """추론은 numpy. 학습은 fit_numpy() 또는 nets_torch.fit_torch()."""

    def __init__(self, ds, do, hidden=(256, 128), vhidden=(128, 64), seed=0):
        self.ds, self.do = ds, do
        self.pol = NpMLP([ds + do, *hidden, 1], seed)
        self.val = NpMLP([ds, *vhidden, 1], seed + 1)
        self.ms = np.zeros(ds, np.float32)
        self.ss = np.ones(ds, np.float32)
        self.mo = np.zeros(do, np.float32)
        self.so = np.ones(do, np.float32)
        self.info = {}

    # ---- 정규화
    def set_norm(self, S_list, O_list):
        S = np.stack(S_list)
        O = np.concatenate(O_list)
        sd = S.std(0)
        od = O.std(0)
        self.ms, self.ss = S.mean(0), np.where(sd < 1e-2, 1.0, sd).astype(np.float32)
        self.mo, self.so = O.mean(0), np.where(od < 1e-2, 1.0, od).astype(np.float32)

    def ns(self, s):
        return np.clip((s - self.ms) / self.ss, -6, 6).astype(np.float32)

    def no(self, o):
        return np.clip((o - self.mo) / self.so, -6, 6).astype(np.float32)

    # ---- 추론
    def logits(self, s, O):
        sn = self.ns(np.asarray(s, np.float32))
        On = self.no(np.asarray(O, np.float32))
        X = np.concatenate([np.repeat(sn[None], len(On), 0), On], 1)
        return self.pol.forward(X)[:, 0]

    def policy(self, s, O, T=1.0):
        z = self.logits(s, O) / T
        z = np.exp(z - z.max())
        return z / z.sum()

    def value(self, s):
        z = self.val.forward(self.ns(np.asarray(s, np.float32))[None])[0, 0]
        return float(1 / (1 + math.exp(-max(-30, min(30, z)))))

    # ---- numpy 학습
    def fit_numpy(self, data, epochs=12, lr=1e-3, batch=64, log=print, seed=0, val_idx=None, tr_idx=None):
        rng = np.random.default_rng(seed)
        tr = tr_idx if tr_idx is not None else np.arange(len(data["S"]))
        opt_p = Adam(self.pol.params(), lr)
        opt_v = Adam(self.val.params(), lr)
        best = (1e9, None)
        for ep in range(epochs):
            rng.shuffle(tr)
            for s0 in range(0, len(tr), batch):
                idx = tr[s0:s0 + batch]
                # 정책: 결정들을 이어 붙여 한 번에
                rows, seg = [], []
                for k, i in enumerate(idx):
                    sn = self.ns(data["S"][i])
                    On = self.no(data["O"][i])
                    rows.append(np.concatenate([np.repeat(sn[None], len(On), 0), On], 1))
                    seg.append(len(On))
                X = np.concatenate(rows)
                z, acts = self.pol.forward(X, keep=True)
                z = z[:, 0]
                g = np.zeros_like(z)
                o = 0
                for k, i in enumerate(idx):
                    zz = z[o:o + seg[k]]
                    p = np.exp(zz - zz.max())
                    p /= p.sum()
                    g[o:o + seg[k]] = (p - data["P"][i]) / len(idx)
                    o += seg[k]
                gW, gb = self.pol.backward(acts, g[:, None])
                opt_p.step(gW + gb)
                # 가치
                Sv = np.stack([self.ns(data["S"][i]) for i in idx])
                zv, av = self.val.forward(Sv, keep=True)
                pv = 1 / (1 + np.exp(-np.clip(zv[:, 0], -30, 30)))
                gv = ((pv - data["Y"][idx]) / len(idx))[:, None]
                gW, gb = self.val.backward(av, gv)
                opt_v.step(gW + gb)
            if val_idx is not None:
                m = self.evaluate(data, val_idx)
                if log:
                    log(f"    에폭 {ep + 1}: 정책 손실 {m['policyCE']:.4f} · 정답 일치 {m['top1']:.1%} · 가치 손실 {m['valueLoss']:.4f}")
                score = m["policyCE"] + m["valueLoss"]
                if score < best[0] - 1e-4:
                    best = (score, (self.pol.to_json(), self.val.to_json()))
        if best[1]:
            self.pol = NpMLP.from_json(best[1][0])
            self.val = NpMLP.from_json(best[1][1])

    def evaluate(self, data, idx) -> dict:
        ce = top1 = vl = acc = 0.0
        for i in idx:
            p = self.policy(data["S"][i], data["O"][i])
            ce += -float(np.sum(data["P"][i] * np.log(np.clip(p, 1e-8, 1))))
            top1 += float(np.argmax(p) == np.argmax(data["P"][i]))
            v = self.value(data["S"][i])
            y = data["Y"][i]
            vl += -(y * math.log(max(v, 1e-6)) + (1 - y) * math.log(max(1 - v, 1e-6)))
            acc += float((v > 0.5) == (y > 0.5))
        n = max(1, len(idx))
        return {"policyCE": ce / n, "top1": top1 / n, "valueLoss": vl / n, "valueAcc": acc / n, "n": int(n)}

    # ---- 저장
    def save(self, path, meta, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {"kind": "policyvalue", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "ds": self.ds, "do": self.do, "pol": self.pol.to_json(), "val": self.val.to_json(),
             "norm": {"ms": self.ms.tolist(), "ss": self.ss.tolist(), "mo": self.mo.tolist(), "so": self.so.tolist()},
             "state_groups": meta["state_groups"], "option_groups": meta["option_groups"],
             "games": meta.get("games"), "info": self.info, **(extra or {})}
        path.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls.__new__(cls)
        m.ds, m.do = d["ds"], d["do"]
        m.pol, m.val = NpMLP.from_json(d["pol"]), NpMLP.from_json(d["val"])
        nrm = d["norm"]
        m.ms, m.ss = np.array(nrm["ms"], np.float32), np.array(nrm["ss"], np.float32)
        m.mo, m.so = np.array(nrm["mo"], np.float32), np.array(nrm["so"], np.float32)
        m.info = d.get("info", {})
        m.file = d
        keys_s = [k for k, _ in d["state_groups"]]
        keys_o = [k for k, _ in d["option_groups"]]
        m.st = [g for g in F.REGISTRY if g.key in keys_s]
        m.op = [g for g in F.PART_GROUPS if g.key in keys_o]
        if [g.key for g in m.st] != keys_s or [g.key for g in m.op] != keys_o:
            raise ValueError("신경망의 변수 묶음이 지금 등록부와 다릅니다 — 다시 학습이 필요합니다.")
        return m


def path_for(game_type):
    return MODELS / f"net_{game_type}.json"


def load_net(game_type):
    p = path_for(game_type)
    if not p.is_file():
        return None
    try:
        return PolicyValueNet.load(p)
    except Exception:
        return None


class NetPolicy:
    """신경망 정책 (정책망 확률대로, 낮은 온도면 거의 최선)."""
    name = "net"

    def __init__(self, dex: Dex, net: PolicyValueNet, temperature=0.3, seed=None):
        import random
        self.dex, self.net, self.T = dex, net, temperature
        self.rng = random.Random(seed)
        self.calc = HeuristicPolicy(dex)

    def start(self, side, team_sets, dex):
        self.sets = team_sets
        self.calc.start(side, team_sets, dex)

    def probs(self, view, req, options):
        from .selfplay import compact_req
        snap = view.snapshot() if hasattr(view, "snapshot") else view
        x = F.FCtx(self.dex, snap, compact_req(req), self.sets, self.calc)
        s = F.state_vector(x, self.net.st)
        O = [F.option_vector(x, o["choice"], self.net.op) for o in options]
        return self.net.policy(s, O, self.T), self.net.value(s)

    def choose(self, view, req, options):
        p, _ = self.probs(view, req, options)
        i = int(self.rng.choices(range(len(options)), weights=p.tolist())[0])
        return i, p.tolist()
