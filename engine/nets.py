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
# 정책 정답으로 삼을 만한 플레이어 (리그의 약한 상대 — 쇼다운 기본 AI·최대 데미지·무작위 — 의 선택은 따라 하지 않는다)
TEACHERS = ("search", "human", "heuristic")


def build_listwise(games, dex: Dex, game_type: str, max_games=None, log=print, target="probs", teachers=TEACHERS):
    """판 기록 → 결정마다 (상태 벡터, [선택지 벡터...], 정답 분포, 승패, 판 번호).

    target: 'probs' = 기록된 정책 분포(선생님·수읽기), 없으면 고른 행동 하나에 1.
    teachers: 이 정책들이 내린 결정만 쓴다 (None = 전부).
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
        pols = g.get("policies") or {}
        for d in g["decisions"]:
            opts = d.get("options") or []
            if len(opts) < 2 or d.get("chosen") is None:
                continue
            if teachers and pols.get(d["side"], "human").split(":")[0] not in teachers:
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
        best_p, best_v = (1e9, None), (1e9, None)   # 정책·가치 따로 가장 좋았던 때
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
                if m["policyCE"] < best_p[0] - 1e-4:
                    best_p = (m["policyCE"], self.pol.to_json())
                if m["valueLoss"] < best_v[0] - 1e-4:
                    best_v = (m["valueLoss"], self.val.to_json())
        if best_p[1]:
            self.pol = NpMLP.from_json(best_p[1])
        if best_v[1]:
            self.val = NpMLP.from_json(best_v[1])

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
        return {"policyCE": float(ce / n), "top1": float(top1 / n), "valueLoss": float(vl / n),
                "valueAcc": float(acc / n), "n": int(n)}

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


# ======================================================================== 신경망 수술 (패치·새 변수)
def _layout(groups):
    out, o = {}, 0
    for key, size in groups:
        out[key] = (o, int(size))
        o += int(size)
    return out, o


def surgery(old: "PolicyValueNet", st=None, op=None, log=print) -> "PolicyValueNet":
    """옛 신경망 → 지금 변수 등록부 모양의 신경망. 처음부터 다시 학습하지 않는다 (OpenAI Five 의 'surgery').

    - 이름이 같고 크기가 같은 변수 묶음은 첫 층 가중치·정규화 값을 그대로 옮긴다.
    - 새 묶음(또는 크기가 바뀐 묶음)의 칸은 첫 층 가중치를 0 으로 둔다 → 수술 직후 출력이 옛 신경망과 똑같다.
    - 없어진 묶음은 버린다.
    이어서 새 데이터 + 옛 데이터(리허설)로 미세조정하면 새 칸의 쓰임을 배운다.
    """
    if st is None or op is None:
        st, op = F.active_groups({"groups": {}})
    old_s, _ = _layout(old.file["state_groups"])
    new_s, ds = _layout([(g.key, g.size) for g in st])
    # 선택지 벡터 = 자리 2개(더블의 두 마리) × 묶음들
    o1, oh = _layout(old.file["option_groups"])
    n1, nh = _layout([(g.key, g.size) for g in op])
    old_o = {(slot, k): (slot * oh + o, sz) for slot in (0, 1) for k, (o, sz) in o1.items()}
    new_o = {(slot, k): (slot * nh + o, sz) for slot in (0, 1) for k, (o, sz) in n1.items()}
    do = 2 * nh

    def mapping(new_l, old_l):
        src = np.full(sum(sz for _, sz in new_l.values()), -1)
        kept, added = [], []
        for k, (o, sz) in new_l.items():
            if k in old_l and old_l[k][1] == sz:
                src[o:o + sz] = np.arange(old_l[k][0], old_l[k][0] + sz)
                kept.append(k)
            else:
                added.append(k)
        return src, kept, added
    ms_src, s_kept, s_added = mapping(new_s, old_s)
    mo_src, o_kept, o_added = mapping(new_o, old_o)
    o_kept = sorted({k for _, k in o_kept})
    o_added = sorted({k for _, k in o_added})
    net = PolicyValueNet.__new__(PolicyValueNet)
    net.ds, net.do = ds, do
    net.pol = NpMLP.from_json(old.pol.to_json())
    net.val = NpMLP.from_json(old.val.to_json())
    W0 = old.pol.W[0]
    h = W0.shape[1]
    Wp = np.zeros((ds + do, h), np.float32)
    for i, j in enumerate(ms_src):
        if j >= 0:
            Wp[i] = W0[j]
    for i, j in enumerate(mo_src):
        if j >= 0:
            Wp[ds + i] = W0[old.ds + j]
    net.pol.W[0] = Wp
    V0 = old.val.W[0]
    Wv = np.zeros((ds, V0.shape[1]), np.float32)
    for i, j in enumerate(ms_src):
        if j >= 0:
            Wv[i] = V0[j]
    net.val.W[0] = Wv
    net.ms = np.array([old.ms[j] if j >= 0 else 0.0 for j in ms_src], np.float32)
    net.ss = np.array([old.ss[j] if j >= 0 else 1.0 for j in ms_src], np.float32)
    net.mo = np.array([old.mo[j] if j >= 0 else 0.0 for j in mo_src], np.float32)
    net.so = np.array([old.so[j] if j >= 0 else 1.0 for j in mo_src], np.float32)
    net.new_s = np.where(ms_src < 0)[0]
    net.new_o = np.where(mo_src < 0)[0]
    net.st, net.op = list(st), list(op)
    net.info = dict(old.info or {})
    net.info["surgery"] = {"from": old.file.get("created"), "stateAdded": s_added, "optionAdded": o_added,
                           "stateKept": len(s_kept), "optionKept": len(o_kept)}
    net.file = {"created": old.file.get("created")}
    if log:
        log(f"  신경망 수술: 상태 묶음 {len(s_kept)}개 유지 + {len(s_added)}개 추가 {s_added} · "
            f"선택지 묶음 {len(o_kept)}개 유지 + {len(o_added)}개 추가 {o_added}")
    return net


def fit_new_norms(net, data, sample=20000):
    """수술로 새로 생긴 칸만 정규화 값을 데이터에서 정한다 (옛 칸은 그대로 → 옛 판단 보존)."""
    if len(getattr(net, "new_s", [])):
        S = np.stack(data["S"][:sample])[:, net.new_s]
        sd = S.std(0)
        net.ms[net.new_s] = S.mean(0)
        net.ss[net.new_s] = np.where(sd < 1e-2, 1.0, sd)
    if len(getattr(net, "new_o", [])):
        O = np.concatenate(data["O"][:sample])[:, net.new_o]
        od = O.std(0)
        net.mo[net.new_o] = O.mean(0)
        net.so[net.new_o] = np.where(od < 1e-2, 1.0, od)


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


def selfcheck_surgery(seed=0) -> dict:
    """셀프테스트: 변수 묶음 일부로 만든 신경망을 수술해도 같은 입력에 같은 출력을 내는가."""
    rng = np.random.default_rng(seed)
    st_all, op_all = F.active_groups({"groups": {}})
    st_old = [g for i, g in enumerate(st_all) if i % 3 != 1]
    op_old = [g for i, g in enumerate(op_all) if i % 2 == 0]
    ds, do1 = sum(g.size for g in st_old), sum(g.size for g in op_old)
    old = PolicyValueNet(ds, 2 * do1, hidden=(32, 16), vhidden=(16, 8), seed=seed)
    old.ms, old.ss = rng.normal(0, 1, ds).astype(np.float32), rng.uniform(0.5, 2, ds).astype(np.float32)
    old.mo, old.so = rng.normal(0, 1, 2 * do1).astype(np.float32), rng.uniform(0.5, 2, 2 * do1).astype(np.float32)
    old.file = {"state_groups": [(g.key, g.size) for g in st_old], "option_groups": [(g.key, g.size) for g in op_old],
                "created": "test"}
    new = surgery(old, st_all, op_all, log=None)

    def pick(groups_all, groups_old, v, slots=1):
        out = []
        half = sum(g.size for g in groups_all)
        for s in range(slots):
            o = s * half
            for g in groups_all:
                if any(x.key == g.key for x in groups_old):
                    out.extend(v[o:o + g.size])
                o += g.size
        return np.array(out, np.float32)
    worst = 0.0
    for _ in range(20):
        s_new = rng.normal(0, 2, new.ds).astype(np.float32)
        O_new = rng.normal(0, 2, (5, new.do)).astype(np.float32)
        s_old = pick(st_all, st_old, s_new)
        O_old = np.stack([pick(op_all, op_old, o, 2) for o in O_new])
        worst = max(worst, float(np.abs(old.policy(s_old, O_old) - new.policy(s_new, O_new)).max()),
                    abs(old.value(s_old) - new.value(s_new)))
    return {"ok": worst < 1e-4, "maxDiff": worst,
            "added": new.info["surgery"]["stateAdded"] + new.info["surgery"]["optionAdded"]}


# ======================================================================== 변수 묶음 제거 실험
def drop_group(data, key, st=None, op=None):
    """데이터에서 변수 묶음 하나의 칸을 뺀 사본 (상태 묶음이면 상태에서, 선택지 묶음이면 두 자리 모두에서)."""
    if st is None or op is None:
        st, op = F.active_groups({"groups": {}})
    s_lay, _ = _layout([(g.key, g.size) for g in st])
    o_lay, oh = _layout([(g.key, g.size) for g in op])
    out = dict(data)
    if key in s_lay:
        o, sz = s_lay[key]
        keep = np.r_[0:o, o + sz:len(data["S"][0])]
        out["S"] = [x[keep] for x in data["S"]]
    elif key in o_lay:
        o, sz = o_lay[key]
        full = 2 * oh
        drop = set(range(o, o + sz)) | set(range(oh + o, oh + o + sz))
        keep = np.array([i for i in range(full) if i not in drop])
        out["O"] = [x[:, keep] for x in data["O"]]
    return out


def ablation(data, keys, epochs=8, seeds=(0, 1), log=print):
    """묶음마다 빼고 처음부터 학습 → 검증 손실(정책 + 가치) 증가량. +면 그 묶음이 쓸모 있다."""
    tr, va = split(data, val=0.15, seed=99)

    def score(d, seed):
        net = PolicyValueNet(len(d["S"][0]), d["O"][0].shape[1], seed=seed)
        net.set_norm([d["S"][i] for i in tr[:20000]], [d["O"][i] for i in tr[:20000]])
        net.fit_numpy(d, epochs=epochs, tr_idx=tr.copy(), val_idx=va, log=None, seed=seed)
        m = net.evaluate(d, va)
        return m["policyCE"] + m["valueLoss"], m
    base = [score(data, s)[0] for s in seeds]
    b = float(np.mean(base))
    if log:
        log(f"  전체 변수: 검증 손실 {b:.4f} (시드 {len(seeds)}개)")
    res = {}
    for k in keys:
        d = drop_group(data, k)
        w = float(np.mean([score(d, s)[0] for s in seeds]))
        res[k] = {"with": round(b, 5), "without": round(w, 5), "delta": round(w - b, 5)}
        if log:
            log(f"  {k:12s} 빼면 {w - b:+.4f}")
    return b, res
