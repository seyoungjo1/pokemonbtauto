# -*- coding: utf-8 -*-
"""토큰 배틀망 — 요소 토큰 → 세트 인코더(공유) → 배틀 인코더 → 정책(선택지 점수)·가치 (ARCHITECTURE.md 2.2~2.3).

  세트 인코더 : [CLS, 종/능력치, 특성, 도구, 기술×4] 어텐션 → 세트 벡터 (팀·선출·배틀이 공유할 부분)
  배틀 인코더 : [CLS, 포켓몬 12 (세트 벡터 ⊕ 동적 상태), 전역, 역사 8] 어텐션
  정책 헤드   : 선택지 k 의 점수 = MLP([CLS, 행위자 토큰, 대상 토큰, 기술 하위 토큰, 선택지 변수]) → 합법 선택지 softmax
  가치 헤드   : σ(MLP(CLS))

numpy(engine/tinytf.py) 로 학습·추론이 모두 된다 (샌드박스·CPU). PC 의 GPU 학습은 같은 구조의 torch 판(engine/battlenet_torch.py)이
학습한 뒤 같은 JSON 으로 내보내므로 추론 코드는 하나다.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import tinytf as TF
from .dex import Dex
from .statetok import N_DYN, N_EVENT, N_GLOBAL, N_OPTION, StateTokenizer
from .tokens import Vocab

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SET_TYPES = ["mon", "ability", "item", "move"]
N_MON, N_HIST = 12, 8


class MLP:
    def __init__(self, sizes, rng, dtype=np.float32):
        self.layers = [TF.Linear(a, b, rng, dtype) for a, b in zip(sizes, sizes[1:])]
        self.acts = [TF.GELU() for _ in self.layers[:-1]]

    def forward(self, x):
        for l, a in zip(self.layers[:-1], self.acts):
            x = a.forward(l.forward(x))
        return self.layers[-1].forward(x)

    def backward(self, g):
        g = self.layers[-1].backward(g)
        for l, a in zip(reversed(self.layers[:-1]), reversed(self.acts)):
            g = l.backward(a.backward(g))
        return g

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def grads(self):
        return [p for l in self.layers for p in l.grads()]

    def to_json(self):
        return [[w.tolist() for w in l.params()] for l in self.layers]

    def load(self, d):
        for l, ws in zip(self.layers, d):
            for p, w in zip(l.params(), ws):
                p[...] = np.array(w, p.dtype)


class BattleNet:
    def __init__(self, vocab_dims: dict, d_set=64, d_battle=96, set_layers=2, battle_layers=2, heads=4, dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = {"vocab_dims": dict(vocab_dims), "d_set": d_set, "d_battle": d_battle, "set_layers": set_layers,
                    "battle_layers": battle_layers, "heads": heads, "dropout": dropout, "seed": seed, "version": 1}
        self.set_enc = TF.Encoder(SET_TYPES, {t: vocab_dims[t] for t in SET_TYPES}, d_set, set_layers, heads, dropout=dropout, seed=seed)
        self.bat_enc = TF.Encoder(["mon", "global", "hist"], {"mon": d_set + N_DYN, "global": N_GLOBAL, "hist": N_EVENT},
                                  d_battle, battle_layers, heads, dropout=dropout, seed=seed + 1)
        d_mv = vocab_dims["move"]
        self.pol = MLP([3 * d_battle + d_mv + N_OPTION, 128, 64, 1], rng)
        self.val = MLP([d_battle, 64, 1], rng)
        self.d_mv = d_mv
        self.info = {}

    # ------------------------------------------------------------ 파라미터
    def params(self):
        return self.set_enc.all_params() + self.bat_enc.all_params() + self.pol.params() + self.val.params()

    def grads(self):
        return self.set_enc.all_grads() + self.bat_enc.all_grads() + self.pol.grads() + self.val.grads()

    # ------------------------------------------------------------ 순전파 (배치)
    def forward(self, batch: dict, train=False):
        """batch: StateTokenizer.encode 결과 B개를 묶은 것 (pack_batch). 돌려줌: 선택지 점수 [B, K] (패딩 −inf), 승률 [B]."""
        B = len(batch["actor"])
        # 1) 세트 인코더: B×12 세트를 한 번에
        set_in = {t: batch["set_in"][t] for t in SET_TYPES}
        set_mask = batch["set_mask"]
        cls_s, _ = self.set_enc.forward(set_in, set_mask, train)            # [B*12, d_set]
        d_set = cls_s.shape[1]
        setvec = cls_s.reshape(B, N_MON, d_set) * batch["mon_mask"][:, :, None]
        # 2) 배틀 인코더
        mon_in = np.concatenate([setvec, batch["dyn"]], 2)
        bat_in = {"mon": mon_in, "global": batch["global"][:, None, :], "hist": batch["hist"]}
        bat_mask = {"mon": batch["mon_mask"], "global": np.ones((B, 1), bool), "hist": batch["hist_mask"]}
        cls_b, _ = self.bat_enc.forward(bat_in, bat_mask, train)             # [B, d_b]
        H = self.bat_enc.H                                                   # [B, 1+12+1+8, d_b]
        # 3) 정책: 선택지마다 [CLS, 행위자, 대상, 기술 토큰, 선택지 변수]
        K = batch["options"].shape[1]
        bi = np.repeat(np.arange(B), K)
        ki = np.tile(np.arange(K), B)
        actor = H[bi, 1 + batch["actor"][bi]]
        tgt_idx = batch["opt_target"].reshape(-1)
        target = H[bi, 1 + np.clip(tgt_idx, 0, N_MON - 1)] * (tgt_idx >= 0)[:, None]
        X = np.concatenate([np.repeat(cls_b, K, 0), actor, target, batch["opt_move"].reshape(B * K, -1),
                            batch["options"].reshape(B * K, -1)], 1)
        z = self.pol.forward(X)[:, 0].reshape(B, K)
        z = np.where(batch["opt_mask"], z, -1e9)
        v = self.val.forward(cls_b)[:, 0]
        self._cache = {"B": B, "K": K, "bi": bi, "ki": ki, "tgt_idx": tgt_idx, "H_shape": H.shape, "d_set": d_set,
                       "actor_pos": 1 + batch["actor"][bi], "target_pos": 1 + np.clip(tgt_idx, 0, N_MON - 1), "mon_mask": batch["mon_mask"]}
        return z, 1 / (1 + np.exp(-np.clip(v, -30, 30)))

    def backward(self, g_z, g_v):
        """g_z [B,K] (−∞ 칸은 0), g_v [B] (σ 전 기울기)."""
        c = self._cache
        B, K = c["B"], c["K"]
        gX = self.pol.backward(g_z.reshape(B * K, 1))
        d_b = c["H_shape"][2]
        gH = np.zeros(c["H_shape"], np.float32)
        g_cls = gX[:, :d_b].reshape(B, K, d_b).sum(1)
        np.add.at(gH, (c["bi"], c["actor_pos"]), gX[:, d_b:2 * d_b])
        np.add.at(gH, (c["bi"], c["target_pos"]), gX[:, 2 * d_b:3 * d_b] * (c["tgt_idx"] >= 0)[:, None])
        g_cls = g_cls + self.val.backward(g_v[:, None])
        g_in = self.bat_enc.backward(g_cls=g_cls, g_tokens=gH)
        g_mon = g_in["mon"]                                                  # [B, 12, d_set + N_DYN]
        g_set_cls = (g_mon[:, :, :c["d_set"]] * c["mon_mask"][:, :, None]).reshape(B * N_MON, c["d_set"])
        self.set_enc.backward(g_cls=g_set_cls)

    # ------------------------------------------------------------ 추론 (결정 하나)
    def predict(self, enc: dict, T=1.0):
        batch = pack_batch([enc], self.d_mv)
        z, v = self.forward(batch, train=False)
        z = z[0][: len(enc["options"])] / T
        p = np.exp(z - z.max())
        return p / p.sum(), float(v[0])

    # ------------------------------------------------------------ 저장
    def save(self, path, extra=None):
        d = {"kind": "battlenet", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cfg": self.cfg,
             "set_enc": self.set_enc.to_json(), "bat_enc": self.bat_enc.to_json(), "pol": self.pol.to_json(), "val": self.val.to_json(),
             "info": self.info, **(extra or {})}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = d["cfg"]
        m = cls(c["vocab_dims"], c["d_set"], c["d_battle"], c["set_layers"], c["battle_layers"], c["heads"], c["dropout"], c["seed"])
        m.set_enc = TF.Encoder.from_json(d["set_enc"])
        m.bat_enc = TF.Encoder.from_json(d["bat_enc"])
        m.pol.load(d["pol"])
        m.val.load(d["val"])
        m.info = d.get("info", {})
        m.file = d
        return m


# ======================================================================== 배치 묶기
def pack_batch(encs: list, d_mv: int) -> dict:
    B = len(encs)
    K = max(len(e["options"]) for e in encs)
    dims = None
    for e in encs:
        for s in e["sets"]:
            if s:
                dims = {t: len(s[t][0]) for t in ("mon", "ability", "item")}
                dims["move"] = len(s["moves"][0][0]) if s["moves"] else d_mv
                break
        if dims:
            break
    set_in = {"mon": np.zeros((B * N_MON, 1, dims["mon"]), np.float32), "ability": np.zeros((B * N_MON, 1, dims["ability"]), np.float32),
              "item": np.zeros((B * N_MON, 1, dims["item"]), np.float32), "move": np.zeros((B * N_MON, 4, dims["move"]), np.float32)}
    set_mask = {"mon": np.ones((B * N_MON, 1), bool), "ability": np.ones((B * N_MON, 1), bool), "item": np.ones((B * N_MON, 1), bool),
                "move": np.zeros((B * N_MON, 4), bool)}
    dyn = np.zeros((B, N_MON, N_DYN), np.float32)
    mon_mask = np.zeros((B, N_MON), bool)
    glob = np.zeros((B, N_GLOBAL), np.float32)
    hist = np.zeros((B, N_HIST, N_EVENT), np.float32)
    hmask = np.zeros((B, N_HIST), bool)
    options = np.zeros((B, K, N_OPTION), np.float32)
    opt_mask = np.zeros((B, K), bool)
    opt_move = np.zeros((B, K, dims["move"]), np.float32)
    opt_target = np.full((B, K), -1, np.int64)
    actor = np.zeros(B, np.int64)
    for b, e in enumerate(encs):
        for j, s in enumerate(e["sets"]):
            r = b * N_MON + j
            if not s:
                set_mask["move"][r, 0] = True   # 빈 자리도 토큰 하나는 있어야 softmax 가 정의됨
                continue
            for t in ("mon", "ability", "item"):
                set_in[t][r, 0] = s[t][0]
            for k, (vec, _) in enumerate(s["moves"][:4]):
                set_in["move"][r, k] = vec
                set_mask["move"][r, k] = True
            if not s["moves"]:
                set_mask["move"][r, 0] = True
        dyn[b], mon_mask[b], glob[b] = e["dyn"], e["mon_mask"], e["global"]
        hist[b], hmask[b] = e["hist"], e["hist_mask"]
        n = len(e["options"])
        options[b, :n] = e["options"]
        opt_mask[b, :n] = True
        opt_target[b, :n] = e["opt_target"]
        for k, mv in enumerate(e["opt_move"]):
            if mv is not None:
                opt_move[b, k] = mv
        actor[b] = e["actor"]
    if not hmask.any():
        hmask[:, 0] = True
    return {"set_in": set_in, "set_mask": set_mask, "dyn": dyn, "mon_mask": mon_mask, "global": glob, "hist": hist, "hist_mask": hmask,
            "options": options, "opt_mask": opt_mask, "opt_move": opt_move, "opt_target": opt_target, "actor": actor}


# ======================================================================== 데이터
TEACHERS = ("search", "human", "heuristic")


def build_data(games, dex: Dex, game_type: str, max_games=None, log=print, teachers=TEACHERS, tokenizer=None):
    """판 기록 → 결정마다 (토큰 묶음, 정답 분포, 승패, 판 번호)."""
    st = tokenizer or StateTokenizer(dex, game_type=game_type)
    encs, P, Y, G, K = [], [], [], [], []
    n = 0
    t0 = time.time()
    for g in games:
        if g.get("gameType") != game_type or g.get("winner") not in ("p1", "p2"):
            continue
        n += 1
        if max_games and n > max_games:
            break
        pols = g.get("policies") or {}
        for d in g["decisions"]:
            opts = d.get("options") or []
            if len(opts) < 2 or d.get("chosen") is None:
                continue
            if teachers and pols.get(d["side"], "human").split(":")[0] not in teachers:
                continue
            try:
                e = st.encode(d["view"], d["req"], g["teams"][d["side"]], opts)
            except Exception:
                continue
            if e is None:
                continue
            pr = d.get("probs")
            p = np.array(pr, np.float32) if pr and len(pr) == len(opts) else np.eye(len(opts), dtype=np.float32)[d["chosen"]]
            encs.append(e)
            P.append(p / max(1e-9, p.sum()))
            Y.append(1.0 if g["winner"] == d["side"] else 0.0)
            G.append(n)
            K.append({"preview": 0, "force": 1, "turn": 2}[d["kind"]])
        if n % 200 == 0:
            log(f"  토큰화 {n}판 · {len(encs)}개 결정 · {time.time() - t0:.0f}초")
    return {"encs": encs, "P": P, "Y": np.array(Y, np.float32), "G": np.array(G), "K": np.array(K), "games": n}


def split(data, val=0.15, seed=0):
    games = np.unique(data["G"])
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    vs = set(games[:max(1, int(len(games) * val))].tolist())
    va = np.array([g in vs for g in data["G"]])
    return np.where(~va)[0], np.where(va)[0]


# ======================================================================== 학습 (numpy)
def fit(net: BattleNet, data, tr_idx, va_idx, epochs=6, lr=1e-3, batch=32, log=print, seed=0, val_weight=0.5, patience=3):
    rng = np.random.default_rng(seed)
    opt = TF.Adam(net.params(), lr=lr, wd=1e-4)
    tr = np.array(tr_idx)
    best = (1e9, None)
    bad = 0
    t0 = time.time()
    for ep in range(epochs):
        rng.shuffle(tr)
        tot = n = 0.0
        for s0 in range(0, len(tr), batch):
            idx = tr[s0:s0 + batch]
            b = pack_batch([data["encs"][i] for i in idx], net.d_mv)
            z, v = net.forward(b, train=True)
            # 정책: listwise CE
            zz = z - z.max(1, keepdims=True)
            p = np.exp(zz) * b["opt_mask"]
            p = p / p.sum(1, keepdims=True)
            P = np.zeros_like(p)
            for k, i in enumerate(idx):
                P[k, :len(data["P"][i])] = data["P"][i]
            ce = -(P * np.log(np.clip(p, 1e-8, 1))).sum(1).mean()
            y = data["Y"][idx]
            vl = -(y * np.log(np.clip(v, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - v, 1e-6, 1))).mean()
            g_z = (p - P) / len(idx)
            g_v = val_weight * (v - y) / len(idx)
            net.backward(g_z.astype(np.float32), g_v.astype(np.float32))
            grads = net.grads()
            TF.clip_grad_norm(grads, 5.0)
            opt.step(grads)
            tot += float(ce + val_weight * vl) * len(idx)
            n += len(idx)
        m = evaluate(net, data, va_idx)
        if log:
            log(f"    에폭 {ep + 1}: 학습 손실 {tot / max(1, n):.4f} · 검증 정책 손실 {m['policyCE']:.4f} · 정답 일치 {m['top1']:.1%} · 승패 예측 {m['valueAcc']:.1%} · {time.time() - t0:.0f}초")
        score = m["policyCE"] + val_weight * m["valueLoss"]
        if score < best[0] - 1e-4:
            best = (score, [p.copy() for p in net.params()])
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best[1]:
        for p, w in zip(net.params(), best[1]):
            p[...] = w
    return evaluate(net, data, va_idx)


def evaluate(net: BattleNet, data, idx, batch=64) -> dict:
    ce = top1 = vl = acc = 0.0
    idx = np.array(idx)
    for s0 in range(0, len(idx), batch):
        ii = idx[s0:s0 + batch]
        b = pack_batch([data["encs"][i] for i in ii], net.d_mv)
        z, v = net.forward(b, train=False)
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


# ======================================================================== 정책 (자기대전·탐색 사전확률·앱)
def path_for(game_type):
    return MODELS / f"battle_{game_type}.json"


def load(game_type):
    p = path_for(game_type)
    try:
        return BattleNet.load(p) if p.is_file() else None
    except Exception:
        return None


class TokenNetPolicy:
    name = "tokennet"

    def __init__(self, dex: Dex, net: BattleNet, game_type="singles", temperature=0.3, seed=None):
        import random
        self.dex, self.net, self.T = dex, net, temperature
        self.rng = random.Random(seed)
        self.st = StateTokenizer(dex, game_type=game_type)

    def start(self, side, team_sets, dex):
        self.sets = team_sets

    def probs(self, view, req, options):
        from .selfplay import compact_req
        snap = view.snapshot() if hasattr(view, "snapshot") else view
        e = self.st.encode(snap, compact_req(req), self.sets, [o["choice"] for o in options])
        p, v = self.net.predict(e, self.T)
        self.last_value = v
        return p, v

    def choose(self, view, req, options):
        p, _ = self.probs(view, req, options)
        i = int(self.rng.choices(range(len(options)), weights=p.tolist())[0])
        return i, p.tolist()


def selfcheck(dex: Dex | None = None) -> dict:
    """작은 망으로 순전파·역전파 모양과 기울기 검사(유한차분, 일부 파라미터)."""
    import glob
    from .selfplay import GAMES_DIR, iter_games
    dex = dex or Dex()
    vb = Vocab(dex)
    st = StateTokenizer(dex, vb)
    files = sorted(glob.glob(str(GAMES_DIR / "**" / "games_singles_*.jsonl.gz"), recursive=True))[:1]
    encs = []
    for g in iter_games(files):
        for d in g["decisions"]:
            if len(d.get("options") or []) >= 2:
                encs.append(st.encode(d["view"], d["req"], g["teams"][d["side"]], d["options"]))
            if len(encs) >= 3:
                break
        break
    if len(encs) < 2:
        return {"ok": True, "skipped": "기록 없음"}
    net = BattleNet(vb.dims, d_set=16, d_battle=24, set_layers=1, battle_layers=1, heads=2, dropout=0.0)
    b = pack_batch(encs, vb.dims["move"])
    z, v = net.forward(b)
    ok = z.shape == (len(encs), b["options"].shape[1]) and v.shape == (len(encs),)
    # 손실: 첫 선택지 log-softmax + 가치 → 유한차분 (몇 개 파라미터만)
    def loss():
        z, v = net.forward(b)
        zz = z - z.max(1, keepdims=True)
        p = np.exp(zz) * b["opt_mask"]
        p = p / p.sum(1, keepdims=True)
        return float(-np.log(p[:, 0] + 1e-9).sum() + (v ** 2).sum())
    z, v = net.forward(b)
    zz = z - z.max(1, keepdims=True)
    p = np.exp(zz) * b["opt_mask"]
    p = p / p.sum(1, keepdims=True)
    P = np.zeros_like(p)
    P[:, 0] = 1.0
    g_z = (p - P)
    g_v = 2 * v * v * (1 - v)   # d(v²)/dz = 2v·v(1−v)
    net.backward(g_z.astype(np.float32), g_v.astype(np.float32))
    grads = net.grads()
    params = net.params()
    rng = np.random.default_rng(0)
    worst = 0.0
    for pi in rng.choice(len(params), min(6, len(params)), replace=False):
        prm, grd = params[pi], grads[pi]
        if prm.size == 0:
            continue
        j = tuple(rng.integers(0, s) for s in prm.shape)
        old = prm[j]
        eps = 1e-2
        prm[j] = old + eps
        lp = loss()
        prm[j] = old - eps
        lm = loss()
        prm[j] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - grd[j]) / max(1e-3, abs(num) + abs(grd[j]))
        worst = max(worst, rel)
    ok = ok and worst < 0.2   # float32 + 큰 eps 라 느슨한 기준 (모양·부호 오류 잡기용)
    return {"ok": bool(ok), "gradRelErr": round(worst, 4), "summary": f"토큰 배틀망 순전파·역전파 정상 (기울기 상대오차 {worst:.3f})"}
