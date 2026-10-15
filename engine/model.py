# -*- coding: utf-8 -*-
"""학습 모델 — numpy 만으로 도는 작은 신경망 (PyTorch 불필요, 설치 가볍게).

두 가지를 학습한다.
  Q(상태, 행동) : 이 행동을 하면 이길 확률  → 행동별 예상 승률 (추천의 핵심)
  V(상태)       : 지금 상태의 승률            → 화면의 '현재 승률'

정답: 그 판을 실제로 이겼는가 (자기대전·실제 대전 결과). 판 단위로 학습/검증을 나눠 섞임을 막는다.
학습이 쌓일수록 Q 를 정책으로 써서 다시 자기대전 → 더 나은 데이터 → 다시 학습 (정책 반복).
"""
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dex import Dex
from . import features as F
from .policies import HeuristicPolicy

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


# ======================================================================== 데이터셋
def build_dataset(games, dex: Dex, game_type: str | None = None, max_games=None, log=print,
                  include_teacher=True):
    """기록된 판 → (X_state, X_option, y, game_idx, kind, teacher_prob). 모든 묶음을 다 계산해 둔다."""
    st, op = F.active_groups({"groups": {}})
    calc = HeuristicPolicy(dex)
    Xs, Xo, ys, gi, kinds, tp = [], [], [], [], [], []
    t0 = time.time()
    n = 0
    for g in games:
        if game_type and g.get("gameType") != game_type:
            continue
        if g.get("winner") not in ("p1", "p2"):
            continue
        n += 1
        if max_games and n > max_games:
            break
        calc.beliefs = {}
        for d in g["decisions"]:
            s = d["side"]
            if d["chosen"] is None or not d["options"]:
                continue
            x = F.FCtx(dex, d["view"], d["req"], g["teams"][s], calc)
            sv = F.state_vector(x, st)
            ov = F.option_vector(x, d["options"][d["chosen"]], op)
            Xs.append(sv)
            Xo.append(ov)
            ys.append(1.0 if g["winner"] == s else 0.0)
            gi.append(n)
            kinds.append({"preview": 0, "force": 1, "turn": 2}[x.kind])
            tp.append((d.get("probs") or [None] * len(d["options"]))[d["chosen"]] or 0.0)
        if n % 200 == 0:
            log(f"  변수 계산 {n}판 · {len(ys)}개 결정 · {time.time() - t0:.0f}초")
    meta = {"state_groups": [(g.key, g.size) for g in st], "option_groups": [(g.key, g.size) for g in op],
            "games": n}
    return (np.array(Xs, dtype=np.float32), np.array(Xo, dtype=np.float32), np.array(ys, dtype=np.float32),
            np.array(gi), np.array(kinds), np.array(tp, dtype=np.float32), meta)


def column_mask(meta, exclude=()):
    """묶음 제외용 열 선택 (상태 + 선택지 두 자리)."""
    cols_s, cols_o = [], []
    i = 0
    for k, sz in meta["state_groups"]:
        if k not in exclude:
            cols_s += list(range(i, i + sz))
        i += sz
    part = sum(sz for _, sz in meta["option_groups"])
    for slot in range(2):
        i = slot * part
        for k, sz in meta["option_groups"]:
            if k not in exclude:
                cols_o += list(range(i, i + sz))
            i += sz
    return np.array(cols_s, dtype=int), np.array(cols_o, dtype=int)


# ======================================================================== 신경망
class MLP:
    def __init__(self, d_in, hidden=(128, 64), seed=0, l2=1e-4):
        rng = np.random.default_rng(seed)
        sizes = [d_in, *hidden, 1]
        self.W = [rng.normal(0, math.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.l2 = l2
        self.mean = np.zeros(d_in, dtype=np.float32)
        self.std = np.ones(d_in, dtype=np.float32)

    def _fwd(self, X):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            acts.append(h)
        return acts

    def predict(self, X):
        X = np.clip((np.asarray(X, dtype=np.float32) - self.mean) / self.std, -6, 6)
        z = self._fwd(X)[-1][:, 0]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, Xv, yv, epochs=40, lr=1e-3, batch=256, patience=5, seed=0, w=None, log=None):
        self.mean = X.mean(0)
        sd = X.std(0)
        # 학습 데이터에서 거의 변하지 않은 칸은 배율 1 (앱 입력에서 튀는 값이 폭주하지 않게)
        self.std = np.where(sd < 1e-2, 1.0, sd).astype(np.float32)
        Xn = np.clip((X - self.mean) / self.std, -6, 6)
        rng = np.random.default_rng(seed)
        mW = [np.zeros_like(W) for W in self.W]
        vW = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        t = 0
        best, best_state, bad = 1e9, None, 0
        w = np.ones(len(y), dtype=np.float32) if w is None else w
        for ep in range(epochs):
            idx = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                bi = idx[s:s + batch]
                acts = self._fwd(Xn[bi])
                p = 1 / (1 + np.exp(-np.clip(acts[-1][:, 0], -30, 30)))
                g = ((p - y[bi]) * w[bi] / len(bi))[:, None]
                t += 1
                for li in range(len(self.W) - 1, -1, -1):
                    gW = acts[li].T @ g + self.l2 * self.W[li]
                    gb = g.sum(0)
                    if li > 0:
                        g = (g @ self.W[li].T) * (acts[li] > 0)
                    for P, G, M, V in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                        M[li] = 0.9 * M[li] + 0.1 * G
                        V[li] = 0.999 * V[li] + 0.001 * G * G
                        mh = M[li] / (1 - 0.9 ** t)
                        vh = V[li] / (1 - 0.999 ** t)
                        P[li] -= lr * mh / (np.sqrt(vh) + 1e-8)
            vl = logloss(yv, self.predict(Xv))
            if log:
                log(f"    에폭 {ep + 1}: 검증 손실 {vl:.4f}")
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = ([W.copy() for W in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            self.W, self.b = best_state
        return best

    def to_json(self):
        return {"W": [W.tolist() for W in self.W], "b": [b.tolist() for b in self.b],
                "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, d):
        m = cls.__new__(cls)
        m.W = [np.array(W, dtype=np.float32) for W in d["W"]]
        m.b = [np.array(b, dtype=np.float32) for b in d["b"]]
        m.mean = np.array(d["mean"], dtype=np.float32)
        m.std = np.array(d["std"], dtype=np.float32)
        m.l2 = 0
        return m


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(y, p):
    base = float(np.mean(y)) if len(y) else 0.5
    return {"logloss": round(logloss(y, p), 5), "baseline": round(logloss(y, np.full_like(p, base)), 5),
            "acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 5), "n": int(len(y))}


def split_by_game(gi, val_frac=0.2, seed=0):
    games = np.unique(gi)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    nv = max(1, int(len(games) * val_frac))
    val = set(games[:nv].tolist())
    mv = np.array([g in val for g in gi])
    return ~mv, mv


def train_q_v(data, exclude=(), seed=0, log=None, epochs=40):
    Xs, Xo, y, gi, kinds, tp, meta = data
    cs, co = column_mask(meta, exclude)
    X = np.concatenate([Xs[:, cs], Xo[:, co]], 1)
    tr, va = split_by_game(gi, seed=seed)
    q = MLP(X.shape[1], seed=seed)
    q.fit(X[tr], y[tr], X[va], y[va], seed=seed, log=log, epochs=epochs)
    v = MLP(len(cs), hidden=(64, 32), seed=seed)
    v.fit(Xs[tr][:, cs], y[tr], Xs[va][:, cs], y[va], seed=seed, epochs=epochs)
    return q, v, {"q": metrics(y[va], q.predict(X[va])), "v": metrics(y[va], v.predict(Xs[va][:, cs]))}


# ======================================================================== 저장 · 불러오기
def save(game_type, q, v, meta, exclude, mets, dex: Dex, extra=None):
    MODELS.mkdir(exist_ok=True)
    st = [(k, s) for k, s in meta["state_groups"] if k not in exclude]
    op = [(k, s) for k, s in meta["option_groups"] if k not in exclude]
    out = {
        "gameType": game_type, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataVersion": dex.meta.get("builtAt"), "state_groups": st, "option_groups": op,
        "games": meta["games"], "metrics": mets, "q": q.to_json(), "v": v.to_json(), **(extra or {}),
    }
    p = MODELS / f"{game_type}.json"
    p.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return p


class Model:
    def __init__(self, path: Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.info = {k: v for k, v in d.items() if k not in ("q", "v")}
        self.q = MLP.from_json(d["q"])
        self.v = MLP.from_json(d["v"])
        keys_s = {k for k, _ in d["state_groups"]}
        keys_o = {k for k, _ in d["option_groups"]}
        self.st = [g for g in F.REGISTRY if g.key in keys_s]
        self.op = [g for g in F.PART_GROUPS if g.key in keys_o]
        if [g.key for g in self.st] != [k for k, _ in d["state_groups"]] or \
                [g.key for g in self.op] != [k for k, _ in d["option_groups"]]:
            raise ValueError("모델의 변수 묶음이 지금 등록부와 다릅니다 — 다시 학습이 필요합니다.")

    def evaluate(self, x: F.FCtx, options: list):
        sv = F.state_vector(x, self.st)
        rows = [sv + F.option_vector(x, o, self.op) for o in options]
        q = self.q.predict(np.array(rows, dtype=np.float32)) if rows else np.array([])
        v = float(self.v.predict(np.array([sv], dtype=np.float32))[0])
        return q, v


def load(game_type) -> Model | None:
    p = MODELS / f"{game_type}.json"
    if not p.is_file():
        return None
    try:
        return Model(p)
    except Exception:
        return None


class ModelPolicy:
    """선생님 기본기 + 학습 보정.

    점수 = 선생님 점수 / T  +  beta × logit(Q(상태, 행동))
    beta 는 학습 때 리그 대결로 고른다 (보정이 도움이 안 되면 0 → 선생님과 같음).
    """
    name = "model"

    def __init__(self, dex: Dex, model: Model, temperature=0.03, seed=None, beta=None):
        self.dex, self.m, self.T = dex, model, temperature
        self.beta = model.info.get("beta", 1.0) if beta is None else beta
        self.rng = random.Random(seed)
        self.calc = HeuristicPolicy(dex)

    def start(self, side, team_sets, dex):
        self.sets = team_sets
        self.calc.start(side, team_sets, dex)

    def scores(self, view, req, options):
        from .selfplay import compact_req
        snap = view.snapshot() if hasattr(view, "snapshot") else view
        teacher = (self.calc._preview_scores(snap, req, options) if req.get("teamPreview")
                   else self.calc._turn_scores(snap, req, options))
        x = F.FCtx(self.dex, snap, compact_req(req), self.sets, self.calc)
        q, v = self.m.evaluate(x, [o["choice"] for o in options])
        qc = np.clip(q, 1e-4, 1 - 1e-4)
        comb = np.array(teacher) / 0.12 + self.beta * np.log(qc / (1 - qc))
        return comb, q, v, teacher

    def choose(self, view, req, options):
        comb, q, _, _ = self.scores(view, req, options)
        # comb 는 선생님 점수/0.12 단위 → 온도 T 만큼 더 날카롭게(낮으면 거의 최선만)
        p = np.exp((comb - comb.max()) * (0.12 / max(self.T, 1e-3)))
        p /= p.sum()
        i = int(self.rng.choices(range(len(options)), weights=p.tolist())[0])
        return i, p.tolist()


# ======================================================================== 셀프테스트
def selfcheck(dex: Dex) -> dict:
    out = {"ok": True, "models": {}}
    msgs = []
    for gt in ("singles", "doubles"):
        p = MODELS / f"{gt}.json"
        if not p.is_file():
            msgs.append(f"{gt}: 모델 없음")
            continue
        try:
            m = Model(p)
        except Exception as e:
            out["ok"] = False
            msgs.append(f"{gt}: 불러오기 실패 — {e}")
            continue
        mq = m.info["metrics"]["q"]
        better = mq["logloss"] < mq["baseline"]
        arena = m.info.get("arena") or {}
        msgs.append(f"{gt}: {m.info['games']}판 학습 · 검증 손실 {mq['logloss']} (기준 {mq['baseline']})"
                    + (f" · 선생님 상대 승률 {arena.get('winrate', 0):.0%}" if arena else ""))
        if not better:
            out["ok"] = False
            msgs.append(f"{gt}: 모델이 기준선보다 못함")
        out["models"][gt] = {"metrics": m.info["metrics"], "arena": arena}
    out["summary"] = " / ".join(msgs)
    return out
