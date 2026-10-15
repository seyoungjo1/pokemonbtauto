"""tinytf — numpy 전용 소형 트랜스포머 인코더 (수동 역전파).

용도: "세트 인코더" — 가변 길이 서브토큰 집합(예: 포켓몬 = [종/스탯, 특성, 도구, 기술×(2~4)])
  + 학습되는 [CLS] 토큰 → 벡터 하나. 같은 구조로 6~12 포켓몬 벡터를 받는 "팀 인코더"도 만든다.
  torch 없이 CPU 에서 수천 샘플 규모 학습이 돌고, 나중에 torch 로 그대로 옮길 수 있도록
  수식은 표준(pre-LN, scaled dot-product, GELU(tanh), AdamW)만 쓴다.

규약
  * 각 모듈: forward(...) 가 필요한 값을 캐시, backward(grad) 가 입력 기울기를 돌려주고
    params()/grads() 는 같은 순서로 정렬된 배열 리스트. backward 는 기울기를 '대입'한다
    (누적 아님 — forward 1회당 backward 1회).
  * 마스크: key_pad[B, T] 불리언, True = 패딩. 마스크된 키는 softmax 전에 -1e9.
  * 입력 마스크(TokenProjector.masks)는 True = 유효 토큰 (key_pad 의 반대).
"""
from __future__ import annotations

import math
import time

import numpy as np

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_C = 0.044715


# ----------------------------------------------------------------------------- 기본 모듈
class Module:
    """파라미터를 가진 모듈의 공통 골격."""

    def params(self) -> list:
        return []

    def grads(self) -> list:
        return []

    def submodules(self) -> list:
        return []

    def all_params(self) -> list:
        out = list(self.params())
        for m in self.submodules():
            out += m.all_params()
        return out

    def all_grads(self) -> list:
        out = list(self.grads())
        for m in self.submodules():
            out += m.all_grads()
        return out


class Linear(Module):
    """y = x @ W + b. x 는 [..., d_in] (앞 차원은 자유)."""

    def __init__(self, d_in, d_out, rng, dtype=np.float32, scale=None):
        s = scale if scale is not None else 1.0 / math.sqrt(d_in)
        self.W = rng.normal(0, s, (d_in, d_out)).astype(dtype)
        self.b = np.zeros(d_out, dtype=dtype)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        x2 = self.x.reshape(-1, self.W.shape[0])
        g2 = g.reshape(-1, self.W.shape[1])
        self.gW = x2.T @ g2
        self.gb = g2.sum(0)
        return g @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.gW, self.gb]


class LayerNorm(Module):
    """마지막 축 정규화. eps=1e-5 (torch 기본값과 동일)."""

    def __init__(self, d, dtype=np.float32, eps=1e-5):
        self.g = np.ones(d, dtype=dtype)
        self.b = np.zeros(d, dtype=dtype)
        self.eps = eps
        self.gg = np.zeros_like(self.g)
        self.gb = np.zeros_like(self.b)

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        self.xh = (x - mu) * self.rstd
        return self.g * self.xh + self.b

    def backward(self, g):
        self.gg = (g * self.xh).reshape(-1, self.g.shape[0]).sum(0)
        self.gb = g.reshape(-1, self.g.shape[0]).sum(0)
        gx = g * self.g
        m1 = gx.mean(-1, keepdims=True)
        m2 = (gx * self.xh).mean(-1, keepdims=True)
        return self.rstd * (gx - m1 - self.xh * m2)

    def params(self):
        return [self.g, self.b]

    def grads(self):
        return [self.gg, self.gb]


class GELU(Module):
    """GELU tanh 근사 (torch nn.GELU(approximate='tanh') 와 동일).
    gelu(x) = 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))"""

    def forward(self, x):
        self.x = x
        u = _SQRT_2_OVER_PI * (x + _GELU_C * x ** 3)
        self.t = np.tanh(u)
        return 0.5 * x * (1.0 + self.t)

    def backward(self, g):
        x, t = self.x, self.t
        du = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_C * x * x)
        return g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)


class Dropout(Module):
    """inverted dropout. train=False 또는 p=0 이면 항등."""

    def __init__(self, p=0.0, rng=None):
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.mask = None

    def forward(self, x, train=False):
        if not train or self.p <= 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
        return x * self.mask

    def backward(self, g):
        return g if self.mask is None else g * self.mask


# ----------------------------------------------------------------------------- 어텐션 / FFN / 블록
class MultiHeadSelfAttention(Module):
    """표준 scaled dot-product 멀티헤드 셀프어텐션 (배치·헤드 벡터화).
    key_pad[B, T] 불리언(True=패딩) 이면 해당 키의 점수를 -1e9 로 두고 softmax 한다."""

    def __init__(self, d_model, n_heads, rng, dtype=np.float32, dropout=0.0):
        assert d_model % n_heads == 0, "d_model 은 n_heads 의 배수여야 한다"
        self.d, self.h = d_model, n_heads
        self.dh = d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model, rng, dtype)
        self.out = Linear(d_model, d_model, rng, dtype)
        self.drop = Dropout(dropout, rng)

    def _split(self, x):  # [B,T,d] -> [B,H,T,dh]
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)

    def _merge(self, x):  # [B,H,T,dh] -> [B,T,d]
        B, H, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)

    def forward(self, x, key_pad=None, train=False):
        B, T, d = x.shape
        qkv = self.qkv.forward(x)
        q, k, v = (self._split(a) for a in np.split(qkv, 3, axis=-1))
        self.q, self.k, self.v = q, k, v
        s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.dh))
        if key_pad is not None:
            s = np.where(key_pad[:, None, None, :], x.dtype.type(-1e9), s)
        s = s - s.max(-1, keepdims=True)  # 수치 안정 softmax
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        self.p = p
        pd = self.drop.forward(p, train)
        ctx = pd @ v  # [B,H,T,dh]
        return self.out.forward(self._merge(ctx))

    def backward(self, g):
        g_ctx = self._split(self.out.backward(g))
        pd = self.p if self.drop.mask is None else self.p * self.drop.mask
        g_pd = g_ctx @ self.v.transpose(0, 1, 3, 2)  # [B,H,T,T]
        g_v = pd.transpose(0, 1, 3, 2) @ g_ctx
        g_p = self.drop.backward(g_pd)
        g_s = self.p * (g_p - (g_p * self.p).sum(-1, keepdims=True))  # softmax 역전파
        g_s *= 1.0 / math.sqrt(self.dh)  # 마스크 위치는 p≈0 이라 자동으로 0
        g_q = g_s @ self.k
        g_k = g_s.transpose(0, 1, 3, 2) @ self.q
        g_qkv = np.concatenate([self._merge(g_q), self._merge(g_k), self._merge(g_v)], axis=-1)
        return self.qkv.backward(g_qkv)

    def submodules(self):
        return [self.qkv, self.out]


class FeedForward(Module):
    """d → d_ff(기본 4d) → d, 활성 GELU(tanh)."""

    def __init__(self, d_model, rng, dtype=np.float32, d_ff=None, dropout=0.0):
        d_ff = d_ff or 4 * d_model
        self.l1 = Linear(d_model, d_ff, rng, dtype)
        self.act = GELU()
        self.drop = Dropout(dropout, rng)
        self.l2 = Linear(d_ff, d_model, rng, dtype)

    def forward(self, x, train=False):
        return self.l2.forward(self.drop.forward(self.act.forward(self.l1.forward(x)), train))

    def backward(self, g):
        return self.l1.backward(self.act.backward(self.drop.backward(self.l2.backward(g))))

    def submodules(self):
        return [self.l1, self.l2]


class EncoderBlock(Module):
    """pre-LN 블록: x + Drop(MHA(LN1(x))); x + Drop(FFN(LN2(x)))."""

    def __init__(self, d_model, n_heads, rng, dtype=np.float32, d_ff=None, dropout=0.0):
        self.ln1 = LayerNorm(d_model, dtype)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rng, dtype, dropout)
        self.drop1 = Dropout(dropout, rng)
        self.ln2 = LayerNorm(d_model, dtype)
        self.ffn = FeedForward(d_model, rng, dtype, d_ff, dropout)
        self.drop2 = Dropout(dropout, rng)

    def forward(self, x, key_pad=None, train=False):
        x = x + self.drop1.forward(self.attn.forward(self.ln1.forward(x), key_pad, train), train)
        x = x + self.drop2.forward(self.ffn.forward(self.ln2.forward(x), train), train)
        return x

    def backward(self, g):
        g = g + self.ln2.backward(self.ffn.backward(self.drop2.backward(g)))
        g = g + self.ln1.backward(self.attn.backward(self.drop1.backward(g)))
        return g

    def submodules(self):
        return [self.ln1, self.attn, self.ln2, self.ffn]


# ----------------------------------------------------------------------------- 토큰 투영 / 인코더
class TokenProjector(Module):
    """타입별로 폭이 다른 입력 토큰을 d_model 로 투영하고 타입 임베딩을 더한 뒤
    [CLS](학습 벡터) 를 0번 위치에 붙여 T 축으로 이어 붙인다.

    inputs: {type: [B, n_type, d_type]}, masks: {type: bool [B, n_type]} (True=유효, 없으면 전부 유효)
    반환: X [B, 1 + Σ n_type, d_model], key_pad [B, T] (True=패딩; CLS 는 항상 유효)
    `types` 순서가 곧 T 축 순서이므로 마스크·슬라이스가 그대로 맞물린다."""

    def __init__(self, types, d_types, d_model, rng, dtype=np.float32):
        self.types = list(types)
        self.d_types = {t: int(d_types[t]) for t in self.types}
        self.d = d_model
        self.proj = {t: Linear(self.d_types[t], d_model, rng, dtype) for t in self.types}
        self.type_emb = (rng.normal(0, 0.02, (len(self.types), d_model))).astype(dtype)
        self.cls = (rng.normal(0, 0.02, (d_model,))).astype(dtype)
        self.g_type_emb = np.zeros_like(self.type_emb)
        self.g_cls = np.zeros_like(self.cls)
        self.dtype = dtype

    def forward(self, inputs, masks=None):
        B = next(iter(inputs.values())).shape[0]
        parts = [np.broadcast_to(self.cls, (B, 1, self.d))]
        pads = [np.zeros((B, 1), dtype=bool)]
        self.slices = []
        pos = 1
        for i, t in enumerate(self.types):
            x = np.asarray(inputs[t], dtype=self.dtype)
            n = x.shape[1]
            parts.append(self.proj[t].forward(x) + self.type_emb[i])
            if masks is not None and t in masks and masks[t] is not None:
                pads.append(~np.asarray(masks[t], dtype=bool))
            else:
                pads.append(np.zeros((B, n), dtype=bool))
            self.slices.append((t, pos, pos + n))
            pos += n
        return np.concatenate(parts, axis=1), np.concatenate(pads, axis=1)

    def backward(self, gX):
        """gX [B,T,d] → 입력별 기울기 {type: [B,n_type,d_type]} (상위 인코더와 연결할 때 사용)."""
        self.g_cls = gX[:, 0].sum(0)
        g_in = {}
        for i, (t, a, b) in enumerate(self.slices):
            gs = gX[:, a:b]
            self.g_type_emb[i] = gs.reshape(-1, self.d).sum(0)
            g_in[t] = self.proj[t].backward(gs)
        return g_in

    def params(self):
        return [self.type_emb, self.cls]

    def grads(self):
        return [self.g_type_emb, self.g_cls]

    def submodules(self):
        return [self.proj[t] for t in self.types]


class Encoder(Module):
    """TokenProjector → n_layers × EncoderBlock → 최종 LayerNorm → 풀링.

    forward(inputs, masks, train) → (cls_out [B,d], mean_out [B,d])
      cls_out  : 0번 위치(CLS) 출력
      mean_out : CLS 를 제외한 유효 토큰의 마스크 평균
    backward(g_cls, g_mean) → 입력 기울기 dict (둘 중 쓰지 않는 쪽은 None 가능)."""

    def __init__(self, types, d_types, d_model, n_layers, n_heads, d_ff=None,
                 dropout=0.0, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.cfg = {"types": list(types), "d_types": {t: int(d_types[t]) for t in types},
                    "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                    "d_ff": d_ff or 4 * d_model, "dropout": dropout, "seed": seed,
                    "dtype": np.dtype(dtype).name}
        self.dtype = np.dtype(dtype)
        self.tok = TokenProjector(types, d_types, d_model, rng, self.dtype)
        self.blocks = [EncoderBlock(d_model, n_heads, rng, self.dtype, d_ff, dropout) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model, self.dtype)

    def forward(self, inputs, masks=None, train=False):
        X, key_pad = self.tok.forward(inputs, masks)
        for blk in self.blocks:
            X = blk.forward(X, key_pad, train)
        H = self.ln_f.forward(X)
        self.H, self.key_pad = H, key_pad
        valid = (~key_pad).astype(H.dtype)
        valid[:, 0] = 0.0  # 평균 풀링은 CLS 제외
        self.valid = valid
        self.cnt = np.maximum(valid.sum(1, keepdims=True), 1.0)  # [B,1]
        mean_out = (H * valid[:, :, None]).sum(1) / self.cnt
        return H[:, 0], mean_out

    def backward(self, g_cls=None, g_mean=None, g_tokens=None):
        """g_tokens [B,T,d]: 토큰별 출력(self.H)에 대한 기울기 (정책 헤드가 행위자·대상 토큰을 쓸 때)."""
        gH = np.zeros_like(self.H) if g_tokens is None else np.array(g_tokens, dtype=self.H.dtype, copy=True)
        if g_cls is not None:
            gH[:, 0] += g_cls
        if g_mean is not None:
            gH += (g_mean / self.cnt)[:, None, :] * self.valid[:, :, None]
        g = self.ln_f.backward(gH)
        for blk in reversed(self.blocks):
            g = blk.backward(g)
        return self.tok.backward(g)

    def submodules(self):
        return [self.tok] + self.blocks + [self.ln_f]

    # ---- 직렬화 (engine/nets.py 의 NpMLP 처럼 JSON 가능한 dict)
    def to_json(self):
        return {"cfg": self.cfg, "params": [p.tolist() for p in self.all_params()]}

    @classmethod
    def from_json(cls, d):
        c = d["cfg"]
        m = cls(c["types"], c["d_types"], c["d_model"], c["n_layers"], c["n_heads"],
                d_ff=c.get("d_ff"), dropout=c.get("dropout", 0.0), seed=c.get("seed", 0),
                dtype=np.dtype(c.get("dtype", "float32")))
        m.load_params(d["params"])
        return m

    def load_params(self, arrays):
        ps = self.all_params()
        assert len(ps) == len(arrays), f"파라미터 수 불일치 {len(ps)} != {len(arrays)}"
        for p, a in zip(ps, arrays):
            a = np.asarray(a, dtype=p.dtype)
            assert a.shape == p.shape, f"모양 불일치 {a.shape} != {p.shape}"
            p[...] = a


# ----------------------------------------------------------------------------- 최적화
class Adam:
    """AdamW(분리 weight decay). params 는 in-place 로 갱신된다."""

    def __init__(self, params, lr=1e-3, wd=0.0, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, self.wd, self.eps = list(params), lr, wd, eps
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1, c2 = 1.0 - self.b1 ** self.t, 1.0 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            if self.wd:
                p *= (1.0 - self.lr * self.wd)
            p -= (self.lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)).astype(p.dtype)


def clip_grad_norm(grads, max_norm):
    """전체 L2 노름이 max_norm 을 넘으면 in-place 로 축소. 클리핑 전 노름을 돌려준다."""
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    if total > max_norm and total > 0.0:
        s = max_norm / (total + 1e-6)
        for g in grads:
            g *= s
    return total


# ----------------------------------------------------------------------------- 자가 점검
def _make_batch(rng, types, d_types, n_types, B, pad=True, dtype=np.float64):
    inputs = {t: rng.normal(0, 1, (B, n_types[t], d_types[t])).astype(dtype) for t in types}
    masks = {}
    for t in types:
        m = np.ones((B, n_types[t]), dtype=bool)
        if pad and n_types[t] > 1:
            for b in range(B):
                k = int(rng.integers(1, n_types[t] + 1))  # 최소 1개는 유효
                m[b, k:] = False
        masks[t] = m
    return inputs, masks


def selfcheck(eps=1e-5, n_sample=20, seed=0):
    """float64 중앙차분으로 모든 파라미터 배열의 해석 기울기를 검사하고,
    float32 로 B=64, T=12, d_model=128, 2층 4헤드의 forward+backward 시간을 잰다."""
    rng = np.random.default_rng(seed)
    types, d_types, n_types = ["species", "ability", "move"], {"species": 5, "ability": 7, "move": 3}, \
        {"species": 1, "ability": 1, "move": 4}
    enc = Encoder(types, d_types, d_model=16, n_layers=2, n_heads=2, seed=seed, dtype=np.float64)
    inputs, masks = _make_batch(rng, types, d_types, n_types, B=3, dtype=np.float64)
    w = rng.normal(0, 1, (16,))

    def loss_fn():
        c, m = enc.forward(inputs, masks, train=False)
        return float((c * w).sum() + 0.5 * (m * m).sum()), c, m

    L, c, m = loss_fn()
    enc.backward(g_cls=np.broadcast_to(w, c.shape).copy(), g_mean=m.copy())
    params, grads = enc.all_params(), enc.all_grads()
    max_rel, worst = 0.0, None
    for pi, (p, g) in enumerate(zip(params, grads)):
        flat = p.reshape(-1)
        idx = rng.choice(flat.size, size=min(n_sample, flat.size), replace=False)
        for j in idx:
            old = flat[j]
            flat[j] = old + eps
            lp, _, _ = loss_fn()
            flat[j] = old - eps
            lm, _, _ = loss_fn()
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            ana = float(g.reshape(-1)[j])
            diff = abs(num - ana)
            # 절대 오차 1e-7 미만은 통과(예: k 바이어스처럼 해석적으로 0인 기울기 vs 차분 잡음)
            rel = 0.0 if diff < 1e-7 else diff / max(abs(num), abs(ana), 1e-6)
            if rel > max_rel:
                max_rel, worst = rel, (pi, int(j), ana, num)
    n_params = int(sum(p.size for p in params))

    # 직렬화 왕복 확인
    enc2 = Encoder.from_json(enc.to_json())
    c2, m2 = enc2.forward(inputs, masks)
    json_ok = bool(np.allclose(c2, c) and np.allclose(m2, m))

    # float32 속도: T = 1(CLS) + 11 토큰
    fenc = Encoder(["tok"], {"tok": 64}, d_model=128, n_layers=2, n_heads=4, seed=seed, dtype=np.float32)
    fin, fmask = _make_batch(rng, ["tok"], {"tok": 64}, {"tok": 11}, B=64, dtype=np.float32)
    g_c = rng.normal(0, 1, (64, 128)).astype(np.float32)
    times = []
    for _ in range(4):
        t0 = time.perf_counter()
        _, mm = fenc.forward(fin, fmask, train=True)
        fenc.backward(g_c, mm)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"ok": bool(max_rel < 1e-4 and json_ok), "maxRelErr": max_rel, "worst": worst,
            "params": n_params, "nArrays": len(params), "loss": L, "jsonRoundTrip": json_ok,
            "fwdBwdMs_B64_T12_d128": round(min(times[1:]), 2)}


if __name__ == "__main__":
    print(selfcheck())
