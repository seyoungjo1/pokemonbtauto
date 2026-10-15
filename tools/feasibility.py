#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1단계 가능성 시험 — 요소 토큰 어텐션 인코더 (ARCHITECTURE.md 1단계 합격 기준의 축소판).

  python tools/feasibility.py prep              존재 924개 토큰 + 경주 상성표 → userdata/feasibility/pairs.pkl (종 15% 홀드아웃)
  python tools/feasibility.py run [변형,...] [스텝]   변형: attn+phys+id / attn+phys / attn+id / bag  → encoder_feasibility.json

numpy 만으로 돈다(토치 없이). PC 에서는 스텝을 늘려 돌리면 된다.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "userdata" / "feasibility"
OUT.mkdir(parents=True, exist_ok=True)

def prep():

    import sys, time, pickle, numpy as np
    sys.path.insert(0, str(ROOT))
    from engine.dex import Dex
    from engine import universe as UV, tokens as T
    dex = Dex(); t0 = time.time()
    E, G = UV.universe(dex, 'singles', 3, log=None)
    vb = T.Vocab(dex)
    toks = [T.set_tokens(dex, e['set'], vb) for e in E]
    print(f'토큰 {len(toks)}개 · {time.time()-t0:.0f}초', flush=True)
    inputs, ids, mask, extra = T.batch_sets(toks, vb)
    species = [dex.base_of_mega(e['set']['species']) for e in E]
    uniq = sorted(set(species)); rng = np.random.default_rng(0); rng.shuffle(uniq)
    held = set(uniq[:int(len(uniq) * 0.15)])          # 종 15% 는 학습에서 완전히 뺀다
    is_held = np.array([s in held for s in species])
    pickle.dump({'inputs': inputs, 'ids': ids, 'mask': mask, 'extra': extra, 'G': G.astype(np.float32), 'species': species,
                 'held': is_held, 'ko': [e['ko'] for e in E], 'vocab': vb.to_json(), 'dims': vb.dims, 'sizes': vb.sizes},
                open(str(OUT / 'pairs.pkl'), 'wb'))
    print('홀드아웃 종', len(held), '존재', int(is_held.sum()), '/', len(E), f'{time.time()-t0:.0f}초')


import sys, time, json, pickle, numpy as np
sys.path.insert(0, str(ROOT))
from engine import tinytf as TF
D = None
TYPES = ['mon', 'ability', 'item', 'move']
EMB = 8
t0 = time.time()
log = lambda m: print(f'[{time.time()-t0:5.0f}s] {m}', flush=True)


class IdEmb:
    def __init__(self, n, d, rng):
        self.W = (rng.normal(0, 0.1, (n, d))).astype(np.float32)
        self.g = np.zeros_like(self.W)
    def forward(self, ids):
        self.ids = ids
        return self.W[ids]
    def backward(self, g):
        self.g[:] = 0
        np.add.at(self.g, self.ids, g)


class PairHead:
    """s(i,j) = tanh(f(h_i,h_j) − f(h_j,h_i)),  f = MLP([h_i, h_j, h_i∘h_j])."""
    def __init__(self, d, rng, hidden=64):
        self.l1 = TF.Linear(3 * d, hidden, rng); self.act = TF.GELU(); self.l2 = TF.Linear(hidden, 1, rng); self.d = d
    def params(self): return self.l1.params() + self.l2.params()
    def grads(self): return self.l1.grads() + self.l2.grads()
    def _f(self, A, B):
        return np.concatenate([A, B, A * B], 1)
    def forward(self, H, I, J):
        self.H, self.I, self.J = H, I, J
        X = np.concatenate([self._f(H[I], H[J]), self._f(H[J], H[I])], 0)
        self.z = self.l2.forward(self.act.forward(self.l1.forward(X)))[:, 0]
        n = len(I)
        self.raw = self.z[:n] - self.z[n:]
        self.out = np.tanh(self.raw)
        return self.out
    def backward(self, g_out):
        n = len(self.I)
        g_raw = g_out * (1 - self.out ** 2)
        gz = np.concatenate([g_raw, -g_raw], 0)[:, None]
        gX = self.l1.backward(self.act.backward(self.l2.backward(gz)))
        d = self.d
        gH = np.zeros_like(self.H)
        A1, B1 = self.H[self.I], self.H[self.J]
        gA = gX[:n, :d] + gX[:n, 2 * d:] * B1 + gX[n:, d:2 * d] + gX[n:, 2 * d:] * B1
        gB = gX[:n, d:2 * d] + gX[:n, 2 * d:] * A1 + gX[n:, :d] + gX[n:, 2 * d:] * A1
        np.add.at(gH, self.I, gA)
        np.add.at(gH, self.J, gB)
        return gH


def run(variant, steps=400, d_model=64, layers=2, heads=4, lr=1e-3, id_drop=0.3, seed=0, batch=8192):
    global D, G, held, N
    if D is None:
        D = pickle.load(open(OUT / 'pairs.pkl', 'rb'))
        G, held = D['G'], D['held']
        N = G.shape[0]
    rng = np.random.default_rng(seed)
    use_phys = variant != 'attn+id'
    use_id = variant in ('attn+phys+id', 'attn+id')
    n_layers = 0 if variant == 'bag' else layers
    d_types = {t: D['dims'][t] + (EMB if use_id else 0) for t in TYPES}
    enc = TF.Encoder(TYPES, d_types, d_model, n_layers, heads, dropout=0.1, seed=seed)
    head = PairHead(d_model, rng)
    embs = {t: IdEmb(D['sizes'][{'mon': 'species'}.get(t, t)], EMB, rng) for t in TYPES} if use_id else {}
    params = enc.all_params() + head.params() + [e.W for e in embs.values()]
    opt = TF.Adam(params, lr=lr, wd=1e-4)
    train_idx = np.where(~held)[0]
    hist = []

    def build_inputs(train):
        inp = {}
        for t in TYPES:
            x = D['inputs'][t] if use_phys else np.zeros_like(D['inputs'][t])
            if use_id:
                e = embs[t].forward(D['ids'][t])
                if train and id_drop > 0:
                    keep = (rng.random(e.shape[:2]) >= id_drop).astype(np.float32)[:, :, None]
                    e = e * keep
                    embs[t]._keep = keep
                else:
                    embs[t]._keep = np.ones(e.shape[:2] + (1,), np.float32)
                x = np.concatenate([x, e], 2)
            inp[t] = x
        return inp

    def encode(train):
        cls, mean = enc.forward(build_inputs(train), D['mask'], train=train)
        return cls if n_layers > 0 else mean

    def evaluate():
        H = encode(False)
        res = {}
        for name, cond in (('train', lambda i, j: ~held[i] & ~held[j]), ('one_held', lambda i, j: held[i] ^ held[j]),
                           ('both_held', lambda i, j: held[i] & held[j])):
            I = rng.integers(0, N, 60000); J = rng.integers(0, N, 60000)
            m = cond(I, J) & (I != J)
            I, J = I[m][:20000], J[m][:20000]
            y = G[I, J]; p = head.forward(H, I, J)
            res[name] = {'r2': float(1 - ((p - y) ** 2).mean() / y.var()), 'mse': float(((p - y) ** 2).mean()),
                         'sign': float(((p > 0) == (y > 0))[np.abs(y) > 0.2].mean()), 'n': int(len(I))}
        return res

    for step in range(1, steps + 1):
        H = encode(True)
        I = rng.choice(train_idx, batch); J = rng.choice(train_idx, batch)
        y = G[I, J]
        p = head.forward(H, I, J)
        loss = float(((p - y) ** 2).mean())
        gH = head.backward(2 * (p - y) / batch)
        if n_layers > 0:
            g_in = enc.backward(g_cls=gH)
        else:
            g_in = enc.backward(g_mean=gH)
        if use_id:
            for t in TYPES:
                embs[t].backward(g_in[t][:, :, -EMB:] * embs[t]._keep)
        grads = enc.all_grads() + head.grads() + [e.g for e in embs.values()]
        TF.clip_grad_norm(grads, 5.0)
        opt.step(grads)
        if step % 50 == 0 or step == steps:
            ev = evaluate()
            hist.append({'step': step, 'loss': loss, **{k: round(v['r2'], 3) for k, v in ev.items()}})
            log(f"{variant:14s} step {step:4d} loss {loss:.4f} | R² 학습종 {ev['train']['r2']:.3f} · 한쪽 홀드아웃 {ev['one_held']['r2']:.3f} · 양쪽 {ev['both_held']['r2']:.3f} | 부호 일치 {ev['train']['sign']:.1%}/{ev['one_held']['sign']:.1%}")
    return {'variant': variant, 'final': evaluate(), 'hist': hist, 'params': int(sum(p.size for p in params))}




if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if cmd == 'prep':
        prep()
    else:
        variants = sys.argv[2].split(',') if len(sys.argv) > 2 else ['attn+phys+id', 'attn+phys', 'attn+id', 'bag']
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
        out = []
        for v in variants:
            out.append(run(v, steps=steps))
            json.dump(out, open(OUT / 'encoder_feasibility.json', 'w'), ensure_ascii=False, indent=1)
        log('끝')
