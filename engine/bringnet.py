# -*- coding: utf-8 -*-
"""선출망 — 6→3(더블 6→4) 선출과 선봉을 '결과로' 배운다.

한 매치업(내 6마리 세트 + 상대 6마리 종)에서
  내 선출 i(선봉 포함) × 상대 선출 j  의 승률 M[i, j] 를 예측하고,
  그 표 전체(싱글 60×60, 더블 90×90)를 균형해(내시)로 풀어 선출을 정한다.

입력 변수는 포켓몬 이름이 아니라 데미지 엔진이 계산한 1:1 상성표 G(6×6)에서 뽑는다
(선봉끼리의 상성, 내 선출이 상대 선출을 얼마나 덮는지, 상대가 나를 얼마나 덮는지, 스피드, 메가 수…).
그래서 처음 보는 포켓몬·패치 후에도 같은 의미로 쓸 수 있다.

V(i, j) = σ(h(내 시점) − h(상대 시점)) — 자리를 바꾸면 정확히 1−V 가 되도록(반대칭) 만든다.

정답: 대전 기록마다 (내가 실제로 낸 선출, 상대가 낸 선출, 이겼는가). 자기대전·리플레이·나와 둔 대전 전부.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .belief import OppBelief
from .bring import Pairwise, orders, solve_matrix
from .dex import Dex, to_id
from .nets import Adam, NpMLP
from .teamgen import normalize_set

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FEAT_NAMES = ["lead_vs_lead", "lead_vs_their", "their_lead_vs_mine", "block_mean", "block_max", "block_min",
              "my_cover_mean", "my_cover_min", "their_cover_mean", "their_cover_max", "n_answered",
              "n_threatened", "lead_speed", "megas", "unbrought_best", "rest_vs_their",
              # 조합 시너지 (1:1 상성표로는 안 보이는 것): 설치자와 수혜자가 같이 나오는가
              "syn_weather", "syn_trickroom", "syn_tailwind", "syn_screens_setup", "intimidate", "fakeout_lead",
              "redirect", "pivots", "hazard_lead", "setup", "speed_ctrl_vs", "weather_clash",
              # 특성 시너지
              "syn_prankster", "syn_regen_pivot", "terrain_set", "ability_immune_cover"]

WEATHER_SET = {"drought": "sun", "drizzle": "rain", "sandstream": "sand", "snowwarning": "snow", "orichalcumpulse": "sun",
               "sunnyday": "sun", "raindance": "rain", "sandstorm": "sand", "snowscape": "snow"}
WEATHER_USE = {"chlorophyll": "sun", "solarpower": "sun", "flowergift": "sun", "protosynthesis": "sun",
               "swiftswim": "rain", "raindish": "rain", "dryskin": "rain", "sandrush": "sand", "sandforce": "sand",
               "sandveil": "sand", "slushrush": "snow", "icebody": "snow", "snowcloak": "snow"}
WEATHER_MOVE = {"solarbeam": "sun", "solarblade": "sun", "weatherball": "any", "hurricane": "rain", "thunder": "rain",
                "blizzard": "snow", "auroraveil": "snow"}
SETUP = {"swordsdance", "nastyplot", "dragondance", "calmmind", "quiverdance", "bulkup", "shellsmash", "coil",
         "agility", "irondefense", "curse", "tidyup", "victorydance", "shiftgear", "bellydrum"}


def roles(dex: Dex, s: dict) -> dict:
    """세트 하나의 역할 (조합 시너지 계산용)."""
    mv = set(s.get("moves") or [])
    ab = s.get("ability") or ""
    sp = dex.species.get(s.get("species"), {})
    mega = dex.mega_for(s.get("species", ""), s.get("item", ""))
    if mega:
        ab = to_id(dex.species[mega]["abilities"].get("0", "")) or ab
        sp = dex.species[mega]
    bs = sp.get("baseStats", {})
    sets_w = {WEATHER_SET[x] for x in mv | {ab} if x in WEATHER_SET}
    uses_w = {WEATHER_USE[ab]} if ab in WEATHER_USE else set()
    uses_w |= {WEATHER_MOVE[m] for m in mv if m in WEATHER_MOVE and WEATHER_MOVE[m] != "any"}
    # 날씨로 강해지는 타입: 비 = 물 기술 1.5배, 쾌청 = 불 기술 1.5배, 모래 = 바위 특방 1.5배, 설경 = 얼음 방어 1.5배
    types = set(sp.get("types", []))
    move_types = {dex.moves[m]["type"] for m in mv if m in dex.moves and dex.moves[m]["category"] != "Status"}
    if "Water" in move_types:
        uses_w.add("rain")
    if "Fire" in move_types:
        uses_w.add("sun")
    if "Rock" in types:
        uses_w.add("sand")
    if "Ice" in types:
        uses_w.add("snow")
    attacker = max(bs.get("atk", 0), bs.get("spa", 0)) >= 100
    return {
        "w_set": sets_w, "w_use": uses_w, "tr": "trickroom" in mv, "slow": bs.get("spe", 100) <= 55 and attacker,
        "tw": "tailwind" in mv, "attacker": attacker, "screens": bool(mv & {"reflect", "lightscreen", "auroraveil"}),
        "setup": bool(mv & SETUP), "intim": ab == "intimidate", "fakeout": "fakeout" in mv,
        "redirect": bool(mv & {"followme", "ragepowder"}), "pivot": bool(mv & {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception"}),
        "hazard": bool(mv & {"stealthrock", "spikes", "toxicspikes", "stickyweb"}),
        "speedctl": bool(mv & {"trickroom", "tailwind", "icywind", "electroweb", "thunderwave", "scaryface"}),
        "prankster": ab == "prankster" and bool(mv & {"tailwind", "reflect", "lightscreen", "thunderwave", "taunt", "encore", "willowisp", "trickroom", "followme"}),
        "regen": ab == "regenerator", "terrain": ab in {"electricsurge", "psychicsurge", "grassysurge", "mistysurge", "hadronengine"},
        # 특성으로 막는 타입 (부유·저수·피뢰침·타오르는불꽃·초식 등) — 같은 편이 약한 타입을 받아주는가
        "immune": {"levitate": "Ground", "waterabsorb": "Water", "stormdrain": "Water", "dryskin": "Water",
                   "voltabsorb": "Electric", "lightningrod": "Electric", "motordrive": "Electric", "flashfire": "Fire",
                   "wellbakedbody": "Fire", "sapsipper": "Grass", "eartheater": "Ground"}.get(ab, ""),
        "weak": [t for t in dex.typechart if t not in ("Stellar", "???")
                 and sum(dex.effectiveness(t, d) for d in sp.get("types", [])) >= 1
                 and not any(dex.immune(t, d) for d in sp.get("types", []))],
    }


def synergy(R: list, A: list, lead: int, R_opp: list, B: list) -> list:
    ra = [R[i] for i in A]
    rb = [R_opp[j] for j in B]
    w_set = set().union(*[r["w_set"] for r in ra])
    w_use = sum(1 for r in ra if r["w_use"] & w_set)
    opp_w = set().union(*[r["w_set"] for r in rb])
    tr = any(r["tr"] for r in ra)
    return [
        float(w_use) if w_set else 0.0,
        float(sum(r["slow"] for r in ra)) if tr else 0.0,
        float(sum(r["attacker"] for r in ra)) if any(r["tw"] for r in ra) else 0.0,
        float(sum(r["setup"] for r in ra)) if any(r["screens"] for r in ra) else 0.0,
        float(sum(r["intim"] for r in ra)),
        float(any(R[i]["fakeout"] for i in A[:lead])),
        float(sum(r["redirect"] for r in ra)),
        float(sum(r["pivot"] for r in ra)),
        float(any(R[i]["hazard"] for i in A[:lead])),
        float(sum(r["setup"] for r in ra)),
        float(sum(r["speedctl"] for r in ra)) - float(sum(r["speedctl"] for r in rb)),
        float(bool(w_set and opp_w and w_set != opp_w)),
        float(sum(r["prankster"] for r in ra)),
        float(sum(r["pivot"] for r in ra)) if any(r["regen"] for r in ra) else 0.0,
        float(any(r["terrain"] for r in ra)),
        float(sum(1 for r in ra for o in ra if o is not r and r["immune"] and r["immune"] in o["weak"])),
    ]
NF = len(FEAT_NAMES)


def typical_set(dex: Dex, sid: str, game_type="singles") -> dict:
    """상대 포켓몬의 '가장 그럴듯한' 세트 (프리뷰에서는 종만 보이므로)."""
    b = OppBelief(dex, sid, game_type)
    ip = b.item_probs()
    sp = b.spread_probs() or [{"nature": "serious", "sp": {}}]
    return {"species": dex.base_of_mega(b.species), "ability": b.likely_ability(),
            "item": max(ip, key=ip.get) if ip else "", "nature": sp[0].get("nature") or "serious",
            "sp": dict(sp[0].get("sp") or {}), "moves": list(b.move_probs())[:4]}


def all_orders(n, game_type):
    pick, lead = (4, 2) if game_type == "doubles" else (3, 1)
    return [[int(c) - 1 for c in o] for o in orders(n, pick, lead)], lead


def pair_features(G: np.ndarray, A: list, B: list, lead: int, mega_a=None, spd=None, R=None, R_opp=None) -> np.ndarray:
    """G: 내 6 × 상대 6 상성 (+ 유리). A, B: 선출(0부터, 앞 lead 마리가 선봉)."""
    blk = G[np.ix_(A, B)]
    la, lb = A[:lead], B[:lead]
    cols = blk.max(0)            # 상대 포켓몬별 내 최선의 답
    their = (-blk).max(1)        # 내 포켓몬별 상대의 최선의 답
    rest = [i for i in range(G.shape[0]) if i not in A]
    f = [
        float(G[np.ix_(la, lb)].mean()),
        float(G[np.ix_(la, B)].mean()),
        float(-G[np.ix_(A, lb)].mean()),
        float(blk.mean()), float(blk.max()), float(blk.min()),
        float(cols.mean()), float(cols.min()),
        float(their.mean()), float(their.max()),
        float((cols > 0.5).sum()) / len(B),
        float((their > 0.5).sum()) / len(A),
        float(np.mean([spd[i] for i in la])) if spd is not None else 0.0,
        float(sum(1 for i in A if mega_a and mega_a[i])),
        float(G[np.ix_(rest, B)].max()) if rest else 0.0,
        float(G[np.ix_(A[lead:], B)].mean()) if len(A) > lead else 0.0,
    ]
    f += synergy(R, A, lead, R_opp, B) if R is not None else [0.0] * 16
    return np.array(f, np.float32)


class Matchup:
    """한 매치업의 상성표와 모든 선출 쌍의 변수 (내 시점·상대 시점)."""

    def __init__(self, dex: Dex, pw: Pairwise, mine: list, theirs: list, game_type="singles"):
        self.G = pw.grid(mine, theirs)
        ma = [pw.mon(s) for s in mine]
        mb = [pw.mon(s) for s in theirs]
        self.spd_a = [m.stats["spe"] / 200.0 for m in ma]
        self.spd_b = [m.stats["spe"] / 200.0 for m in mb]
        self.mega_a = [bool(dex.mega_for(s["species"], s.get("item", ""))) for s in mine]
        self.mega_b = [bool(dex.mega_for(s["species"], s.get("item", ""))) for s in theirs]
        self.ra = [roles(dex, x) for x in mine]
        self.rb = [roles(dex, x) for x in theirs]
        self.oa, self.lead = all_orders(len(mine), game_type)
        self.ob, _ = all_orders(len(theirs), game_type)

    def feats(self, i_list=None, j_list=None):
        """선출 쌍마다 (내 시점 변수, 상대 시점 변수)."""
        ia = range(len(self.oa)) if i_list is None else i_list
        jb = range(len(self.ob)) if j_list is None else j_list
        Fa, Fb = [], []
        GT = -self.G.T
        for i in ia:
            for j in jb:
                A, B = self.oa[i], self.ob[j]
                Fa.append(pair_features(self.G, A, B, self.lead, self.mega_a, self.spd_a, self.ra, self.rb))
                Fb.append(pair_features(GT, B, A, self.lead, self.mega_b, self.spd_b, self.rb, self.ra))
        return np.stack(Fa), np.stack(Fb)

    def index_of(self, side_orders, order):
        lead = self.lead
        for k, o in enumerate(side_orders):
            if set(o[:lead]) == set(order[:lead]) and set(o) == set(order):
                return k
        return None


class BringNet:
    def __init__(self, hidden=(32, 16), seed=0):
        self.h = NpMLP([NF, *hidden, 1], seed)
        self.mu = np.zeros(NF, np.float32)
        self.sd = np.ones(NF, np.float32)
        self.info = {}

    def _n(self, F):
        return np.clip((F - self.mu) / self.sd, -6, 6).astype(np.float32)

    def logits(self, Fa, Fb):
        return self.h.forward(self._n(Fa))[:, 0] - self.h.forward(self._n(Fb))[:, 0]

    def prob(self, Fa, Fb):
        z = self.logits(Fa, Fb)
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def matrix(self, mu: Matchup) -> np.ndarray:
        Fa, Fb = mu.feats()
        return self.prob(Fa, Fb).reshape(len(mu.oa), len(mu.ob))

    # ---- 학습
    def fit(self, Fa, Fb, Y, epochs=30, lr=2e-3, batch=128, val=None, log=print, seed=0, wd=1e-4):
        allF = np.concatenate([Fa, Fb])
        self.mu = allF.mean(0)
        s = allF.std(0)
        self.sd = np.where(s < 1e-3, 1.0, s).astype(np.float32)
        rng = np.random.default_rng(seed)
        opt = Adam(self.h.params(), lr, wd)
        idx = np.arange(len(Y))
        best = (1e9, None)
        for ep in range(epochs):
            rng.shuffle(idx)
            for s0 in range(0, len(idx), batch):
                b = idx[s0:s0 + batch]
                za, aa = self.h.forward(self._n(Fa[b]), keep=True)
                zb, ab = self.h.forward(self._n(Fb[b]), keep=True)
                p = 1 / (1 + np.exp(-np.clip(za[:, 0] - zb[:, 0], -30, 30)))
                g = ((p - Y[b]) / len(b))[:, None]
                gWa, gba = self.h.backward(aa, g)
                gWb, gbb = self.h.backward(ab, -g)
                opt.step([x + y for x, y in zip(gWa, gWb)] + [x + y for x, y in zip(gba, gbb)])
            if val is not None:
                m = self.evaluate(*val)
                if log:
                    log(f"    에폭 {ep + 1}: 검증 손실 {m['loss']:.4f} · 승패 예측 {m['acc']:.1%}")
                if m["loss"] < best[0] - 1e-5:
                    best = (m["loss"], self.h.to_json())
        if best[1]:
            self.h = NpMLP.from_json(best[1])

    def evaluate(self, Fa, Fb, Y):
        p = np.clip(self.prob(Fa, Fb), 1e-6, 1 - 1e-6)
        loss = float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean())
        base = float(Y.mean())
        bl = -(base * math.log(max(base, 1e-6)) + (1 - base) * math.log(max(1 - base, 1e-6)))
        return {"loss": loss, "baseline": bl, "acc": float(((p > 0.5) == (Y > 0.5)).mean()), "n": int(len(Y))}

    # ---- 저장
    def save(self, path, extra=None):
        d = {"kind": "bringnet", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "features": FEAT_NAMES, "h": self.h.to_json(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
             "info": self.info, **(extra or {})}
        Path(path).write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("features") != FEAT_NAMES:
            raise ValueError("선출망 변수 목록이 다릅니다 — 다시 학습이 필요합니다.")
        m = cls.__new__(cls)
        m.h = NpMLP.from_json(d["h"])
        m.mu, m.sd = np.array(d["mu"], np.float32), np.array(d["sd"], np.float32)
        m.info = d.get("info", {})
        return m


def path_for(game_type):
    return MODELS / f"bring_{game_type}.json"


def load(game_type):
    p = path_for(game_type)
    try:
        return BringNet.load(p) if p.is_file() else None
    except Exception:
        return None


# ======================================================================== 데이터
def build_data(games, dex: Dex, game_type: str, max_games=None, log=print):
    """판 기록 → (내 시점 변수, 상대 시점 변수, 이겼는가). 한 판에서 양쪽 시점 2개."""
    pw = Pairwise(dex, game_type)
    typ_cache = {}
    Fa, Fb, Y, G_ = [], [], [], []
    n = 0
    t0 = time.time()
    for g in games:
        if g.get("gameType") != game_type or g.get("winner") not in ("p1", "p2"):
            continue
        pv = {d["side"]: d for d in g.get("decisions", []) if d.get("kind") == "preview" and d.get("chosen") is not None}
        if len(pv) < 2:
            continue
        n += 1
        if max_games and n > max_games:
            break
        chosen = {}
        for s, d in pv.items():
            probs = d.get("probs")
            opt = d["options"][d["chosen"]]
            chosen[s] = [int(c) - 1 for c in opt.split()[1]]
        for s in ("p1", "p2"):
            o = "p2" if s == "p1" else "p1"
            mine = [normalize_set(dex, x) for x in g["teams"][s] if x.get("species") in dex.species]
            theirs = []
            for x in g["teams"][o]:
                sid = dex.base_of_mega(x["species"]) if x.get("species") in dex.species else None
                if not sid:
                    continue
                if sid not in typ_cache:
                    typ_cache[sid] = typical_set(dex, sid, game_type)
                theirs.append(typ_cache[sid])
            if len(mine) != len(g["teams"][s]) or len(theirs) != len(g["teams"][o]):
                continue
            try:
                mu = Matchup(dex, pw, mine, theirs, game_type)
            except Exception:
                continue
            i = mu.index_of(mu.oa, chosen[s])
            j = mu.index_of(mu.ob, chosen[o])
            if i is None or j is None:
                continue
            fa, fb = mu.feats([i], [j])
            Fa.append(fa[0])
            Fb.append(fb[0])
            Y.append(1.0 if g["winner"] == s else 0.0)
            G_.append(n)
        if n % 200 == 0:
            log(f"  선출 변수 {n}판 · {time.time() - t0:.0f}초")
    return np.array(Fa, np.float32), np.array(Fb, np.float32), np.array(Y, np.float32), np.array(G_)


# ======================================================================== 선출 정하기
def decide(net: BringNet, dex: Dex, my_sets: list, opp_species: list, game_type="singles", pw=None, iters=400):
    """내 세트 6 + 상대 종 6 → 모든 선출 쌍의 예측 승률표 → 균형해."""
    pw = pw or Pairwise(dex, game_type)
    theirs = [typical_set(dex, dex.base_of_mega(s), game_type) for s in opp_species if s in dex.species]
    mu = Matchup(dex, pw, [normalize_set(dex, s) for s in my_sets], theirs, game_type)
    M = net.matrix(mu)
    p, q, v = solve_matrix(M, iters=iters)
    return mu, M, p, q, v


class BringPolicy:
    """배틀은 다른 정책에 맡기고, 선출(선봉 포함)만 선출망 + 균형해로 정한다.

    explore: 자기대전에서 다양한 선출을 시도하도록 균형해 확률에 섞는 고른 확률 (학습 데이터 다양화).
    """

    def __init__(self, dex: Dex, battle_policy, net: BringNet, game_type="singles", explore=0.0, seed=None):
        import random
        self.dex, self.inner, self.net, self.gt = dex, battle_policy, net, game_type
        self.name = getattr(battle_policy, "name", "policy")
        self.explore = explore
        self.rng = random.Random(seed)
        self.pw = Pairwise(dex, game_type)

    def start(self, side, team_sets, dex):
        self.sets = team_sets
        self.inner.start(side, team_sets, dex)

    def choose(self, view, req, options):
        if not req.get("teamPreview"):
            return self.inner.choose(view, req, options)
        snap = view.snapshot() if hasattr(view, "snapshot") else view
        opp = "p2" if snap["me"] == "p1" else "p1"
        try:
            mu, M, p, q, v = decide(self.net, self.dex, self.sets, snap["preview"].get(opp, []), self.gt, self.pw)
        except Exception:
            return self.inner.choose(view, req, options)
        probs = [0.0] * len(options)
        for k, o in enumerate(options):
            order = [x - 1 for x in o["parts"]]
            i = mu.index_of(mu.oa, order)
            if i is not None:
                probs[k] = float(p[i])
        s = sum(probs)
        if s <= 0:
            return self.inner.choose(view, req, options)
        probs = [x / s for x in probs]
        if self.explore:
            probs = [(1 - self.explore) * x + self.explore / len(probs) for x in probs]
        idx = self.rng.choices(range(len(options)), weights=probs)[0]
        self.last_value = v
        return idx, probs
