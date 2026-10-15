# -*- coding: utf-8 -*-
"""선출 — 상대 6마리를 보고 내 3마리(더블 4마리)와 선봉을 고른다.

방법
  1. 내 선출 후보(조합 × 선봉)와 상대 선출 후보를 휴리스틱으로 추린다.
  2. 칸마다 쇼다운 엔진으로 끝까지 대전시켜 승률표를 채운다 (상대 세트는 메타 통계로 여러 번 뽑음).
  3. 행렬 게임으로 푼다 — 상대도 최선의 선출을 한다고 볼 때의 균형(혼합 전략)과,
     상대가 메타대로 낼 때 가장 좋은 선출을 함께 낸다.
"""
from __future__ import annotations

import itertools
import random

import numpy as np

from .belief import OppBelief, sample_set
from .dex import Dex
from .mon import FieldState, Mon
from .policies import HeuristicPolicy
from .teamgen import to_showdown_set


def orders(n: int, pick: int, lead: int) -> list:
    """선출 순서 문자열 목록 ('123' = 1·2·3번, 앞 lead 마리가 선봉)."""
    out = []
    for combo in itertools.combinations(range(1, n + 1), pick):
        for leads in itertools.combinations(combo, lead):
            rest = [c for c in combo if c not in leads]
            out.append("".join(str(x) for x in list(leads) + rest))
    return out


def solve_matrix(M: np.ndarray, iters: int = 3000):
    """0합 행렬 게임(행: 나, 값: 내 승률)의 근사 균형 — 가상 플레이(fictitious play)."""
    n, m = M.shape
    row_counts = np.zeros(n)
    col_counts = np.zeros(m)
    r, c = int(np.argmax(M.min(1))), int(np.argmin(M.max(0)))
    for _ in range(iters):
        row_counts[r] += 1
        col_counts[c] += 1
        r = int(np.argmax(M @ (col_counts / col_counts.sum())))
        c = int(np.argmin((row_counts / row_counts.sum()) @ M))
    p = row_counts / row_counts.sum()
    q = col_counts / col_counts.sum()
    return p, q, float(p @ M @ q)


class Pairwise:
    """1:1 상성 (데미지 엔진) — 후보를 추릴 때 쓴다."""

    def __init__(self, dex: Dex, game_type="singles"):
        self.dex = dex
        self.pol = HeuristicPolicy(dex)
        self.field = FieldState(game_type=game_type)

    def mon(self, s: dict) -> Mon:
        m = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""),
                nature=s.get("nature", "serious"), sp=dict(s.get("sp") or {}), moves=list(s.get("moves") or [])).setup(self.dex)
        if self.dex.mega_for(m.species, m.item):
            m.mega_evolve()
        return m

    def profile(self, a: Mon, b: Mon, a_side=0) -> dict:
        """a 가 b 를 칠 때: 첫 타(가득 HP, 멀티스케일 반영) 비율·처치 확률, 선공기 비율, 회복기 여부."""
        from .policies import RECOVERY
        g = self.pol.best_vs(a, b, self.field, a_side)
        prio = 0.0
        for mv in a.moves:
            mo = self.dex.moves.get(mv)
            if mo and mo["category"] != "Status" and mo.get("priority", 0) > 0:
                f, ko = self.pol.exp_frac(a, b, mv, self.field, a_side)
                prio = max(prio, f)
        first = min(1.0, g[0])
        nxt = min(1.0, first * 2) if b.ability in ("multiscale", "shadowshield") else first
        return {"first": first, "ko": g[1], "next": nxt, "prio": prio, "move": g[2],
                "heal": 0.5 if any(m in RECOVERY for m in a.moves) else 0.0,
                "spe": self.pol.speed(a, self.field, a_side)}

    def race(self, a: Mon, b: Mon) -> float:
        """1:1 경주: 번갈아 가장 센 기술로 치고(빠른 쪽 먼저, 선공기로 마무리 가능하면 그쪽 먼저),
        HP 가 적고 받는 타격이 작으면 회복. 이긴 쪽의 남은 HP (+: a 승, −: b 승), −1~1."""
        pa, pb = self.profile(a, b, 0), self.profile(b, a, 1)
        if pa["first"] <= 0 and pb["first"] <= 0:
            return 0.0
        ha = hb = 1.0
        hits = {"a": 0, "b": 0}
        heals = {"a": 0, "b": 0}
        a_fast = pa["spe"] > pb["spe"] or (pa["spe"] == pb["spe"] and a.stats.get("spe", 0) >= b.stats.get("spe", 0))
        for _ in range(40):
            a_prio_ko = pa["prio"] >= hb > 0
            b_prio_ko = pb["prio"] >= ha > 0
            order = ["a", "b"] if a_fast else ["b", "a"]
            if a_prio_ko and not b_prio_ko:
                order = ["a", "b"]
            elif b_prio_ko and not a_prio_ko:
                order = ["b", "a"]
            for who in order:
                if ha <= 0 or hb <= 0:
                    break
                me, hp, other_p = (pa, ha, pb) if who == "a" else (pb, hb, pa)
                dmg = me["first"] if hits[who] == 0 else me["next"]
                incoming = other_p["first"] if hits["b" if who == "a" else "a"] == 0 else other_p["next"]
                if me["heal"] and hp < 0.5 and heals[who] < 4 and incoming < me["heal"] - 0.05 and dmg < (hb if who == "a" else ha):
                    heals[who] += 1
                    if who == "a":
                        ha = min(1.0, ha + me["heal"])
                    else:
                        hb = min(1.0, hb + me["heal"])
                    continue
                if dmg <= 0:
                    continue
                hits[who] += 1
                if who == "a":
                    hb -= dmg
                else:
                    ha -= dmg
            if ha <= 0 or hb <= 0:
                break
        if ha <= 0 and hb <= 0:
            return 0.0
        if hb <= 0:
            return float(max(0.05, ha))
        if ha <= 0:
            return float(-max(0.05, hb))
        return float(0.3 * (ha - hb))   # 끝나지 않으면 (둘 다 못 뚫음) 남은 HP 차의 일부

    def grid(self, mine: list, theirs: list) -> np.ndarray:
        """내 세트 × 상대 세트의 1:1 경주 결과 (+ 유리)."""
        A = [self.mon(s) for s in mine]
        B = [self.mon(s) for s in theirs]
        G = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                G[i, j] = self.race(a, b)
        return G


def prune_orders(G: np.ndarray, pick: int, lead: int, k: int, mega_flags=None) -> list:
    """상성표 G(행=내 포켓몬)로 선출 후보 k 개를 고른다 (조합이 겹치지 않게 다양하게)."""
    n = G.shape[0]
    scored = []
    for o in orders(n, pick, lead):
        idx = [int(c) - 1 for c in o]
        cover = G[idx].max(0).mean()
        avg = G[idx].mean()
        ld = G[idx[:lead]].mean()
        megas = sum(1 for i in idx if mega_flags and mega_flags[i])
        scored.append((0.5 * cover + 0.25 * avg + 0.25 * ld - 0.3 * max(0, megas - 1), o))
    scored.sort(reverse=True)
    out, seen = [], {}
    for sc, o in scored:
        key = "".join(sorted(o))
        if seen.get(key, 0) >= 2:      # 같은 조합은 선봉만 다른 것 2개까지
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(o)
        if len(out) >= k:
            break
    return out


def recommend(dex: Dex, bridge, fmt: str, my_team: list, opp_species: list, game_type="singles",
              k_mine=10, k_opp=10, worlds=6, games=6, seed=1, budget_ms=60000) -> dict:
    """내 6마리 세트 + 상대 6마리 종 → 선출 추천."""
    rng = random.Random(seed)
    pick, lead = (4, 2) if game_type == "doubles" else (3, 1)
    opp_species = [dex.find("species", s) for s in opp_species if dex.find("species", s)]
    beliefs = {s: OppBelief(dex, s, game_type) for s in opp_species}
    opp_worlds = [[sample_set(dex, beliefs[s], rng) for s in opp_species] for _ in range(worlds)]
    pw = Pairwise(dex, game_type)
    # 상대의 대표 세트(가장 그럴듯한)로 상성표
    opp_typ = [dict(species=s, ability=beliefs[s].likely_ability(), item=(lambda ip: max(ip, key=ip.get) if ip else "")(beliefs[s].item_probs()),
                    nature=(beliefs[s].spread_probs() or [{"nature": "serious"}])[0]["nature"],
                    sp=(beliefs[s].spread_probs() or [{"sp": {}}])[0]["sp"], moves=list(beliefs[s].move_probs())[:4])
               for s in opp_species]
    G = pw.grid(my_team, opp_typ)
    mega_me = [bool(dex.mega_for(s["species"], s.get("item", ""))) for s in my_team]
    mega_op = [bool(dex.mega_for(s["species"], s.get("item", ""))) for s in opp_typ]
    oa = prune_orders(G, pick, lead, k_mine, mega_me)
    ob = prune_orders(-G.T, pick, lead, k_opp, mega_op)
    packedA = bridge.pack([to_showdown_set(dex, s) for s in my_team])
    packedB = [bridge.pack([to_showdown_set(dex, s) for s in w]) for w in opp_worlds]
    res = bridge.call(op="matrix", format=fmt, A={"packed": packedA}, B={"worlds": packedB},
                      ordersA=oa, ordersB=ob, games=games, seed=seed, budgetMs=budget_ms)
    M = np.array(res["M"])
    p, q, value = solve_matrix(M)
    # 상대가 메타대로(사용률 높은 포켓몬을 더 잘 데려온다고) 낼 때
    use = np.array([((dex.usage_of(s, game_type) or {}).get("usage", 0.02) + 0.02) for s in opp_species])
    prior = np.array([np.prod([use[int(c) - 1] for c in o]) for o in ob])
    prior = prior / prior.sum()
    vs_meta = M @ prior
    best_meta = int(np.argmax(vs_meta))
    best_safe = int(np.argmax(M.min(1)))
    name = lambda team, o: [dex.species[team[int(c) - 1]["species"]]["ko"] for c in o]
    rows = []
    for i, o in enumerate(oa):
        worst = int(np.argmin(M[i]))
        rows.append({"order": o, "names": name(my_team, o), "nash": round(float(p[i]), 3),
                     "vsMeta": round(float(vs_meta[i]), 3), "worst": round(float(M[i].min()), 3),
                     "worstVs": name(opp_typ, ob[worst]), "avg": round(float(M[i].mean()), 3)})
    rows.sort(key=lambda r: -(r["nash"] * 2 + r["vsMeta"]))
    return {
        "value": round(value, 3), "rows": rows,
        "bestMeta": rows and next(r for r in rows if r["order"] == oa[best_meta]),
        "bestSafe": next(r for r in rows if r["order"] == oa[best_safe]),
        "oppLikely": [{"order": ob[j], "names": name(opp_typ, ob[j]), "prob": round(float(q[j]), 3),
                       "metaProb": round(float(prior[j]), 3)} for j in np.argsort(-q)[:5]],
        "matrix": M.round(3).tolist(), "ordersA": oa, "ordersB": ob, "games": res.get("played"),
        "ms": res.get("ms"), "pairwise": G.round(3).tolist(),
    }
