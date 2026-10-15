# -*- coding: utf-8 -*-
"""파티 최적화 — 6마리 × (포켓몬·특성·도구·성격·SP·기술 4개) 전체를 실제 대전으로 평가하며 진화시킨다.

평가: 후보 파티를 메타 상대 파티 여러 개와 '선출(승률표의 균형) → 끝까지 배틀' 시켜 평균 승률을 낸다.
      상대 파티와 난수는 고정(같은 조건 비교)이라, 한 군데만 바꾼 파티끼리 공정하게 비교된다.
탐색: 좋은 출발점(메타 세트 기반)에서 한 군데씩 바꿔 보고 승률이 오르면 남긴다 (언덕 오르기 + 재평가).
기록: 평가할 때마다 (파티 → 승률)을 userdata/teamlog 에 남긴다 — 팀 평가망 학습 데이터.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from . import bring as B
from .dex import Dex, to_id
from .teamgen import TeamGen, to_showdown_set

ROOT = Path(__file__).resolve().parent.parent
TEAMLOG = ROOT / "userdata" / "teamlog"
GENERIC_ITEMS = ["lifeorb", "choicescarf", "focussash", "sitrusberry", "leftovers", "expertbelt", "lumberry"]


class SetPool:
    """포켓몬 한 마리의 세트 후보 (메타 통계 · 전문가 세트 · 배울 수 있는 기술)."""

    def __init__(self, dex: Dex, game_type="singles", seed=0):
        self.dex, self.gt = dex, game_type
        self.rng = random.Random(seed)

    def usage(self, sid):
        return self.dex.usage_of(sid, self.gt) or {}

    def learn(self, sid):
        sp = self.dex.species[sid]
        return set(self.dex.learnsets.get(sid) or self.dex.learnsets.get(to_id(sp.get("changesFrom") or sp["baseSpecies"])) or [])

    def default_set(self, sid, used_items=()) -> dict:
        dex, u = self.dex, self.usage(sid)
        learn = self.learn(sid)
        moves = [m for m in (u.get("moves") or {}) if m in dex.moves and (not learn or m in learn)][:4]
        if len(moves) < 4 or not any(dex.moves[m]["category"] != "Status" for m in moves):
            for e in (dex.expert.get(self.gt) or {}).get(sid, []):
                moves = [slot[0] for slot in e["moves"] if slot][:4]
                break
        if len(moves) < 4:
            sp = dex.species[sid]
            dmg = sorted((m for m in learn if dex.moves[m]["category"] != "Status"),
                         key=lambda m: -((dex.moves[m]["basePower"] or 0) * (1.5 if dex.moves[m]["type"] in sp["types"] else 1)))
            for m in dmg:
                if len(moves) >= 4:
                    break
                if m not in moves:
                    moves.append(m)
        items = [i for i in (u.get("items") or {}) if i in dex.items and i not in used_items] or \
            [i for i in GENERIC_ITEMS if i not in used_items]
        abil = u.get("abilities") or {}
        ability = max(abil, key=abil.get) if abil else to_id(dex.species[sid]["abilities"].get("0", ""))
        sps = u.get("spreads") or []
        spread, nature = (dict(sps[0]["sp"]), sps[0]["nature"]) if sps else ({"hp": 32, "atk": 17, "spa": 17}, "serious")
        return {"species": sid, "ability": ability, "item": items[0] if items else "", "nature": nature or "serious",
                "sp": spread, "moves": moves[:4]}

    # ---- 바꾸기 후보
    def alt_moves(self, sid):
        u = self.usage(sid).get("moves") or {}
        learn = self.learn(sid)
        cand = [m for m in u if m in self.dex.moves and (not learn or m in learn)]
        sp = self.dex.species[sid]
        extra = sorted((m for m in learn if self.dex.moves[m]["category"] != "Status"),
                       key=lambda m: -((self.dex.moves[m]["basePower"] or 0) * (1.5 if self.dex.moves[m]["type"] in sp["types"] else 1)))[:8]
        return list(dict.fromkeys(cand[:16] + extra))

    def alt_items(self, sid):
        u = self.usage(sid).get("items") or {}
        return list(dict.fromkeys([i for i in u if i in self.dex.items][:10] + GENERIC_ITEMS))

    def alt_spreads(self, sid):
        u = self.usage(sid).get("spreads") or []
        out = [(dict(s["sp"]), s["nature"] or "serious") for s in u[:8]]
        bs = self.dex.species[sid]["baseStats"]
        a = "atk" if bs["atk"] >= bs["spa"] else "spa"
        out += [({a: 32, "spe": 32, "hp": 2}, "jolly" if a == "atk" else "timid"),
                ({"hp": 32, a: 32, "def": 2}, "adamant" if a == "atk" else "modest"),
                ({"hp": 32, "def": 17, "spd": 17}, "bold" if a == "spa" else "impish")]
        return out


class TeamOptimizer:
    def __init__(self, dex: Dex, bridge, fmt: str, game_type="singles", pool=None, n_opp=10, games=2,
                 k_orders=4, seed=1, opponents=None, log=print):
        self.dex, self.br, self.fmt, self.gt = dex, bridge, fmt, game_type
        self.rng = random.Random(seed)
        self.seed = seed
        self.sets = SetPool(dex, game_type, seed)
        # 보관함이 비어 있으면 이번 시즌에 고를 수 있는 모든 종 (메타 통계가 없는 종도)
        self.pool = [dex.find("species", s) for s in (pool or [])] if pool else dex.season_species()
        self.pool = [s for s in self.pool if s and not dex.species[s].get("isMega")]
        self.games, self.k = games, k_orders
        self.pick, self.lead = (4, 2) if game_type == "doubles" else (3, 1)
        tg = TeamGen(dex, seed=seed + 999, game_type=game_type, meta=1.0)
        self.opponents = opponents or [tg.team(6) for _ in range(n_opp)]
        self.opp_packed = [bridge.pack([to_showdown_set(dex, s) for s in t]) for t in self.opponents]
        self.pw = B.Pairwise(dex, game_type)
        self.log = log
        self.evals = 0
        TEAMLOG.mkdir(parents=True, exist_ok=True)
        self.logfile = TEAMLOG / f"teams_{game_type}_{int(time.time())}.jsonl"

    # ------------------------------------------------------------ 평가
    def evaluate(self, team: list, games=None) -> dict:
        games = games or self.games
        packed = self.br.pack([to_showdown_set(self.dex, s) for s in team])
        mega_me = [bool(self.dex.mega_for(s["species"], s.get("item", ""))) for s in team]
        vals, per = [], []
        for oi, (opp, opk) in enumerate(zip(self.opponents, self.opp_packed)):
            G = self.pw.grid(team, opp)
            oa = B.prune_orders(G, self.pick, self.lead, self.k, mega_me)
            mega_op = [bool(self.dex.mega_for(s["species"], s.get("item", ""))) for s in opp]
            ob = B.prune_orders(-G.T, self.pick, self.lead, self.k, mega_op)
            r = self.br.call(op="matrix", format=self.fmt, A={"packed": packed}, B={"worlds": [opk]},
                             ordersA=oa, ordersB=ob, games=games, seed=self.seed * 1000 + oi, budgetMs=30000)
            M = np.array(r["M"])
            _, _, v = B.solve_matrix(M, iters=600)
            vals.append(v)
            per.append(round(v, 3))
        self.evals += 1
        score = float(np.mean(vals))
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps({"team": team, "score": score, "perOpp": per, "games": games,
                                "opponentsSeed": self.seed}, ensure_ascii=False) + "\n")
        return {"score": score, "perOpp": per}

    # ------------------------------------------------------------ 출발점
    def initial_team(self, fixed=()) -> list:
        """사용률·1:1 상성으로 고른 출발 파티 (메가 1마리, 같은 도구 금지)."""
        dex = self.dex
        team = [copy.deepcopy(s) for s in fixed]
        used = {s.get("item") for s in team}
        opp_flat = [s for t in self.opponents[:4] for s in t]
        cand = [s for s in self.pool if dex.species[s]["baseSpecies"] not in {dex.species[t["species"]]["baseSpecies"] for t in team}]
        scored = []
        for sid in cand[:60]:
            st = self.sets.default_set(sid)
            g = self.pw.grid([st], opp_flat)[0]
            scored.append((float(g.mean()), sid, g))
        scored.sort(key=lambda x: -x[0])
        cover = np.full(len(opp_flat), -9.0)
        while len(team) < 6 and scored:
            # 이미 뽑은 포켓몬이 약한 상대를 메워 주는 쪽을 우선 (조합)
            best = max(scored, key=lambda x: np.maximum(cover, x[2]).mean() + 0.2 * x[0])
            scored.remove(best)
            st = self.sets.default_set(best[1], used)
            megas = sum(1 for s in team if dex.mega_for(s["species"], s.get("item", "")))
            if dex.items.get(st["item"], {}).get("megaStone") and megas >= 1:
                st["item"] = next((i for i in GENERIC_ITEMS if i not in used), "")
            team.append(st)
            used.add(st["item"])
            cover = np.maximum(cover, best[2])
        return team

    # ------------------------------------------------------------ 바꾸기
    def mutate(self, team: list, locked=()) -> tuple:
        dex, rng = self.dex, self.rng
        t = copy.deepcopy(team)
        free = [i for i in range(len(t)) if i not in locked] or list(range(len(t)))
        i = rng.choice(free)
        s = t[i]
        kind = rng.choices(["move", "item", "spread", "species"], weights=[4, 2, 2, 1.5])[0]
        used_items = {x.get("item") for j, x in enumerate(t) if j != i}
        if kind == "move":
            alts = [m for m in self.sets.alt_moves(s["species"]) if m not in s["moves"]]
            if alts and s["moves"]:
                k = rng.randrange(len(s["moves"]))
                old = s["moves"][k]
                s["moves"][k] = rng.choice(alts[:12])
                return t, f"{dex.species[s['species']]['ko']}: {dex.moves[old]['ko']} → {dex.moves[s['moves'][k]]['ko']}"
        if kind == "item":
            megas = sum(1 for j, x in enumerate(t) if j != i and dex.mega_for(x["species"], x.get("item", "")))
            alts = [it for it in self.sets.alt_items(s["species"]) if it not in used_items and it != s["item"]
                    and not (dex.items[it].get("megaStone") and megas >= 1)]
            if alts:
                old = s["item"]
                s["item"] = rng.choice(alts)
                return t, f"{dex.species[s['species']]['ko']}: 도구 {dex.items.get(old, {}).get('ko', '없음')} → {dex.items[s['item']]['ko']}"
        if kind == "spread":
            alts = self.sets.alt_spreads(s["species"])
            if alts:
                sp, nat = rng.choice(alts)
                s["sp"], s["nature"] = dict(sp), nat
                return t, f"{dex.species[s['species']]['ko']}: SP {'/'.join(f'{v}{k}' for k, v in sp.items())} {dex.natures.get(nat, {}).get('ko', nat)}"
        # 포켓몬 교체
        bases = {dex.species[x["species"]]["baseSpecies"] for j, x in enumerate(t) if j != i}
        cand = [p for p in self.pool if dex.species[p]["baseSpecies"] not in bases and p != s["species"]]
        if cand:
            new = rng.choice(cand[:80])
            old = s["species"]
            ns = self.sets.default_set(new, used_items)
            megas = sum(1 for j, x in enumerate(t) if j != i and dex.mega_for(x["species"], x.get("item", "")))
            if dex.items.get(ns["item"], {}).get("megaStone") and megas >= 1:
                ns["item"] = next((it for it in GENERIC_ITEMS if it not in used_items), "")
            t[i] = ns
            return t, f"교체: {dex.species[old]['ko']} → {dex.species[new]['ko']}"
        return t, "변화 없음"

    # ------------------------------------------------------------ 최적화
    def optimize(self, team=None, steps=30, locked=(), time_budget=None, progress=None) -> dict:
        """한 군데씩 바꿔 보며 승률이 오르면 채택.
        빠른 거르기(판 수 적게)와 확인(판 수 2배)은 각각 같은 판 수끼리만 비교한다 (섞으면 숫자가 어긋남)."""
        t0 = time.time()
        team = team or self.initial_team()
        big = self.games * 2
        cur = self.evaluate(team)                    # 빠른 거르기 기준
        cur_big = self.evaluate(team, games=big)     # 확인 기준
        start_big = cur_big["score"]
        history = [{"step": 0, "score": round(cur_big["score"], 4), "change": "출발 파티"}]
        self.log(f"  출발 파티 승률 {cur_big['score']:.1%}")
        for step in range(1, steps + 1):
            if time_budget and time.time() - t0 > time_budget:
                break
            cand, desc = self.mutate(team, locked)
            r = self.evaluate(cand)
            if r["score"] > cur["score"] + 0.005:
                # 우연을 걸러내려고 판을 늘려 한 번 더 확인
                r2 = self.evaluate(cand, games=big)
                if r2["score"] > cur_big["score"]:
                    team, cur, cur_big = cand, r, r2
                    history.append({"step": step, "score": round(r2["score"], 4), "change": desc, "kept": True})
                    self.log(f"  {step}: {desc} → 승률 {r2['score']:.1%} (채택)")
            if progress:
                progress(step, steps, cur_big["score"])
        return {"team": team, "score": cur_big["score"], "startScore": start_big, "perOpp": cur_big["perOpp"],
                "history": history, "evals": self.evals, "seconds": round(time.time() - t0)}

    def recommend_one(self, team5: list, top=5, candidates=None) -> list:
        """지금 파티(5마리 이하)에 넣었을 때 승률을 가장 올리는 포켓몬."""
        dex = self.dex
        bases = {dex.species[s["species"]]["baseSpecies"] for s in team5}
        used = {s.get("item") for s in team5}
        cand = [p for p in (candidates or self.pool) if dex.species.get(p) and dex.species[p]["baseSpecies"] not in bases]
        base_team = list(team5)
        # 빈 자리는 평가를 위해 임시로 채움
        filler = self.initial_team(fixed=team5)
        base_score = self.evaluate(filler)["score"] if len(team5) < 6 else None
        out = []
        for sid in cand:
            st = self.sets.default_set(sid, used)
            megas = sum(1 for s in team5 if dex.mega_for(s["species"], s.get("item", "")))
            if dex.items.get(st["item"], {}).get("megaStone") and megas >= 1:
                st["item"] = next((i for i in GENERIC_ITEMS if i not in used), "")
            trial = base_team + [st]
            if len(trial) < 6:
                trial = trial + [x for x in filler if x["species"] not in {y["species"] for y in trial}][:6 - len(trial)]
            r = self.evaluate(trial[:6])
            out.append({"species": sid, "ko": dex.species[sid]["ko"], "score": round(r["score"], 4), "set": st,
                        "gain": round(r["score"] - base_score, 4) if base_score is not None else None})
        out.sort(key=lambda x: -x["score"])
        return out[:top]
