# -*- coding: utf-8 -*-
"""수읽기 정책 (정보집합 몬테카를로 탐색).

매 결정마다
  1. 상대의 보이지 않는 정보(기술·도구·SP·아직 안 나온 포켓몬)를 추정 확률대로 K 번 뽑아 '가능한 세계'를 만든다.
  2. 쇼다운 시뮬레이터(tools/ps_search.js)로 행동마다, 세계마다 R 번씩 D 턴을 굴려 판세를 평가한다.
  3. 평균 승률이 가장 높은 행동을 고른다 (교체·희생·마무리 같은 여러 턴짜리 흐름이 반영된다).

더블은 선택지 조합이 많으므로 선생님 점수 상위 N 개만 탐색한다.
"""
from __future__ import annotations

import math
import random

from .belief import OppBelief, sample_set
from .dex import Dex, to_id
from .policies import HeuristicPolicy
from .protocol import split_ident
from .psbridge import Bridge
from .teamgen import to_showdown_set

_SHARED = {}


def shared_bridge() -> Bridge:
    """탐색용 쇼다운 프로세스는 하나를 같이 쓴다 (판마다 새로 띄우지 않음)."""
    b = _SHARED.get("b")
    if b is None or b.proc.poll() is not None:
        b = _SHARED["b"] = Bridge()
    return b


SIDE_KEYS = ("reflect", "lightscreen", "auroraveil", "tailwind", "stealthrock", "spikes", "toxicspikes", "stickyweb")


class SearchPolicy:
    name = "search"

    def __init__(self, dex: Dex, worlds=8, rollouts=4, depth=3, top_n=12, seed=None, bridge: Bridge | None = None,
                 temperature=0.02, prior=0.08, leaf="mix", net=None):
        self.dex = dex
        self.K, self.R, self.D, self.top_n = worlds, rollouts, depth, top_n
        self.rng = random.Random(seed)
        self.seed = seed or 1
        self.T = temperature
        self.prior = prior
        self.leaf = leaf
        self.net = net          # 학습된 정책망 (있으면 선생님 대신 사전확률·후보 추리기에 사용)
        self._netpol = None
        self.teacher = HeuristicPolicy(dex, temperature=0.12, seed=seed)
        self._bridge = bridge
        self.last = None

    @property
    def bridge(self):
        if self._bridge is None:
            self._bridge = shared_bridge()
        return self._bridge

    def start(self, side, team_sets, dex):
        self.side = side
        self.sets = team_sets
        self.by_name = {dex.species[s["species"]]["name"]: s for s in team_sets}
        self.teacher.start(side, team_sets, dex)
        self.beliefs = {}

    def _belief(self, sid):
        b = self.beliefs.get(sid)
        if b is None:
            b = self.beliefs[sid] = OppBelief(self.dex, sid, getattr(self, "game_type", "singles"))
        return b

    # ------------------------------------------------------------ 장면
    def my_side(self, snap, req):
        """내 쪽: 나와 있는 포켓몬 먼저, 그다음 대기 (요청 순서). 돌려줌: packed 용 세트 목록, 상태, 요청번호→장면번호"""
        dex = self.dex
        pokes = req["side"]["pokemon"]
        order = [i for i, p in enumerate(pokes) if p.get("active")] + [i for i, p in enumerate(pokes) if not p.get("active")]
        sets, states, remap = [], [], {}
        mv_by_name = {m["name"]: m for m in snap["mons"][snap["me"]]}
        for i in order:
            p = pokes[i]
            name = split_ident(p["ident"])[2]
            s = dict(self.by_name.get(name) or {})
            if not s:
                continue
            pos = len(sets) + 1
            species_now = to_id(p["details"].split(",")[0])
            if dex.species.get(species_now, {}).get("isMega"):
                s["species"] = species_now
            item_now = to_id(p.get("item") or "")
            s["item"] = item_now
            cond = p["condition"]
            fainted = cond.endswith("fnt")
            hp = None
            st = ""
            if not fainted:
                a = cond.split()
                hp = int(a[0].split("/")[0])
                st = a[1] if len(a) > 1 else ""
            mv = mv_by_name.get(name, {})
            sets.append(s)
            states.append({"hp": hp, "status": st, "fainted": fainted, "boosts": mv.get("boosts") or {},
                           "active": bool(p.get("active"))})
            remap[i + 1] = pos
        return sets, states, remap

    def world(self, snap):
        """상대 쪽 '가능한 세계' 하나."""
        dex = self.dex
        me = snap["me"]
        opp = "p2" if me == "p1" else "p1"
        mons = snap["mons"][opp]
        act = sorted([m for m in mons if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
        rest = [m for m in mons if m not in act]
        sets, states = [], []
        seen = set()
        for m in act + rest:
            sid = m["species"]
            sp = dex.species.get(sid)
            if not sp:
                continue
            base = to_id(sp.get("changesFrom") or sp["baseSpecies"]) if sp.get("isMega") else sid
            seen.add(base)
            b = self._belief(base if base in dex.species else sid)
            b.observe([x for x in m.get("moves") or [] if x in dex.moves], m.get("item"), m.get("ability"))
            s = sample_set(dex, b, self.rng, revealed_item=m.get("item"), ability=m.get("ability"),
                           mega_species=sid if sp.get("isMega") else None)
            sets.append(s)
            states.append({"hpPct": m["hpPct"], "status": m.get("status", ""), "fainted": m["fainted"],
                           "boosts": m.get("boosts") or {}})
        pick = snap.get("pickSize") or (4 if snap["gameType"] == "doubles" else 3)
        unseen = [s for s in snap["preview"].get(opp, []) if s in dex.species and s not in seen]
        # 아직 안 나온 상대: 메타 사용률이 높을수록 데려왔을 가능성이 크다고 보고 뽑는다
        w = {s: ((dex.usage_of(s, snap["gameType"]) or {}).get("usage", 0.02) + 0.02) for s in unseen}
        picked = []
        while unseen:
            s_ = self.rng.choices(unseen, weights=[w[x] for x in unseen])[0]
            picked.append(s_)
            unseen.remove(s_)
        unseen = picked
        for sid in unseen[:max(0, pick - len(sets))]:
            b = self._belief(sid)
            sets.append(sample_set(dex, b, self.rng))
            states.append({"hpPct": 100, "status": "", "fainted": False, "boosts": {}})
        return sets, states

    def scenario(self, snap, req, my_sets, my_states, worlds, options):
        me = snap["me"]
        opp = "p2" if me == "p1" else "p1"
        dex = self.dex
        pack = lambda sets: self.bridge.pack([to_showdown_set(dex, s) for s in sets])
        sc_sides = {}
        for s, key in ((me, me), (opp, opp)):
            d = snap["sideConds"].get(s, {})
            sc_sides[key] = {k: (v.get("layers", 1) if k in ("spikes", "toxicspikes") else True) for k, v in d.items() if k in SIDE_KEYS}
        sc = {
            "gameType": snap["gameType"], "mySide": me, "oppSide": opp,
            "field": {"weather": snap["weather"], "weatherTurns": snap.get("weatherTurns", 0), "terrain": snap["terrain"],
                      "terrainTurns": snap.get("terrainTurns", 0), "trickRoom": "trickroom" in snap["pseudo"],
                      "gravity": "gravity" in snap["pseudo"]},
            "sides": sc_sides, "megaUsed": snap["megaUsed"], "turn": snap["turn"],
            "options": options, "rollouts": self.R, "depth": self.D, "seed": self.rng.randrange(1, 2 ** 30),
            "leaf": self.leaf,
            "worlds": [{"packed": pack(ws), "state": wst} for ws, wst in worlds],
        }
        sc[me] = {"packed": pack(my_sets), "state": my_states}
        sc[opp] = sc["worlds"][0]
        return sc

    # ------------------------------------------------------------ 선택
    def choose(self, view, req, options):
        snap = view.snapshot() if hasattr(view, "snapshot") else view
        self.game_type = snap.get("gameType", "singles")
        if req.get("teamPreview"):
            return self.teacher.choose(view, req, options)
        teacher = self.teacher._turn_scores(snap, req, options)
        net_p = None
        if self.net is not None:
            try:
                if self._netpol is None:
                    from .nets import NetPolicy
                    self._netpol = NetPolicy(self.dex, self.net, temperature=1.0)
                    self._netpol.start(self.side, self.sets, self.dex)
                net_p, _ = self._netpol.probs(view, req, options)
            except Exception:
                net_p = None
        my_sets, my_states, remap = self.my_side(snap, req)
        worlds = [self.world(snap) for _ in range(self.K)]
        cand = list(range(len(options)))
        if len(cand) > self.top_n:
            key = (lambda i: -net_p[i]) if net_p is not None else (lambda i: -teacher[i])
            cand = sorted(cand, key=key)[:self.top_n]
        values = {}
        try:
            if req.get("forceSwitch"):
                values = self._force_switch(snap, req, options, cand, my_sets, my_states, remap, worlds)
            else:
                mapped = [self._remap(options[i]["choice"], remap) for i in cand]
                sc = self.scenario(snap, req, my_sets, my_states, worlds, mapped)
                res = self.bridge.call(op="search", **sc)
                for i, r in zip(cand, res.get("results") or []):
                    if r.get("value") is not None and not (r.get("invalid") and r["n"] == 0):
                        values[i] = r["value"]
        except Exception as e:  # 탐색이 실패하면 선생님 판단으로
            import os
            import traceback
            self.last = {"error": str(e), "trace": traceback.format_exc()}
            self.errors = getattr(self, "errors", 0) + 1
            if os.environ.get("SEARCH_DEBUG"):
                raise
            return self.teacher.choose(view, req, options)
        if not values:
            return self.teacher.choose(view, req, options)
        # 선생님 판단을 사전확률로 섞는다 (알파고의 정책망 사전확률처럼): 값 + prior × 선생님 확률
        if net_p is not None:
            pt = [float(x) for x in net_p]
        else:
            mx_t = max(teacher)
            ex_t = [math.exp((t - mx_t) / 0.12) for t in teacher]
            pt = [e / sum(ex_t) for e in ex_t]
        score = []
        for i in range(len(options)):
            if i in values:
                score.append(values[i] + self.prior * pt[i])
            else:
                score.append(-1.0)
        m = max(score)
        ex = [math.exp((s - m) / max(self.T, 1e-4)) if s > -1 else 0.0 for s in score]
        tot = sum(ex) or 1.0
        probs = [e / tot for e in ex]
        self.last = {"values": values, "teacher": teacher}
        r, acc = self.rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i, probs
        return max(range(len(probs)), key=lambda i: probs[i]), probs

    def _remap(self, choice, remap):
        parts = []
        for p in choice.split(","):
            p = p.strip()
            if p.startswith("switch"):
                k = int(p.split()[1])
                parts.append(f"switch {remap.get(k, k)}")
            else:
                parts.append(p)
        return ", ".join(parts)

    def _force_switch(self, snap, req, options, cand, my_sets, my_states, remap, worlds):
        """강제 교체: 후보마다 그 포켓몬이 나와 있는 장면을 만들어 굴려 본다."""
        values = {}
        for i in cand:
            ch = options[i]["choice"]
            order = list(range(1, len(my_sets) + 1))
            ins = [remap.get(int(p.split()[1])) for p in ch.split(",") if p.strip().startswith("switch")]
            ins = [x for x in ins if x]
            # 남아 있는 활성 포켓몬 + 들어갈 포켓몬을 앞으로 (장면의 선봉 자리)
            alive_front = [pos for pos in order if pos not in ins and my_states[pos - 1].get("active")
                           and not my_states[pos - 1]["fainted"]]
            new_order = alive_front + ins + [p for p in order if p not in ins and p not in alive_front]
            s2 = [my_sets[p - 1] for p in new_order]
            st2 = [my_states[p - 1] for p in new_order]
            sc = self.scenario(snap, req, s2, st2, worlds, [None])
            res = self.bridge.call(op="search", **sc)
            r = (res.get("results") or [{}])[0]
            if r.get("value") is not None:
                values[i] = r["value"]
        return values
