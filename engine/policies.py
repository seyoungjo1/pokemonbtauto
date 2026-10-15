# -*- coding: utf-8 -*-
"""자기대전용 정책(행동을 고르는 방법).

- RandomPolicy    : 무작위 (기준선)
- HeuristicPolicy : 데미지 엔진 + 상대 세트 추정으로 점수를 매겨 고른다 (초기 학습용 '선생님')
- ModelPolicy     : 학습된 가치망·정책망 (engine/model.py) — 학습이 진행되면 이것끼리 둔다

모든 정책은 choose(view, req, options) → (고른 번호, 확률 목록) 을 돌려준다.
확률 목록은 학습 데이터(정책망 정답)로 기록된다.
"""
from __future__ import annotations

import math
import random

from .belief import OppBelief, mon_from_view
from .damage import Ctx, calc
from .dex import Dex, to_id
from .mon import FieldState, Mon, SideState
from .protocol import split_ident
from .teamgen import normalize_set

SETUP_BOOST_MOVES = {"swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup", "quiverdance",
                     "shellsmash", "coil", "irondefense", "agility", "shiftgear", "workup", "tailglow",
                     "victorydance", "tidyup", "bellydrum", "growth", "curse", "rockpolish", "autotomize",
                     "clangoroussoul", "noretreat", "filletaway"}
RECOVERY = {"recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun", "synthesis",
            "milkdrink", "shoreup", "strengthsap", "rest", "wish", "lifedew"}
STATUS_INFLICT = {"toxic": "tox", "willowisp": "brn", "thunderwave": "par", "spore": "slp",
                  "sleeppowder": "slp", "hypnosis": "slp", "yawn": "slp", "glare": "par",
                  "stunspore": "par", "nuzzle": "par", "toxicthread": "psn", "poisonpowder": "psn"}
HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
SCREENS = {"reflect", "lightscreen", "auroraveil"}
PROTECT = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
           "burningbulwark", "obstruct", "maxguard"}


class RandomPolicy:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def start(self, side, team_sets, dex):
        pass

    def choose(self, view, req, options):
        i = self.rng.randrange(len(options))
        return i, None


class HeuristicPolicy:
    name = "heuristic"

    def __init__(self, dex: Dex, temperature: float = 0.12, seed=None):
        self.dex = dex
        self.T = temperature
        self.rng = random.Random(seed)
        self.beliefs = {}

    def start(self, side, team_sets, dex):
        self.side = side
        self.sets = {dex.ident_name(s["species"]): normalize_set(dex, s) for s in team_sets}
        self.beliefs = {}

    # ------------------------------------------------------------ 상태 → 계산용 객체
    def _belief(self, species):
        b = self.beliefs.get(species)
        if b is None:
            b = self.beliefs[species] = OppBelief(self.dex, species, getattr(self, "game_type", "singles"))
        return b

    def my_mon(self, mv: dict) -> Mon | None:
        s = self.sets.get(mv["name"])
        if not s:
            return None
        return mon_from_view(self.dex, mv, known_set=s)

    def opp_mon(self, mv: dict) -> Mon:
        base = self.dex.species.get(mv["species"], {})
        key = to_id(base.get("changesFrom") or base.get("baseSpecies") or mv["species"]) if base.get("isMega") else mv["species"]
        return mon_from_view(self.dex, mv, belief=self._belief(mv["species"] if not base.get("isMega") else mv["species"]))

    def field_of(self, snap) -> FieldState:
        me, opp = snap["me"], ("p2" if snap["me"] == "p1" else "p1")
        sides = []
        for s in (me, opp):
            sc = snap["sideConds"].get(s, {})
            sides.append(SideState(reflect="reflect" in sc, lightscreen="lightscreen" in sc,
                                   auroraveil="auroraveil" in sc, tailwind="tailwind" in sc,
                                   fainted=snap["fainted"].get(s, 0)))
        return FieldState(game_type=snap["gameType"], weather=snap["weather"], terrain=snap["terrain"],
                          gravity="gravity" in snap["pseudo"], trick_room="trickroom" in snap["pseudo"], sides=sides)

    # ------------------------------------------------------------ 평가 도우미
    def exp_frac(self, a: Mon, d: Mon, move: str, field, a_side=0, ally_a=None, ally_d=None, spread=None):
        """a 가 d 에게 move 를 쓸 때 기대 데미지(현재 HP 대비, 0~1)와 처치 확률."""
        if d.hp <= 0:
            return 0.0, 0.0
        ctx = Ctx(atk_side=a_side, def_side=1 - a_side, actives=[x for x in (a, d, ally_a, ally_d) if x],
                  atk_ally=ally_a, def_ally=ally_d, spread=spread)
        try:
            r = calc(self.dex, a, d, move, field, ctx)
        except Exception:
            return 0.0, 0.0
        if r.immune or r.blocked or not r.rolls:
            return 0.0, 0.0
        acc = self.dex.moves[r.move].get("accuracy")
        acc = 1.0 if acc is True else (acc or 100) / 100.0
        hits = sum(r.hits) / len(r.hits)
        mean = sum(r.rolls[0]) / 16.0 * hits * (1 + 0.5 * r.crit_chance)
        lo = min(r.rolls[0]) * min(r.hits)
        hi = max(r.rolls[0]) * max(r.hits)
        if lo >= d.hp:
            ko = 1.0
        elif hi < d.hp:
            ko = r.crit_chance if max(r.crit_rolls[0]) * max(r.hits) >= d.hp else 0.0
        else:
            ko = sum(1 for v in r.rolls[0] if v * hits >= d.hp) / 16.0
        return min(1.0, mean / d.hp) * acc, ko * acc

    def best_vs(self, a: Mon, d: Mon, field, a_side=0, moves=None):
        best = (0.0, 0.0, "")
        for mv in (moves or a.moves):
            if mv not in self.dex.moves or self.dex.moves[mv]["category"] == "Status":
                continue
            f, ko = self.exp_frac(a, d, mv, field, a_side)
            if f + ko > best[0] + best[1]:
                best = (f, ko, mv)
        return best

    def speed(self, m: Mon, field, side):
        from .damage import _Calc
        c = _Calc(self.dex, m, m, next(iter(self.dex.moves)), field, Ctx(atk_side=side))
        s = c.spe_stat(m, side)
        return -s if field.trick_room else s

    # ------------------------------------------------------------ 선택
    def choose(self, view, req, options):
        snap = view.snapshot()
        if req.get("teamPreview"):
            scores = self._preview_scores(snap, req, options)
        else:
            scores = self._turn_scores(snap, req, options)
        mx = max(scores)
        ex = [math.exp((s - mx) / max(self.T, 1e-6)) for s in scores]
        tot = sum(ex)
        probs = [e / tot for e in ex]
        r, acc = self.rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i, probs
        return len(probs) - 1, probs

    def _preview_scores(self, snap, req, options):
        opp = "p2" if snap["me"] == "p1" else "p1"
        field = FieldState(game_type=snap["gameType"])
        opps = [self._belief(s).assumed_mon() for s in snap["preview"][opp] if s in self.dex.species]
        mine = []
        for p in req["side"]["pokemon"]:
            _, _, name = split_ident(p["ident"])
            s = self.sets.get(name)
            m = Mon(species=s["species"], ability=s["ability"], item=s["item"], nature=s["nature"],
                    sp=s["sp"], moves=s["moves"]).setup(self.dex) if s else None
            if m and s.get("megaTarget"):
                m.mega_evolve()
            mine.append(m)
        mu = []
        for m in mine:
            if not m:
                mu.append(0.0)
                continue
            v = 0.0
            for o in opps:
                f1, k1, _ = self.best_vs(m, o, field, 0)
                f2, k2, _ = self.best_vs(o, m, field, 1)
                v += (f1 + k1) - (f2 + k2)
            mu.append(v / max(1, len(opps)))
        out = []
        for opt in options:
            order = opt["parts"]
            lead_n = 1 if len(order) <= 3 else 2
            out.append(sum(mu[i - 1] for i in order) / len(order) + 0.5 * sum(mu[i - 1] for i in order[:lead_n]) / lead_n)
        return out

    def _turn_scores(self, snap, req, options):
        me = snap["me"]
        opp = "p2" if me == "p1" else "p1"
        field = self.field_of(snap)
        my_act = sorted([m for m in snap["mons"][me] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
        op_act = sorted([m for m in snap["mons"][opp] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
        my_mons = [self.my_mon(m) for m in my_act]
        op_mons = [self.opp_mon(m) for m in op_act]
        pokemon = req["side"]["pokemon"]
        actives_req = req.get("active") or []
        doubles = snap["gameType"] == "doubles"
        cache = {}

        def bench_mon(k):
            p = pokemon[k - 1]
            _, _, name = split_ident(p["ident"])
            mv = next((m for m in snap["mons"][me] if m["name"] == name), None)
            if mv is None:
                s = self.sets.get(name)
                if not s:
                    return None
                mv = {"name": name, "species": s["species"], "hpPct": 100}
            return self.my_mon(mv)

        def part_score(i, part):
            key = (i, part)
            if key in cache:
                return cache[key]
            v = self._part(i, part, my_mons, op_mons, field, actives_req, bench_mon, doubles)
            cache[key] = v
            return v

        return [sum(part_score(i, p) for i, p in enumerate(o["parts"])) for o in options]

    def _part(self, i, part, my_mons, op_mons, field, actives_req, bench_mon, doubles):
        if part == "pass":
            return 0.0
        ops = [o for o in op_mons if o and o.hp > 0]
        if part.startswith("switch"):
            k = int(part.split()[1])
            m = bench_mon(k)
            if not m or not ops:
                return -0.5
            v = 0.0
            for o in ops:
                f1, k1, _ = self.best_vs(m, o, field, 0)
                f2, k2, _ = self.best_vs(o, m, field, 1)
                v += (f1 + k1) - 1.2 * (f2 + k2)
            return v / len(ops) - 0.25 + (0.3 if (i < len(my_mons) and my_mons[i] and my_mons[i].hp <= 0) else 0)
        # move j [target] [mega]
        toks = part.split()
        j = int(toks[1])
        tgt = None
        mega = toks[-1] == "mega"
        if len(toks) >= 3 and toks[2].lstrip("-").isdigit():
            tgt = int(toks[2])
        if i >= len(my_mons) or my_mons[i] is None or i >= len(actives_req):
            return 0.0
        me_mon = my_mons[i]
        if mega and not me_mon.mega:
            me_mon = Mon.from_dict(self.dex, me_mon.to_dict())
            me_mon.mega_evolve()
        mvinfo = actives_req[i]["moves"][j - 1] if j - 1 < len(actives_req[i].get("moves", [])) else {}
        mid = to_id(mvinfo.get("id", ""))
        mv = self.dex.moves.get(mid)
        if not mv:
            return 0.0
        bonus = 0.08 if mega else 0.0
        if mv["category"] == "Status":
            return bonus + self._status_value(mid, me_mon, ops, field, doubles)
        if tgt is not None and tgt < 0:
            return -1.0   # 아군 공격
        targets = ops if (mv.get("target") in ("allAdjacentFoes", "allAdjacent") or tgt is None) else \
            [op_mons[tgt - 1]] if 0 < tgt <= len(op_mons) and op_mons[tgt - 1] else ops[:1]
        spread = doubles and mv.get("target") in ("allAdjacentFoes", "allAdjacent") and len(ops) > 1
        v = 0.0
        my_spe = self.speed(me_mon, field, 0)
        for o in targets:
            f, ko = self.exp_frac(me_mon, o, mid, field, 0, spread=spread)
            # 상대가 먼저 쓰러뜨릴 위험
            of, ok, _ = self.best_vs(o, me_mon, field, 1)
            first = (mv.get("priority", 0) > 0) or my_spe > self.speed(o, field, 1)
            survive = 1.0 if first else (1.0 - ok)
            v += (f + 0.7 * ko) * survive
        if mv.get("target") == "allAdjacent" and doubles and len(my_mons) > 1:
            ally = my_mons[1 - i] if len(my_mons) > 1 else None
            if ally and ally.hp > 0:
                af, ak = self.exp_frac(me_mon, ally, mid, field, 0, spread=True)
                v -= 1.2 * (af + ak)
        if mv.get("recoil") or mv.get("mindBlownRecoil"):
            v -= 0.05
        return v + bonus

    def _status_value(self, mid, me_mon, ops, field, doubles):
        if not ops:
            return 0.0
        worst = max(self.best_vs(o, me_mon, field, 1)[0] for o in ops)
        hp = me_mon.hp / me_mon.maxhp
        if mid in SETUP_BOOST_MOVES:
            return 0.45 if worst < 0.35 and hp > 0.6 else 0.05
        if mid in RECOVERY:
            return 0.5 * (1 - hp) if hp < 0.6 else 0.0
        if mid in STATUS_INFLICT:
            o = ops[0]
            return 0.3 if not o.status else -0.2
        if mid in PROTECT:
            return 0.3 if doubles else 0.05
        if mid in HAZARDS:
            return 0.2
        if mid in SCREENS:
            return 0.25
        if mid in ("tailwind", "trickroom"):
            return 0.3
        return 0.05


# ======================================================================== 학습하지 않은 기본 AI들 (대전 상대 풀)
class ShowdownRandomPolicy:
    """쇼다운 내장 RandomPlayerAI 와 같은 방식 — 기술만 무작위, 교체는 강제될 때만, 메가진화 안 함."""
    name = "showdown_random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def start(self, side, team_sets, dex):
        pass

    def choose(self, view, req, options):
        if req.get("teamPreview") or req.get("forceSwitch"):
            return self.rng.randrange(len(options)), None
        moves_only = [i for i, o in enumerate(options)
                      if not any(p.startswith("switch") for p in o["parts"]) and not o["choice"].endswith("mega")
                      and " mega," not in o["choice"]]
        pool = moves_only or list(range(len(options)))
        return self.rng.choice(pool), None


class MaxDamagePolicy:
    """매 턴 가장 센 기술(기본 위력 × 자속 × 상성)만 쓴다 — 연구에서 흔한 기준선."""
    name = "max_damage"

    def __init__(self, dex: Dex, seed=None):
        self.dex = dex
        self.rng = random.Random(seed)

    def start(self, side, team_sets, dex):
        pass

    def choose(self, view, req, options):
        if req.get("teamPreview") or req.get("forceSwitch"):
            return self.rng.randrange(len(options)), None
        snap = view.snapshot()
        opp = "p2" if snap["me"] == "p1" else "p1"
        opps = sorted([m for m in snap["mons"][opp] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
        mine = [p for p in req["side"]["pokemon"] if p.get("active")]
        best, best_i = -1.0, 0
        for i, o in enumerate(options):
            v = 0.0
            for slot, part in enumerate(o["parts"]):
                toks = part.split()
                if toks[0] != "move" or slot >= len(req.get("active", [])):
                    continue
                mv = self.dex.moves.get(to_id(req["active"][slot]["moves"][int(toks[1]) - 1].get("id", "")))
                if not mv or mv["category"] == "Status":
                    continue
                my_types = self.dex.species.get(to_id(mine[slot]["details"].split(",")[0]) if slot < len(mine) else "", {}).get("types", [])
                tgt = int(toks[2]) if len(toks) > 2 and toks[2].lstrip("-").isdigit() else 1
                if tgt < 0:
                    continue
                o_types = self.dex.species.get(opps[tgt - 1]["species"], {}).get("types", []) if 0 < tgt <= len(opps) else []
                mult = 1.0
                for t in o_types:
                    if self.dex.immune(mv["type"], t):
                        mult = 0.0
                        break
                    mult *= 2 ** self.dex.effectiveness(mv["type"], t)
                v += (mv.get("basePower") or 0) * (1.5 if mv["type"] in my_types else 1.0) * mult
            if v > best or (v == best and self.rng.random() < 0.5):
                best, best_i = v, i
        return best_i, None


BASELINES = {
    "showdown_random": "쇼다운 기본 AI (무작위)",
    "random": "완전 무작위",
    "max_damage": "최대 데미지 AI",
    "heuristic": "선생님 AI (데미지 계산)",
}


def make_baseline(name: str, dex: Dex, seed=None):
    if name == "showdown_random":
        return ShowdownRandomPolicy(seed)
    if name == "random":
        return RandomPolicy(seed)
    if name == "max_damage":
        return MaxDamagePolicy(dex, seed)
    return HeuristicPolicy(dex, seed=seed)
