# -*- coding: utf-8 -*-
"""상대 세트 추정 (불완전 정보).

처음엔 상대의 기술·도구·특성·SP 를 모른다. 쇼다운 챔피언스 랜덤배틀 세트(역할별 기술 후보)와
배울 수 있는 기술을 사전확률로 두고, 대전 중 드러난 정보(쓴 기술, 발동한 도구·특성, 메가진화)와
맞지 않는 후보를 지워 나간다. 받은/준 데미지로 SP·도구 후보를 좁히는 것은 narrow_by_damage().
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield

from .dex import Dex, to_id
from .mon import Mon

ROLE_SPREADS = {
    "fast": [({"atk": 32, "spe": 32, "hp": 2}, "jolly"), ({"spa": 32, "spe": 32, "hp": 2}, "timid"),
             ({"atk": 32, "spe": 32, "hp": 2}, "adamant"), ({"spa": 32, "spe": 32, "hp": 2}, "modest")],
    "bulky": [({"hp": 32, "atk": 32, "def": 2}, "adamant"), ({"hp": 32, "spa": 32, "spd": 2}, "modest"),
              ({"hp": 32, "def": 17, "spd": 17}, "careful"), ({"hp": 32, "def": 17, "spd": 17}, "bold")],
}


@dataclass
class Candidate:
    role: str
    moves: list                 # 후보 기술 (4개 넘을 수 있음 — movepool)
    abilities: list
    weight: float = 1.0
    spreads: list = dfield(default_factory=list)


class OppBelief:
    """상대 포켓몬 한 마리에 대한 믿음."""

    def __init__(self, dex: Dex, species: str, game_type: str = "singles"):
        self.dex = dex
        self.species = dex.find("species", species) or to_id(species)
        self.game_type = game_type
        self.usage = dex.usage_of(self.species, game_type) or dex.usage_of(self.species, "singles")
        self.cands = self._prior()

    def _base_id(self):
        sp = self.dex.species.get(self.species, {})
        return to_id(sp.get("changesFrom") or sp.get("baseSpecies") or self.species)

    def _prior(self) -> list:
        dex = self.dex
        out = []
        for sid in (self.species, self._base_id()):
            for s in ((dex.expert.get(self.game_type) or {}).get(sid) or []):
                mv = [m for slot in s["moves"] for m in slot]
                out.append(Candidate(role="전문가 세트: " + s["label"], moves=mv, abilities=s["abilities"], weight=2.0))
            if out:
                break
        for sid in (self.species, self._base_id()):
            for s in (dex.sets.get(sid) or {}).get("sets", []):
                mv = [dex.find("moves", m) for m in s["movepool"]]
                ab = [dex.find("abilities", a) for a in s.get("abilities") or []]
                out.append(Candidate(role=s["role"], moves=[m for m in mv if m], abilities=[a for a in ab if a]))
            if out:
                break
        sp = dex.species.get(self.species)
        learn = dex.learnsets.get(self.species) or dex.learnsets.get(self._base_id()) or []
        if sp:
            # 세트에 없는 기술을 쓸 수도 있으므로 '배울 수 있는 모든 기술' 후보를 작은 확률로 둔다
            strong = sorted((m for m in learn if dex.moves[m]["category"] != "Status"),
                            key=lambda m: -(dex.moves[m]["basePower"] or 0))
            stab = [m for m in strong if dex.moves[m]["type"] in sp["types"]][:4]
            other = [m for m in strong if m not in stab][:6]
            out.append(Candidate(role="(배울 수 있는 기술)", moves=stab + other,
                                 abilities=[to_id(a) for a in sp["abilities"].values()], weight=0.15))
        return out

    # ------------------------------------------------------------ 관측 반영
    def observe(self, revealed_moves=(), item=None, ability=None):
        learn = set(self.dex.learnsets.get(self.species) or self.dex.learnsets.get(self._base_id()) or [])
        keep = []
        for c in self.cands:
            missing = [m for m in revealed_moves if m not in c.moves]
            if missing:
                if not all(m in learn for m in missing):
                    continue
                # 세트 밖 기술을 썼다 — 버리지 않고 가중치를 낮춰 그 기술을 더한다
                c.moves = list(dict.fromkeys(list(revealed_moves) + c.moves))
                if c.role != "(배울 수 있는 기술)" and not getattr(c, "_extended", False):
                    c.weight *= 0.4
                    c._extended = True
            if ability and c.abilities and ability not in c.abilities:
                continue
            keep.append(c)
        if not keep:   # 전부 지워지면 배울 수 있는 기술 후보로 다시
            sp = self.dex.species.get(self.species, {})
            keep = [Candidate(role="(관측 기반)", moves=list(dict.fromkeys(list(revealed_moves))),
                              abilities=[ability] if ability else [to_id(a) for a in sp.get("abilities", {}).values()])]
        self.cands = keep
        self.revealed = list(revealed_moves)
        self.item = item
        self.ability = ability

    # ------------------------------------------------------------ 요약
    def move_probs(self) -> dict:
        """기술별 '가지고 있을' 확률. 메타 사용률이 있으면 그것을 주로(80%), 세트 후보를 보조로(20%)."""
        tot = sum(c.weight for c in self.cands) or 1
        sets_p = {}
        for c in self.cands:
            n = len(c.moves)
            for m in c.moves:
                sets_p[m] = sets_p.get(m, 0) + c.weight * min(1.0, 4.0 / max(4, n)) / tot
        revealed = list(getattr(self, "revealed", []) or [])
        if self.usage and self.usage.get("moves"):
            out = {m: 0.8 * p for m, p in self.usage["moves"].items()}
            for m, p in sets_p.items():
                out[m] = out.get(m, 0) + 0.2 * p
        else:
            out = dict(sets_p)
        # 드러난 기술은 확실(1), 나머지 칸은 남은 칸 수에 맞춰 다시 나눈다
        free = max(0, 4 - len(revealed))
        rest = {m: p for m, p in out.items() if m not in revealed}
        s = sum(rest.values()) or 1
        scale = min(1.0, free / s) if free else 0.0
        out = {m: min(1.0, p * (free / s if s > free else 1.0)) if free else 0.0 for m, p in rest.items()}
        for m in revealed:
            out[m] = 1.0
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def item_probs(self) -> dict:
        if getattr(self, "item", None) is not None:
            return {self.item: 1.0}
        if self.usage and self.usage.get("items"):
            return dict(self.usage["items"])
        return {}

    def spread_probs(self) -> list:
        return list((self.usage or {}).get("spreads") or [])

    def likely_role(self) -> str:
        return max(self.cands, key=lambda c: c.weight).role if self.cands else ""

    def likely_ability(self) -> str:
        if getattr(self, "ability", None):
            return self.ability
        if self.usage and self.usage.get("abilities"):
            return max(self.usage["abilities"], key=self.usage["abilities"].get)
        cnt = {}
        for c in self.cands:
            for a in c.abilities:
                cnt[a] = cnt.get(a, 0) + c.weight
        if cnt:
            return max(cnt, key=cnt.get)
        sp = self.dex.species.get(self.species, {})
        return to_id(sp.get("abilities", {}).get("0", ""))

    def assumed_mon(self, hp_pct=100.0, status="", boosts=None, item=None, mega=False) -> Mon:
        """계산에 쓸 '가장 그럴듯한' 상대 포켓몬."""
        dex = self.dex
        sp = dex.species[self.species]
        bs = sp["baseStats"]
        fast = bs["spe"] >= 90
        phys = bs["atk"] >= bs["spa"]
        spread, nature = ROLE_SPREADS["fast" if fast else "bulky"][(0 if phys else 1)]
        sps = self.spread_probs()
        if sps:
            spread, nature = sps[0]["sp"], sps[0]["nature"] or nature
        moves = list(self.move_probs())[:6]
        ip = self.item_probs()
        it = item if item is not None else (getattr(self, "item", None) or (max(ip, key=ip.get) if ip else ""))
        m = Mon(species=self.species, ability=self.likely_ability(), item=it or "", nature=nature,
                sp=dict(spread), moves=moves)
        m.setup(dex)
        m.set_hp_pct(hp_pct)
        m.status = status or ""
        m.boosts = dict(boosts or {})
        return m


def mon_from_view(dex: Dex, mv: dict, belief: OppBelief | None = None, known_set: dict | None = None) -> Mon:
    """BattleView 스냅샷의 포켓몬 한 마리 → 계산용 Mon."""
    if known_set:   # 내 포켓몬: 세트를 정확히 안다
        m = Mon(species=known_set["species"], ability=known_set.get("ability", ""), item=known_set.get("item", ""),
                nature=known_set.get("nature", "serious"), sp=dict(known_set.get("sp") or {}),
                moves=list(known_set.get("moves") or []), gender=known_set.get("gender", ""))
        m.setup(dex)
        if mv.get("mega") or (dex.species.get(mv.get("species", ""), {}).get("isMega")):
            m.mega_evolve()
        if mv.get("item") is not None and mv.get("item") != m.item:
            m.item = mv.get("item") or ""
        if mv.get("hp") is not None and mv.get("maxhp"):
            m.hp = int(mv["hp"] * m.maxhp / mv["maxhp"]) if mv["maxhp"] != m.maxhp else int(mv["hp"])
        else:
            m.set_hp_pct(mv.get("hpPct", 100))
    else:
        b = belief or OppBelief(dex, mv["species"])
        b.observe(mv.get("moves") or [], mv.get("item"), mv.get("ability"))
        m = b.assumed_mon(hp_pct=mv.get("hpPct", 100), item=mv.get("item"))
        if mv.get("species") and mv["species"] != m.species and mv["species"] in dex.species:
            m.species = mv["species"]
            m.recalc()
            m.set_hp_pct(mv.get("hpPct", 100))
    m.status = mv.get("status", "") or ""
    m.boosts = {k: v for k, v in (mv.get("boosts") or {}).items() if v}
    m.volatiles = set(mv.get("volatiles") or [])
    m.times_attacked = mv.get("timesAttacked", 0)
    return m


# ======================================================================== 세트 뽑기 (탐색용 '가능한 세계')
COMMON_ITEMS = [("focussash", 3), ("sitrusberry", 3), ("lifeorb", 3), ("leftovers", 2), ("choicescarf", 2),
                ("expertbelt", 1.5), ("lumberry", 1.5), ("TYPE", 2), ("RESIST", 1.5), ("", 0.5)]


def sample_set(dex: Dex, b: OppBelief, rng, revealed_item=None, ability=None, mega_species=None) -> dict:
    """상대 한 마리의 세트를 추정 확률대로 하나 뽑는다."""
    from .teamgen import TYPE_ITEM, RESIST_BERRY
    sp = dex.species[b.species]
    probs = b.move_probs()
    revealed = list(getattr(b, "revealed", []) or [])
    moves = [m for m in revealed if m in dex.moves][:4]
    pool = [(m, p) for m, p in probs.items() if m not in moves and m in dex.moves]
    while len(moves) < 4 and pool:
        tot = sum(p for _, p in pool)
        r = rng.random() * tot
        for i, (m, p) in enumerate(pool):
            r -= p
            if r <= 0:
                moves.append(m)
                pool.pop(i)
                break
        else:
            moves.append(pool.pop()[0])
    # 도구
    if revealed_item is not None:
        item = revealed_item
    else:
        base = dex.species.get(to_id(sp.get("changesFrom") or sp.get("baseSpecies") or b.species), sp)
        stones = [dex.find("items", dex.species[to_id(f)].get("requiredItem"))
                  for f in base.get("otherFormes", []) if dex.species.get(to_id(f), {}).get("isMega")]
        stones = [s for s in stones if s]
        ip = b.item_probs()
        if mega_species:
            item = dex.find("items", dex.species[mega_species].get("requiredItem")) or ""
        elif ip:
            r = rng.random() * sum(ip.values())
            item = ""
            for it, w in ip.items():
                r -= w
                if r <= 0:
                    item = it
                    break
        elif stones and rng.random() < 0.45:
            item = rng.choice(stones)
        else:
            tot = sum(w for _, w in COMMON_ITEMS)
            r = rng.random() * tot
            item = ""
            for it, w in COMMON_ITEMS:
                r -= w
                if r <= 0:
                    item = it
                    break
            if item == "TYPE":
                dmg_types = [dex.moves[m]["type"] for m in moves if dex.moves[m]["category"] != "Status"]
                item = TYPE_ITEM.get(rng.choice(dmg_types), "") if dmg_types else ""
            elif item == "RESIST":
                weak = [t for t in dex.typechart if t not in ("Stellar", "???")
                        and sum(dex.effectiveness(t, d) for d in sp["types"]) >= 1
                        and not any(dex.immune(t, d) for d in sp["types"])]
                item = RESIST_BERRY.get(rng.choice(weak), "") if weak else ""
    # SP · 성격
    bs = sp["baseStats"]
    phys = sum(1 for m in moves if dex.moves[m]["category"] == "Physical") >= \
        sum(1 for m in moves if dex.moves[m]["category"] == "Special")
    fast = bs["spe"] >= 80 or rng.random() < 0.3
    pool_sp = ROLE_SPREADS["fast" if fast else "bulky"]
    cand = [x for x in pool_sp if (("atk" in x[0]) == phys) or "def" in x[0]] or pool_sp
    spread, nature = rng.choice(cand)
    sps = b.spread_probs()
    if sps:
        r = rng.random() * sum(x["p"] for x in sps)
        for x in sps:
            r -= x["p"]
            if r <= 0:
                spread, nature = x["sp"], x["nature"] or nature
                break
    return {"species": mega_species or b.species, "ability": ability or b.likely_ability(), "item": item,
            "nature": nature, "sp": dict(spread), "moves": moves}
