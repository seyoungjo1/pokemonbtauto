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

    def __init__(self, dex: Dex, species: str):
        self.dex = dex
        self.species = dex.find("species", species) or to_id(species)
        self.cands = self._prior()

    def _base_id(self):
        sp = self.dex.species.get(self.species, {})
        return to_id(sp.get("changesFrom") or sp.get("baseSpecies") or self.species)

    def _prior(self) -> list:
        dex = self.dex
        out = []
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
            if any(m not in c.moves for m in revealed_moves):
                if c.role == "(배울 수 있는 기술)" and all(m in learn for m in revealed_moves):
                    c.moves = list(dict.fromkeys(list(revealed_moves) + c.moves))
                else:
                    continue
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
        """기술별 '가지고 있을' 확률 (후보 가중 평균, 4칸 가정)."""
        tot = sum(c.weight for c in self.cands) or 1
        out = {}
        for c in self.cands:
            n = len(c.moves)
            for m in c.moves:
                p = 1.0 if m in getattr(self, "revealed", []) else min(1.0, 4.0 / max(4, n))
                out[m] = out.get(m, 0) + c.weight * p / tot
        for m in getattr(self, "revealed", []):
            out[m] = 1.0
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def likely_role(self) -> str:
        return max(self.cands, key=lambda c: c.weight).role if self.cands else ""

    def likely_ability(self) -> str:
        if getattr(self, "ability", None):
            return self.ability
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
        moves = list(self.move_probs())[:6]
        it = item if item is not None else (getattr(self, "item", None) or "")
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
