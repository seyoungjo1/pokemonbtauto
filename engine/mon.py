# -*- coding: utf-8 -*-
"""포켓몬·필드 상태와 능력치 계산 (포켓몬 챔피언스 규칙).

챔피언스 능력치 (레벨 50 고정, 노력치 대신 능력 포인트 SP 0~32, 합계 66, 개체값 없음):
    HP      = 종족값 + SP + 75
    그 외   = (종족값 + SP + 20) × 성격(1.1 / 0.9, 내림)
쇼다운 data/mods/champions/scripts.ts 의 statModify 와 같은 식이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield

from .dex import STATS, Dex, to_id

BOOST_TABLE = [1, 1.5, 2, 2.5, 3, 3.5, 4]
STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")


def trunc(x) -> int:
    """쇼다운 trunc (부호 없는 32비트 내림)."""
    return int(x) & 0xFFFFFFFF if x >= 0 else 0


def mod4096(num, den=1) -> int:
    return trunc(num * 4096 / den)


def modify(value: int, m4096: int) -> int:
    """쇼다운 battle.modify — 4096 단위 배율을 반올림 규칙대로 적용."""
    return trunc((trunc(value * m4096) + 2048 - 1) / 4096)


def chain(m: int, nxt: int) -> int:
    """쇼다운 chainModify — 배율을 이어 붙인다 (둘 다 4096 단위)."""
    return (m * nxt + 2048) >> 12


def calc_stats(base: dict, sp: dict, nature: dict | None) -> dict:
    out = {}
    for s in STATS:
        v = base[s] + int(sp.get(s, 0))
        if s == "hp":
            out[s] = v + 75
            continue
        v += 20
        if nature and nature.get("plus") == s:
            v = trunc(trunc(v * 110) / 100)
        elif nature and nature.get("minus") == s:
            v = trunc(trunc(v * 90) / 100)
        out[s] = v
    return out


@dataclass
class Mon:
    species: str
    ability: str = ""
    item: str = ""
    nature: str = "serious"
    sp: dict = dfield(default_factory=dict)
    moves: list = dfield(default_factory=list)
    gender: str = ""
    level: int = 50
    # ---- 대전 중 상태
    hp: int | None = None          # 현재 HP (None = 가득)
    status: str = ""
    boosts: dict = dfield(default_factory=dict)
    volatiles: set = dfield(default_factory=set)
    times_attacked: int = 0
    mega: bool = False             # 메가진화 했는가
    types_override: list | None = None
    # 계산 결과 (setup 에서 채움)
    stats: dict = dfield(default_factory=dict)
    maxhp: int = 0
    _dex: Dex | None = dfield(default=None, repr=False, compare=False)

    # ------------------------------------------------------------
    def setup(self, dex: Dex) -> "Mon":
        self._dex = dex
        self.species = dex.find("species", self.species) or to_id(self.species)
        self.ability = dex.find("abilities", self.ability) or to_id(self.ability)
        self.item = dex.find("items", self.item) or to_id(self.item)
        self.nature = dex.find("natures", self.nature) or to_id(self.nature) or "serious"
        self.moves = [dex.find("moves", m) or to_id(m) for m in self.moves]
        sp = dex.species[self.species]
        if not self.ability:
            self.ability = to_id(sp["abilities"].get("0", ""))
        self.recalc()
        return self

    def recalc(self):
        sp = self.sp_data
        self.stats = calc_stats(sp["baseStats"], self.sp, self._dex.natures.get(self.nature))
        old_max = self.maxhp
        self.maxhp = self.stats["hp"]
        if self.hp is None:
            self.hp = self.maxhp
        elif old_max and old_max != self.maxhp:
            self.hp = max(0, round(self.hp * self.maxhp / old_max))

    def mega_evolve(self) -> bool:
        target = self._dex.mega_for(self.species, self.item)
        if not target or target not in self._dex.species:
            return False
        self.species = target
        self.ability = to_id(self._dex.species[target]["abilities"].get("0", ""))
        self.mega = True
        hp_lost = self.maxhp - self.hp
        self.recalc()
        self.hp = self.maxhp - hp_lost  # 메가진화해도 HP 최대치는 같다 (HP 종족값 불변)
        return True

    # ------------------------------------------------------------
    @property
    def sp_data(self) -> dict:
        return self._dex.species[self.species]

    @property
    def types(self) -> list:
        return self.types_override or self.sp_data["types"]

    def has_type(self, t: str) -> bool:
        return t in self.types

    @property
    def weighthg(self) -> int:
        return max(1, int(round(self.sp_data["weightkg"] * 10)))

    @property
    def base_species(self) -> str:
        return self.sp_data["baseSpecies"]

    def boost(self, stat: str) -> int:
        return max(-6, min(6, int(self.boosts.get(stat, 0))))

    def positive_boosts(self) -> int:
        return sum(v for k, v in self.boosts.items() if v > 0)

    @property
    def hp_pct(self) -> float:
        return 100.0 * self.hp / self.maxhp if self.maxhp else 0.0

    def set_hp_pct(self, pct: float):
        self.hp = max(0, min(self.maxhp, round(self.maxhp * pct / 100.0)))

    @property
    def fainted(self) -> bool:
        return self.hp <= 0

    def to_dict(self) -> dict:
        return {
            "species": self.species, "ability": self.ability, "item": self.item, "nature": self.nature,
            "sp": dict(self.sp), "moves": list(self.moves), "gender": self.gender, "hp": self.hp,
            "maxhp": self.maxhp, "status": self.status, "boosts": dict(self.boosts),
            "volatiles": sorted(self.volatiles), "mega": self.mega, "timesAttacked": self.times_attacked,
        }

    @classmethod
    def from_dict(cls, dex: Dex, d: dict) -> "Mon":
        m = cls(species=d["species"], ability=d.get("ability", ""), item=d.get("item", ""),
                nature=d.get("nature", "serious"), sp=dict(d.get("sp") or {}),
                moves=list(d.get("moves") or []), gender=d.get("gender", ""))
        m.setup(dex)
        if d.get("mega") and not m.mega:
            m.mega_evolve()
        if d.get("hp") is not None:
            m.hp = int(d["hp"])
        if d.get("hpPct") is not None:
            m.set_hp_pct(float(d["hpPct"]))
        m.status = d.get("status", "") or ""
        m.boosts = {k: int(v) for k, v in (d.get("boosts") or {}).items() if v}
        m.volatiles = set(d.get("volatiles") or [])
        m.times_attacked = int(d.get("timesAttacked", 0))
        return m


@dataclass
class SideState:
    reflect: bool = False
    lightscreen: bool = False
    auroraveil: bool = False
    tailwind: bool = False
    fainted: int = 0            # 쓰러진 아군 수 (Supreme Overlord · Last Respects)

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class FieldState:
    game_type: str = "singles"  # singles | doubles
    weather: str = ""           # sunnyday raindance sandstorm snowscape
    terrain: str = ""           # electricterrain grassyterrain psychicterrain mistyterrain
    gravity: bool = False
    trick_room: bool = False
    sides: list = dfield(default_factory=lambda: [SideState(), SideState()])

    @property
    def doubles(self) -> bool:
        return self.game_type == "doubles"

    def to_dict(self):
        return {"game_type": self.game_type, "weather": self.weather, "terrain": self.terrain,
                "gravity": self.gravity, "trick_room": self.trick_room,
                "sides": [s.to_dict() for s in self.sides]}

    @classmethod
    def from_dict(cls, d: dict) -> "FieldState":
        f = cls(game_type=d.get("game_type", "singles"), weather=d.get("weather", "") or "",
                terrain=d.get("terrain", "") or "", gravity=bool(d.get("gravity")),
                trick_room=bool(d.get("trick_room")))
        sides = d.get("sides") or [{}, {}]
        f.sides = [SideState(**{k: v for k, v in s.items() if k in SideState.__dataclass_fields__})
                   for s in sides]
        while len(f.sides) < 2:
            f.sides.append(SideState())
        return f
