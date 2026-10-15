# -*- coding: utf-8 -*-
"""자기대전용 팀 만들기 — 챔피언스 랜덤배틀 세트(쇼다운 sets.json)를 바탕으로
역할에 맞는 SP·성격·도구를 고른다. 같은 포켓몬·같은 도구는 한 팀에 한 번(종족·도구 클로즈).

학습 데이터가 한쪽으로 쏠리지 않도록 SP 배분·성격·도구·기술을 일부러 섞는다.
"""
from __future__ import annotations

import random

from .dex import Dex, to_id

PHYS_NATURES = {"fast": ["jolly", "adamant"], "bulky": ["adamant", "impish", "careful"]}
SPEC_NATURES = {"fast": ["timid", "modest"], "bulky": ["modest", "bold", "calm"]}
GENERIC_ITEMS = ["lifeorb", "choicescarf", "focussash", "sitrusberry", "leftovers", "expertbelt",
                 "lumberry", "scopelens", "shellbell", "quickclaw", "brightpowder", "kingsrock",
                 "mentalherb", "whiteherb", "widelens", "lightclay"]
TYPE_ITEM = {
    "Normal": "silkscarf", "Fighting": "blackbelt", "Flying": "sharpbeak", "Poison": "poisonbarb",
    "Ground": "softsand", "Rock": "hardstone", "Bug": "silverpowder", "Ghost": "spelltag",
    "Steel": "metalcoat", "Fire": "charcoal", "Water": "mysticwater", "Grass": "miracleseed",
    "Electric": "magnet", "Psychic": "twistedspoon", "Ice": "nevermeltice", "Dragon": "dragonfang",
    "Dark": "blackglasses", "Fairy": "fairyfeather",
}
RESIST_BERRY = {
    "Fire": "occaberry", "Water": "passhoberry", "Electric": "wacanberry", "Grass": "rindoberry",
    "Ice": "yacheberry", "Fighting": "chopleberry", "Poison": "kebiaberry", "Ground": "shucaberry",
    "Flying": "cobaberry", "Psychic": "payapaberry", "Bug": "tangaberry", "Rock": "chartiberry",
    "Ghost": "kasibberry", "Dragon": "habanberry", "Dark": "colburberry", "Steel": "babiriberry",
    "Fairy": "roseliberry",
}


class TeamGen:
    def __init__(self, dex: Dex, seed=None):
        self.dex = dex
        self.rng = random.Random(seed)
        self.pool = [s for s in dex.sets if s in dex.species]

    # ------------------------------------------------------------
    def team(self, size: int = 6, explore: float = 0.1) -> list:
        """6마리 팀 (dict 목록). explore 비율만큼은 세트 밖 기술로 탐험."""
        rng = self.rng
        team, bases, items = [], set(), set()
        megas = 0
        tries = 0
        while len(team) < size and tries < 400:
            tries += 1
            sid = rng.choice(self.pool)
            sp = self.dex.species[sid]
            base = sp["baseSpecies"]
            if base in bases:
                continue
            if sp.get("isMega") and megas >= 2:
                continue
            m = self.make_set(sid, used_items=items, explore=rng.random() < explore)
            if not m:
                continue
            if m["item"]:
                items.add(m["item"])
            bases.add(base)
            megas += 1 if sp.get("isMega") else 0
            team.append(m)
        return team

    def make_set(self, sid: str, used_items=(), explore=False) -> dict | None:
        rng, dex = self.rng, self.dex
        sp = dex.species[sid]
        sets = dex.sets.get(sid, {}).get("sets") or []
        role = "Bulky Attacker"
        if sets:
            s = rng.choice(sets)
            role = s["role"]
            pool = [dex.find("moves", m) for m in s["movepool"]]
            pool = [m for m in pool if m]
            abilities = [dex.find("abilities", a) for a in s.get("abilities") or []]
        else:
            pool, abilities = [], []
        learn = dex.learnsets.get(sid) or dex.learnsets.get(to_id(sp["baseSpecies"])) or []
        if explore or len(pool) < 4:
            dmg = [m for m in learn if dex.moves[m]["category"] != "Status" and (dex.moves[m]["basePower"] or 0) >= 60]
            stab = [m for m in dmg if dex.moves[m]["type"] in sp["types"]]
            extra = rng.sample(stab, min(2, len(stab))) + rng.sample(dmg, min(2, len(dmg)))
            pool = list(dict.fromkeys(pool + extra))
        if not pool:
            return None
        moves = self._pick_moves(sp, pool)
        abilities = [a for a in abilities if a] or [to_id(a) for a in sp["abilities"].values()]
        ability = rng.choice(abilities)
        cats = [dex.moves[m]["category"] for m in moves]
        phys = cats.count("Physical") >= cats.count("Special")
        fast = role in ("Fast Attacker", "Setup Sweeper", "Fast Support", "Wallbreaker") or sp["baseStats"]["spe"] >= 95
        nature, spread = self._spread(sp, phys, fast, role)
        species_for_set = sid
        item = ""
        if sp.get("isMega") and sp.get("requiredItem"):
            item = dex.find("items", sp["requiredItem"]) or ""
            species_for_set = to_id(sp["baseSpecies"]) if to_id(sp["baseSpecies"]) in dex.species else sid
            if sp.get("changesFrom"):
                species_for_set = to_id(sp["changesFrom"])
        if not item:
            item = self._item(sp, moves, role, used_items)
        return {"species": species_for_set, "ability": ability if not sp.get("isMega") else
                to_id(dex.species[species_for_set]["abilities"].get("0", "")),
                "item": item, "nature": nature, "sp": spread, "moves": moves,
                "gender": rng.choice(["M", "F"]) if not sp.get("gender") else "", "role": role,
                "megaTarget": sid if sp.get("isMega") else ""}

    def _pick_moves(self, sp, pool):
        rng, dex = self.rng, self.dex
        pool = list(pool)
        rng.shuffle(pool)
        dmg = [m for m in pool if dex.moves[m]["category"] != "Status"]
        stab = [m for m in dmg if dex.moves[m]["type"] in sp["types"]]
        out = []
        if stab:
            out.append(stab[0])
        elif dmg:
            out.append(dmg[0])
        for m in pool:
            if len(out) >= 4:
                break
            if m not in out:
                out.append(m)
        return out[:4]

    def _spread(self, sp, phys, fast, role):
        rng = self.rng
        a = "atk" if phys else "spa"
        nat = (PHYS_NATURES if phys else SPEC_NATURES)["fast" if fast else "bulky"]
        templates = (
            [{a: 32, "spe": 32, "hp": 2}, {a: 32, "spe": 24, "hp": 10}, {a: 28, "spe": 32, "hp": 6}]
            if fast else
            [{"hp": 32, a: 32, "def": 2}, {"hp": 32, a: 16, "def": 9, "spd": 9},
             {"hp": 32, "def": 17, "spd": 17}, {"hp": 20, a: 32, "spd": 14}]
        )
        if role == "Bulky Support":
            templates = [{"hp": 32, "def": 17, "spd": 17}, {"hp": 32, "def": 32, "spd": 2},
                         {"hp": 32, "spd": 32, "def": 2}]
        return rng.choice(nat), dict(rng.choice(templates))

    def _item(self, sp, moves, role, used):
        rng, dex = self.rng, self.dex
        cands = []
        dmg_types = [dex.moves[m]["type"] for m in moves if dex.moves[m]["category"] != "Status"]
        stab = [t for t in dmg_types if t in sp["types"]]
        if stab:
            cands.append(TYPE_ITEM.get(stab[0]))
        weak = [t for t, v in self._weaknesses(sp["types"]).items() if v >= 2]
        if weak:
            cands.append(RESIST_BERRY.get(rng.choice(weak)))
        cands += GENERIC_ITEMS
        cands = [c for c in cands if c and c in dex.items and c not in used]
        return rng.choice(cands) if cands else ""

    def _weaknesses(self, types):
        out = {}
        for t in self.dex.typechart:
            if t in ("Stellar", "???"):
                continue
            mult = 1.0
            for d in types:
                if self.dex.immune(t, d):
                    mult = 0
                    break
                mult *= 2 ** self.dex.effectiveness(t, d)
            out[t] = mult
        return out


def to_showdown_set(dex: Dex, m: dict) -> dict:
    """쇼다운 Teams.pack 이 받는 형태 (레벨 50, SP 는 evs 자리에)."""
    sp = dex.species[m["species"]]
    it = dex.items.get(m.get("item") or "")
    ab = dex.abilities.get(m.get("ability") or "")
    return {
        "name": sp["name"], "species": sp["name"], "item": it["name"] if it else "",
        "ability": ab["name"] if ab else "", "moves": [dex.moves[x]["name"] for x in m["moves"] if x in dex.moves],
        "nature": dex.natures.get(m.get("nature") or "serious", {}).get("name", "Serious"),
        "evs": {k: int(m.get("sp", {}).get(k, 0)) for k in ("hp", "atk", "def", "spa", "spd", "spe")},
        "ivs": {k: 31 for k in ("hp", "atk", "def", "spa", "spd", "spe")},
        "gender": m.get("gender") or "", "level": 50,
    }
