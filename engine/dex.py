# -*- coding: utf-8 -*-
"""챔피언스 데이터(data/champions/*.json) 읽기와 검색."""
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "champions"

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
STAT_KO = {"hp": "HP", "atk": "공격", "def": "방어", "spa": "특공", "spd": "특방", "spe": "스피드"}


def to_id(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


class Dex:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = Path(data_dir)
        load = lambda k: json.loads((self.dir / f"{k}.json").read_text(encoding="utf-8"))
        self.species = load("species")
        self.learnsets = load("learnsets")
        self.moves = load("moves")
        self.items = load("items")
        self.abilities = load("abilities")
        self.typechart = load("typechart")
        self.natures = load("natures")
        self.sets = load("sets")
        self.formats = load("formats")
        self.meta = load("meta")
        # 메타 통계 (tools/fetch_meta.py) — 없으면 비어 있음
        opt = lambda k: json.loads((self.dir / f"{k}.json").read_text(encoding="utf-8")) \
            if (self.dir / f"{k}.json").is_file() else {}
        self.usage = opt("usage")
        self.expert = opt("expert_sets")
        self.types_ko = self.meta.get("typesKo", {})
        # 한글·영어 이름 → id 찾기표
        self._index = {}
        for kind in ("species", "moves", "items", "abilities"):
            tbl = {}
            for i, o in getattr(self, kind).items():
                tbl[to_id(o["name"])] = i
                tbl[_ko_key(o.get("ko", ""))] = i
                tbl[i] = i
            self._index[kind] = tbl
        for n, o in self.natures.items():
            self._index.setdefault("natures", {})[to_id(o["name"])] = n
            self._index["natures"][_ko_key(o.get("ko", ""))] = n

    # ------------------------------------------------------------ 조회
    def find(self, kind: str, name) -> str | None:
        """한글/영어 이름 또는 id 로 id 를 찾는다."""
        if not name:
            return None
        tbl = self._index.get(kind, {})
        return tbl.get(to_id(name)) or tbl.get(_ko_key(name))

    def get(self, kind: str, name) -> dict | None:
        i = self.find(kind, name)
        return getattr(self, kind).get(i) if i else None

    def search(self, kind: str, q: str, limit: int = 20, within=None) -> list[dict]:
        q = (q or "").strip().lower()
        out = []
        pool = getattr(self, kind)
        ids = within if within is not None else pool.keys()
        for i in ids:
            o = pool.get(i)
            if not o:
                continue
            if not q or q in o.get("ko", "").lower() or q in o["name"].lower() or q in i:
                out.append(o)
                if len(out) >= limit:
                    break
        return out

    # ------------------------------------------------------------ 타입 상성
    def effectiveness(self, move_type: str, def_type: str) -> int:
        """+1 효과굉장, -1 효과별로, 0 보통 (무효는 immune() 로 따로)."""
        code = self.typechart.get(def_type, {}).get("damageTaken", {}).get(move_type, 0)
        return {1: 1, 2: -1}.get(code, 0)

    def immune(self, move_type: str, def_type: str) -> bool:
        return self.typechart.get(def_type, {}).get("damageTaken", {}).get(move_type, 0) == 3

    def season_species(self) -> list:
        """이번 시즌(데이터의 규칙)에 엔트리로 고를 수 있는 모든 종 — 메가·대전 중 모습(캐스퐁 날씨 폼 등) 제외.
        메타 통계 유무와 상관없다 (통계 없는 종도 고를 수 있어야 한다)."""
        out = []
        for sid, sp in self.species.items():
            if sp.get("isMega") or sp.get("battleOnly"):
                continue
            if not (self.learnsets.get(sid) or self.learnsets.get(to_id(sp.get("changesFrom") or sp.get("baseSpecies") or ""))):
                continue
            out.append(sid)
        return out

    def base_of_mega(self, species_id: str) -> str:
        """메가 모습이면 메가진화 전 종, 아니면 그대로."""
        sp = self.species.get(species_id) or {}
        if sp.get("isMega"):
            b = to_id(sp.get("changesFrom") or sp.get("baseSpecies") or "")
            if b in self.species:
                return b
        return species_id

    def ident_name(self, species_id: str) -> str:
        """쇼다운이 대전에서 부르는 이름 ('p1a: 이름'). 별명이 없으면 원래 종 이름
        (리전폼·폼·메가 모두 앞부분만: Arcanine-Hisui → Arcanine, Charizard-Mega-Y → Charizard)."""
        sp = self.species.get(self.base_of_mega(species_id)) or {}
        return sp.get("baseSpecies") or sp.get("name") or species_id

    def mega_for(self, species_id: str, item_id: str) -> str | None:
        it = self.items.get(item_id or "")
        if not it or not it.get("megaStone"):
            return None
        sp = self.species.get(species_id)
        if not sp:
            return None
        target = it["megaStone"].get(sp["name"])
        return to_id(target) if target else None

    def usage_of(self, species_id: str, game_type: str = "singles") -> dict | None:
        """메타 사용률 (메가는 원래 종으로)."""
        tbl = ((self.usage.get(game_type) or {}).get("pokemon") or {})
        if species_id in tbl:
            return tbl[species_id]
        sp = self.species.get(species_id) or {}
        base = to_id(sp.get("changesFrom") or sp.get("baseSpecies") or "")
        return tbl.get(base)

    def can_learn(self, species_id: str, move_id: str) -> bool:
        return move_id in set(self.learnsets.get(species_id, []))


def _ko_key(s: str) -> str:
    return "ko:" + re.sub(r"\s+", "", str(s or ""))


@lru_cache(maxsize=1)
def default_dex() -> Dex:
    return Dex()
