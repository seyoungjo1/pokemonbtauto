#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""쇼다운에서 뽑은 원본 JSON(export_champions.js)에 한글 이름과 그림 번호를 붙여
data/champions/ 에 저장한다.

  python tools/build_data.py <원본 폴더> [--out data/champions] [--csv-cache build/pokeapi]

한글 이름은 PokeAPI 저장소의 CSV(언어 id 3 = 한국어)에서 가져온다.
챔피언스에서 새로 생긴 기술·특성처럼 PokeAPI 에 아직 없는 것은 영어 이름을 그대로 쓴다.
"""
import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POKEAPI_CSV = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/"
CSV_FILES = ["pokemon_species_names", "move_names", "ability_names", "item_names",
             "type_names", "nature_names", "pokemon", "stat_names"]
KO, EN = "3", "9"

FORME_KO = {
    "Mega": "메가 {n}", "Mega-X": "메가 {n} X", "Mega-Y": "메가 {n} Y", "Mega-Z": "메가 {n} Z",
    "Alola": "{n} (알로라)", "Galar": "{n} (가라르)", "Hisui": "{n} (히스이)", "Paldea": "{n} (팔데아)",
    "Paldea-Combat": "{n} (팔데아 컴뱃)", "Paldea-Blaze": "{n} (팔데아 블레이즈)",
    "Paldea-Aqua": "{n} (팔데아 워터)", "Therian": "{n} (영물폼)", "Origin": "{n} (오리진폼)",
    "Wash": "워시{n}", "Heat": "히트{n}", "Frost": "프로스트{n}", "Fan": "스핀{n}", "Mow": "커트{n}",
    "F": "{n} ♀", "M": "{n} ♂", "Bloodmoon": "{n} (붉은 달)", "Midnight": "{n} (한밤중의 모습)",
    "Dusk": "{n} (황혼의 모습)", "Eternal": "{n} (영원의 꽃)",
}


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_csv(name: str, cache: Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{name}.csv"
    if not p.is_file():
        with urllib.request.urlopen(POKEAPI_CSV + name + ".csv", timeout=60) as r:
            p.write_bytes(r.read())
    return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))))


def name_table(rows: list[dict], id_col: str) -> dict:
    """영어 이름 id → 한국어 이름."""
    en, ko = {}, {}
    for r in rows:
        if r["local_language_id"] == EN:
            en[r[id_col]] = r["name"]
        elif r["local_language_id"] == KO:
            ko[r[id_col]] = r["name"]
    return {to_id(en[k]): ko[k] for k in en if k in ko}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default=str(ROOT / "data" / "champions"))
    ap.add_argument("--csv-cache", default=str(ROOT / "build" / "pokeapi"))
    a = ap.parse_args(argv)
    raw, out, cache = Path(a.raw), Path(a.out), Path(a.csv_cache)
    out.mkdir(parents=True, exist_ok=True)

    J = {k: json.loads((raw / f"{k}.json").read_text(encoding="utf-8"))
         for k in ["species", "learnsets", "moves", "items", "abilities", "typechart",
                   "natures", "sets", "formats", "meta"]}
    C = {n: load_csv(n, cache) for n in CSV_FILES}

    sp_ko = name_table(C["pokemon_species_names"], "pokemon_species_id")
    mv_ko = name_table(C["move_names"], "move_id")
    ab_ko = name_table(C["ability_names"], "ability_id")
    it_ko = name_table(C["item_names"], "item_id")
    ty_ko = name_table(C["type_names"], "type_id")
    na_ko = name_table(C["nature_names"], "nature_id")
    pokeapi_id = {to_id(r["identifier"]): int(r["id"]) for r in C["pokemon"]}
    pokeapi_num = {}
    for r in C["pokemon"]:
        if r["is_default"] == "1":
            pokeapi_num[int(r["species_id"])] = int(r["id"])

    missing = {"species": [], "moves": [], "abilities": [], "items": []}

    for sid, s in J["species"].items():
        base_ko = sp_ko.get(to_id(s["baseSpecies"]))
        if base_ko is None:
            missing["species"].append(s["name"])
            s["ko"] = s["name"]
        elif s["forme"]:
            tmpl = FORME_KO.get(s["forme"], "{n} (" + s["forme"] + ")")
            s["ko"] = tmpl.format(n=base_ko)
        else:
            s["ko"] = base_ko
        # 그림 번호: 폼까지 일치하는 PokeAPI 번호 → 없으면 기본 폼 번호
        # (쇼다운 charizardmegax ↔ PokeAPI charizard-mega-x 는 to_id 로 같아진다)
        pid = pokeapi_id.get(sid)
        s["pokeapiId"] = pid or pokeapi_num.get(s["num"])
        s["spriteExact"] = pid is not None

    for table, ko, key in [(J["moves"], mv_ko, "moves"), (J["abilities"], ab_ko, "abilities"),
                           (J["items"], it_ko, "items")]:
        for i, o in table.items():
            o["ko"] = ko.get(i)
            if not o["ko"]:
                o["ko"] = o["name"]
                missing[key].append(o["name"])
            o.pop("desc", None)  # 긴 설명은 빼서 크기를 줄인다 (shortDesc 는 남김)

    types_ko = {t: ty_ko.get(to_id(t), t) for t in J["typechart"]}
    for n, o in J["natures"].items():
        o["ko"] = na_ko.get(n, o["name"])

    meta = J["meta"]
    meta["builtAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["koMissing"] = {k: sorted(v) for k, v in missing.items()}
    meta["typesKo"] = types_ko

    for k, v in J.items():
        if k == "meta":
            continue
        (out / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, separators=(",", ":")),
                                      encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("한글 이름 없음:", {k: len(v) for k, v in missing.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
