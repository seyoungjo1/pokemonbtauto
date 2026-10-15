# -*- coding: utf-8 -*-
"""팀 텍스트(쇼다운 붙여넣기 형식) 읽기·쓰기 — 영어와 한글 모두.

    Garchomp @ Life Orb              한카리아스 @ 생명의구슬
    Ability: Rough Skin              특성: 까칠한피부
    EVs: 2 HP / 32 Atk / 32 Spe      능력포인트: 2 HP / 32 공격 / 32 스피드
    Jolly Nature                     성격: 명랑
    - Earthquake                     - 지진

챔피언스에서는 노력치 자리에 능력 포인트(SP, 0~32, 합계 66)를 적는다.
"""
import re

from .dex import STATS, Dex

STAT_ALIASES = {
    "hp": "hp", "h": "hp", "체력": "hp",
    "atk": "atk", "a": "atk", "공격": "atk", "공": "atk",
    "def": "def", "b": "def", "방어": "def", "방": "def",
    "spa": "spa", "spatk": "spa", "c": "spa", "특공": "spa", "특수공격": "spa",
    "spd": "spd", "spdef": "spd", "d": "spd", "특방": "spd", "특수방어": "spd",
    "spe": "spe", "s": "spe", "speed": "spe", "스피드": "spe", "스핏": "spe",
}
STAT_EN = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}
STAT_KO = {"hp": "HP", "atk": "공격", "def": "방어", "spa": "특공", "spd": "특방", "spe": "스피드"}
MAX_SP, TOTAL_SP = 32, 66


def parse_paste(dex: Dex, text: str) -> list:
    team, cur = [], None
    for raw in (text or "").replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            if cur:
                team.append(cur)
                cur = None
            continue
        if cur is None:
            cur = _new_from_header(dex, line)
            continue
        low = line.lower()
        if line.startswith(("-", "~", "•")):
            name = line.lstrip("-~• ").strip()
            name = re.sub(r"\s*\[.*\]$", "", name)
            mid = dex.find("moves", name)
            if mid and len(cur["moves"]) < 4:
                cur["moves"].append(mid)
            elif not mid:
                cur["unknown"].append(name)
            continue
        m = re.match(r"^(ability|특성)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            cur["ability"] = dex.find("abilities", m.group(2)) or ""
            continue
        m = re.match(r"^(evs|sps?|stat points|능력\s*포인트|노력치|포인트)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            cur["sp"] = _parse_spread(m.group(2))
            continue
        m = re.match(r"^(.+?)\s+nature$", line, re.I) or re.match(r"^성격\s*[:：]\s*(.+)$", line)
        if m:
            cur["nature"] = dex.find("natures", m.group(1).strip()) or cur["nature"]
            continue
        if low.startswith(("level", "레벨", "ivs", "shiny", "tera", "happiness", "gigantamax")):
            continue
    if cur:
        team.append(cur)
    for t in team:
        t["sp"] = clamp_sp(t["sp"])
    return team


def _new_from_header(dex, line):
    item = ""
    if "@" in line:
        line, item = [x.strip() for x in line.split("@", 1)]
    gender = ""
    g = re.search(r"\((M|F)\)\s*$", line)
    if g:
        gender = g.group(1)
        line = line[:g.start()].strip()
    sp_name = line
    m = re.match(r"^(.*)\(([^()]+)\)\s*$", line)
    if m and dex.find("species", m.group(2)):
        sp_name = m.group(2)
    sid = dex.find("species", sp_name) or ""
    return {"species": sid, "item": dex.find("items", item) or "" if item else "", "ability": "",
            "nature": "serious", "sp": {}, "moves": [], "gender": gender, "unknown": [] if sid else [sp_name]}


def _parse_spread(s: str) -> dict:
    out = {}
    for part in re.split(r"[/,]", s):
        m = re.match(r"^\s*(\d+)\s*([^\d\s].*?)\s*$", part) or re.match(r"^\s*([^\d\s].*?)\s*(\d+)\s*$", part)
        if not m:
            continue
        a, b = m.groups()
        num, name = (a, b) if a.isdigit() else (b, a)
        key = STAT_ALIASES.get(name.strip().lower().replace(" ", "").replace(".", ""))
        if key:
            out[key] = int(num)
    return out


def clamp_sp(sp: dict) -> dict:
    out, total = {}, 0
    for s in STATS:
        v = max(0, min(MAX_SP, int(sp.get(s, 0) or 0)))
        v = min(v, TOTAL_SP - total)
        if v:
            out[s] = v
        total += v
    return out


def export_paste(dex: Dex, team: list, lang: str = "en") -> str:
    blocks = []
    for t in team:
        sp = dex.species.get(t["species"], {})
        it = dex.items.get(t.get("item") or "", {})
        ab = dex.abilities.get(t.get("ability") or "", {})
        na = dex.natures.get(t.get("nature") or "serious", {})
        ko = lang == "ko"
        name = sp.get("ko" if ko else "name", t["species"])
        head = name + (f" ({t['gender']})" if t.get("gender") in ("M", "F") else "")
        if it:
            head += " @ " + it.get("ko" if ko else "name", "")
        lines = [head]
        if ab:
            lines.append(("특성: " if ko else "Ability: ") + ab.get("ko" if ko else "name", ""))
        spd = t.get("sp") or {}
        if spd:
            names = STAT_KO if ko else STAT_EN
            lines.append(("능력포인트: " if ko else "EVs: ") +
                         " / ".join(f"{spd[s]} {names[s]}" for s in STATS if spd.get(s)))
        if na:
            lines.append(f"성격: {na.get('ko', na['name'])}" if ko else f"{na['name']} Nature")
        for mv in t.get("moves") or []:
            mo = dex.moves.get(mv, {})
            lines.append("- " + mo.get("ko" if ko else "name", mv))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
