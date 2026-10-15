# -*- coding: utf-8 -*-
"""포켓몬 = 이름이 아니라 '속성의 묶음' 토큰.

한 마리(세트) → 고정 길이 벡터:
  타입(18) · 실능력치(6, SP·성격 반영) · 메가 후 실능력치(6)와 메가 여부 ·
  받는 상성(18, 특성 무효 포함) · 주는 상성(18: 기술 타입·위력·자속·공격/특공으로 각 타입을 얼마나 세게 치는가) ·
  특성 효과(쇼다운 효과 처리기 목록) · 도구 효과(처리기 목록 + 메가스톤·열매) ·
  기술 묶음(분류·우선도·대상·부가효과·플래그·처리기, 4개를 합침 — 순서 무관)

특성·도구·기술을 이름이 아니라 '어떤 효과 처리기를 가졌는가'로 나타내므로, 새 특성·새 기술이 나와도
같은 방식으로 표현된다 (패치·새 포켓몬에 일반화). 팀 = 토큰 6개의 집합.
"""
from __future__ import annotations

import math

import numpy as np

from .dex import Dex, to_id
from .mon import Mon

TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying",
         "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
ABILITY_IMMUNE = {"levitate": "Ground", "waterabsorb": "Water", "stormdrain": "Water", "dryskin": "Water",
                  "voltabsorb": "Electric", "lightningrod": "Electric", "motordrive": "Electric", "flashfire": "Fire",
                  "wellbakedbody": "Fire", "sapsipper": "Grass", "eartheater": "Ground"}
MOVE_FLAGS = ["contact", "sound", "punch", "slicing", "bite", "pulse", "wind", "heal", "charge", "recharge",
              "bullet", "dance", "powder"]
TARGETS = {"normal": 0, "any": 0, "adjacentFoe": 0, "randomNormal": 0, "allAdjacentFoes": 1, "allAdjacent": 1,
           "self": 2, "adjacentAlly": 3, "adjacentAllyOrSelf": 3, "allySide": 4, "allies": 4, "foeSide": 5,
           "all": 6, "scripted": 0}
STATUSES = ["brn", "par", "psn", "tox", "slp", "frz"]


class Vocab:
    """데이터에서 효과 처리기 이름 목록을 만든다 (자주 쓰이는 순)."""

    def __init__(self, dex: Dex, n_ability=56, n_item=29, n_move=25):
        def top(tbl, n):
            cnt = {}
            for v in tbl.values():
                for c in v.get("callbacks") or []:
                    cnt[c] = cnt.get(c, 0) + 1
            return [k for k, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:n]]
        self.ab = top(dex.abilities, n_ability)
        self.it = top(dex.items, n_item)
        self.mv = top(dex.moves, n_move)

    def size(self):
        return (18 + 6 + 7 + 18 + 18 + len(self.ab) + len(self.it) + 2
                + 3 + 3 + 7 + len(STATUSES) + 4 + len(MOVE_FLAGS) + len(self.mv) + 2 + 8)


def _eff(dex: Dex, atk_type, def_types):
    m = 1.0
    for t in def_types:
        if dex.immune(atk_type, t):
            return 0.0
        m *= {1: 2.0, -1: 0.5}.get(dex.effectiveness(atk_type, t), 1.0)
    return m


def token(dex: Dex, s: dict, vocab: Vocab) -> np.ndarray:
    m = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""),
            nature=s.get("nature", "serious"), sp=dict(s.get("sp") or {}), moves=list(s.get("moves") or [])).setup(dex)
    base_types = list(m.types)
    stats = dict(m.stats)
    mm = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""),
             nature=s.get("nature", "serious"), sp=dict(s.get("sp") or {}), moves=list(s.get("moves") or [])).setup(dex)
    is_mega = mm.mega_evolve()
    ab = mm.ability if is_mega else m.ability
    types = list(mm.types) if is_mega else base_types
    v = []
    v += [1.0 if t in types else 0.0 for t in TYPES]
    v += [stats[k] / 200.0 for k in STATS]
    v += [float(is_mega)] + ([mm.stats[k] / 200.0 for k in STATS] if is_mega else [0.0] * 6)
    # 받는 상성 (log2, 특성 무효 반영)
    imm = ABILITY_IMMUNE.get(ab, "")
    v += [(-3.0 if (t == imm) else (math.log2(e) if (e := _eff(dex, t, types)) > 0 else -3.0)) for t in TYPES]
    # 주는 상성: 각 방어 타입을 가장 세게 치는 기술의 (위력 × 자속 × 상성 × 공격 실능) 크기
    off = [0.0] * 18
    mv_rows = []
    for mid in m.moves:
        mo = dex.moves.get(mid)
        if not mo:
            continue
        mv_rows.append(mo)
        if mo["category"] == "Status" or not mo.get("basePower"):
            continue
        stab = 1.5 if mo["type"] in types else 1.0
        atk = (mm if is_mega else m).stats["atk" if mo["category"] == "Physical" else "spa"]
        for k, t in enumerate(TYPES):
            val = mo["basePower"] * stab * _eff(dex, mo["type"], [t]) * atk / (100 * 150)
            off[k] = max(off[k], val)
    v += [min(4.0, x) for x in off]
    # 특성 · 도구 효과
    acb = set((dex.abilities.get(ab) or {}).get("callbacks") or [])
    v += [1.0 if c in acb else 0.0 for c in vocab.ab]
    it = dex.items.get(m.item) or {}
    icb = set(it.get("callbacks") or [])
    v += [1.0 if c in icb else 0.0 for c in vocab.it]
    v += [1.0 if it.get("megaStone") else 0.0, 1.0 if it.get("isBerry") else 0.0]
    # 기술 묶음 (합)
    cat = [0.0] * 3
    pri = [0.0] * 3
    tgt = [0.0] * 7
    st = [0.0] * len(STATUSES)
    sec = [0.0] * 4          # 부가효과 확률 합 · 상대 랭크다운 · 자기 랭크업 · 풀죽음
    fl = [0.0] * len(MOVE_FLAGS)
    cbs = [0.0] * len(vocab.mv)
    power = 0.0
    for mo in mv_rows:
        cat[["Physical", "Special", "Status"].index(mo["category"])] += 1
        p = mo.get("priority", 0)
        pri[0 if p < 0 else (2 if p > 0 else 1)] += 1
        tgt[TARGETS.get(mo.get("target", "normal"), 0)] += 1
        for sd in ([mo["secondary"]] if mo.get("secondary") else []) + list(mo.get("secondaries") or []):
            ch = (sd.get("chance") or 100) / 100.0
            sec[0] += ch
            if sd.get("status") in STATUSES:
                st[STATUSES.index(sd["status"])] += ch
            if sd.get("boosts"):
                sec[1] += ch
            if (sd.get("self") or {}).get("boosts"):
                sec[2] += ch
            if sd.get("volatileStatus") == "flinch":
                sec[3] += ch
        if mo.get("category") == "Status" and mo.get("status") in STATUSES:
            st[STATUSES.index(mo["status"])] += 1
        for k, f in enumerate(MOVE_FLAGS):
            if (mo.get("flags") or {}).get(f):
                fl[k] += 1
        mcb = set(mo.get("callbacks") or [])
        for k, c in enumerate(vocab.mv):
            if c in mcb:
                cbs[k] += 1
        power = max(power, mo.get("basePower") or 0)
    v += cat + pri + tgt + st + sec + fl + cbs + [power / 150.0, len(mv_rows) / 4.0]
    # 날씨: 까는가(4) · 그 날씨에서 얻는 이득(4, 자속·위력·실능 반영)
    wp = weather_profile(dex, s)
    v += [1.0 if w in wp["sets"] else 0.0 for w in ("rain", "sun", "sand", "snow")]
    v += [wp["gain"][w] for w in ("rain", "sun", "sand", "snow")]
    return np.array(v, np.float32)


def team_tokens(dex: Dex, team: list, vocab: Vocab | None = None) -> np.ndarray:
    from .teamgen import normalize_set
    vocab = vocab or Vocab(dex)
    return np.stack([token(dex, normalize_set(dex, s), vocab) for s in team])


def selfcheck(dex: Dex | None = None) -> dict:
    """셀프테스트: 토큰 길이 일정 · 같은 포켓몬도 세트가 다르면 토큰이 다름 · 이름 없이 계산됨."""
    dex = dex or Dex()
    vb = Vocab(dex)
    a = token(dex, {"species": "garchomp", "ability": "roughskin", "item": "choicescarf", "nature": "jolly",
                    "sp": {"atk": 32, "spe": 32}, "moves": ["earthquake", "dragonclaw", "rockslide", "poisonjab"]}, vb)
    b = token(dex, {"species": "garchomp", "ability": "roughskin", "item": "lifeorb", "nature": "adamant",
                    "sp": {"atk": 32, "hp": 32}, "moves": ["earthquake", "swordsdance", "scaleshot", "protect"]}, vb)
    c = token(dex, {"species": "rotomwash", "ability": "levitate", "item": "leftovers", "nature": "bold",
                    "sp": {"hp": 32, "def": 32}, "moves": ["hydropump", "voltswitch", "willowisp", "protect"]}, vb)
    ok = len(a) == len(b) == len(c) == vb.size() and not np.allclose(a, b) and np.isfinite(a).all()
    # 부유: 땅 기술을 받는 칸이 무효(-3)
    ok = ok and c[18 + 6 + 7 + TYPES.index("Ground")] == -3.0
    return {"ok": bool(ok), "size": int(vb.size()), "summary": f"토큰 {vb.size()}칸 · 같은 종 다른 세트 구분 · 특성 무효 반영"}


# ======================================================================== 날씨·필드 시너지 (이득의 양)
WEATHER_SETTERS = {"drought": "sun", "drizzle": "rain", "sandstream": "sand", "snowwarning": "snow",
                   "orichalcumpulse": "sun", "primordialsea": "rain", "desolateland": "sun",
                   "sunnyday": "sun", "raindance": "rain", "sandstorm": "sand", "snowscape": "snow"}
SPEED_ABILITY = {"swiftswim": "rain", "chlorophyll": "sun", "sandrush": "sand", "slushrush": "snow"}
POWER_ABILITY = {"solarpower": ("sun", "Special", 1.5), "sandforce": ("sand", None, 1.3)}
CHARGE_SKIP = {"electroshot": "rain", "solarbeam": "sun", "solarblade": "sun"}
ACCURACY = {"thunder": "rain", "hurricane": "rain", "blizzard": "snow"}
TYPE_BOOST = {"rain": ("Water", "Fire"), "sun": ("Fire", "Water")}   # (1.5배, 0.5배)


def weather_profile(dex: Dex, s: dict) -> dict:
    """세트 하나 → 어떤 날씨를 까는가, 날씨마다 얻는 이득(화력·스피드·내구, 손해는 음수).

    화력 이득 = Σ 기술 (날씨 배율 − 1) × 위력 × 자속 × 공격 실능 / 기준.  예: 잔비 + 자속 일렉트로빔은
    '충전 생략'(실질 2배)이 자속 1.5배·특공에 곱해져 크게 잡힌다."""
    m = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""),
            nature=s.get("nature", "serious"), sp=dict(s.get("sp") or {}), moves=list(s.get("moves") or [])).setup(dex)
    if dex.mega_for(m.species, m.item):
        m.mega_evolve()
    ab, types = m.ability, list(m.types)
    sets = {WEATHER_SETTERS[x] for x in [ab] + list(m.moves) if x in WEATHER_SETTERS}
    gain = {w: 0.0 for w in ("rain", "sun", "sand", "snow")}
    for mid in m.moves:
        mo = dex.moves.get(mid)
        if not mo or mo["category"] == "Status":
            continue
        bp = mo.get("basePower") or 0
        stab = 1.5 if mo["type"] in types else 1.0
        atk = m.stats["atk" if mo["category"] == "Physical" else "spa"]
        unit = bp * stab * atk / (100 * 150)
        for w, (up, down) in TYPE_BOOST.items():
            if mo["type"] == up:
                gain[w] += 0.5 * unit
            elif mo["type"] == down:
                gain[w] -= 0.5 * unit
        if mid == "weatherball":
            for w in gain:
                gain[w] += 1.0 * unit                      # 날씨가 있으면 위력 2배 (타입도 바뀜)
        if mid in CHARGE_SKIP:
            gain[CHARGE_SKIP[mid]] += 1.0 * unit           # 2턴 → 1턴
        if mid in ACCURACY:
            acc = (mo.get("accuracy") or 100) / 100.0 if mo.get("accuracy") is not True else 1.0
            gain[ACCURACY[mid]] += (1.0 - acc) * unit
        if ab in POWER_ABILITY:
            w, cat, mul = POWER_ABILITY[ab]
            if cat is None or cat == mo["category"]:
                gain[w] += (mul - 1) * unit
    if ab in SPEED_ABILITY:
        gain[SPEED_ABILITY[ab]] += m.stats["spe"] / 100.0 * max(m.stats["atk"], m.stats["spa"]) / 150.0
    if "Rock" in types:
        gain["sand"] += 0.5 * m.stats["spd"] / 150.0
    if "Ice" in types:
        gain["snow"] += 0.5 * m.stats["def"] / 150.0
    return {"sets": sets, "gain": gain}
