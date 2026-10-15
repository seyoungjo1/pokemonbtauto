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
        return (18 + 6 + 7 + 18 + 18 + 3 + len(self.ab) + len(self.it) + 2
                + 3 + 3 + 7 + len(STATUSES) + 4 + len(MOVE_FLAGS) + len(self.mv) + 2 + 8 + 2 + len(TEMPO_NAMES))


_REF = {}


def move_multipliers(dex: Dex, atk: Mon, mid: str):
    """기술 하나에 특성이 주는 배율, 도구가 주는 배율 — 데미지 엔진으로 (특성 있음 ÷ 없음)."""
    from .damage import calc
    import copy
    if "ref" not in _REF:
        sid = next((x for x in ("mew", "mewtwo", "alakazam", "espeon") if x in dex.species),
                   next(k for k, v in dex.species.items() if v["types"] == ["Psychic"]))
        _REF["ref"] = Mon(species=sid, ability="", item="", nature="serious", sp={"hp": 32, "def": 32, "spd": 32},
                          moves=[]).setup(dex)
        _REF["ref"].ability = "noability"

    def dmg(a):
        try:
            r = calc(dex, a, _REF["ref"], mid)
            return sum(sum(h) / len(h) for h in r.rolls if h)   # 타격마다 16단계 평균의 합
        except Exception:
            return 0.0
    full = copy.copy(atk)
    no_ab = copy.copy(atk)
    no_ab.ability = "noability"
    none = copy.copy(no_ab)
    none.item = ""
    d_full, d_noab, d_none = dmg(full), dmg(no_ab), dmg(none)
    ra = d_full / d_noab if d_noab > 0 else 1.0
    ri = d_noab / d_none if d_none > 0 else 1.0
    return min(3.0, ra), min(3.0, ri)


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
    # 주는 상성: 각 방어 타입을 가장 세게 치는 기술의 (위력 × 자속 × 특성·도구 배율 × 상성 × 공격 실능) 크기
    # 특성·도구 배율은 데미지 엔진으로 잰다 (테크니션 + 불릿펀치, 철주먹 + 펀치, 적응력 자속 2배 …)
    off = [0.0] * 18
    mv_rows = []
    ab_mul, it_mul = [], []
    atk_mon = mm if is_mega else m
    for mid in m.moves:
        mo = dex.moves.get(mid)
        if not mo:
            continue
        mv_rows.append(mo)
        if mo["category"] == "Status" or not mo.get("basePower"):
            continue
        stab = 1.5 if mo["type"] in types else 1.0
        ra, ri = move_multipliers(dex, atk_mon, mid)
        ab_mul.append(ra)
        it_mul.append(ri)
        atk = atk_mon.stats["atk" if mo["category"] == "Physical" else "spa"]
        for k, t in enumerate(TYPES):
            val = mo["basePower"] * stab * ra * ri * _eff(dex, mo["type"], [t]) * atk / (100 * 150)
            off[k] = max(off[k], val)
    v += [min(4.0, x) for x in off]
    v += [max(ab_mul, default=1.0) - 1.0, (sum(ab_mul) / len(ab_mul) - 1.0) if ab_mul else 0.0,
          max(it_mul, default=1.0) - 1.0]
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
    # 랭크업 · 장판 · 선공기 · 교체 · 회복
    v += tempo_profile(dex, s, atk_mon)
    # 특성 ↔ 도구·기술 맞물림 (포이즌힐 + 맹독구슬, 근성 + 화염구슬, 곡예 등)
    al = ability_links(dex, s)
    v += [al["link"], 1.0 if al["selfStatus"] else 0.0]
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


# ======================================================================== 특성 ↔ 도구 · 기술 (데미지 배율 밖의 관계)
# 스스로 상태이상을 거는 도구 → 그 상태이상으로 이득을 보는 특성·기술
SELF_STATUS_ITEM = {"toxicorb": "tox", "flameorb": "brn"}
STATUS_BENEFIT = {   # 특성: {상태: 이득(대략, 턴당 HP 비율 또는 화력 배율 − 1)}
    "poisonheal": {"tox": 0.125 + 0.25, "psn": 0.125 + 0.125},   # 회복 1/8 + 독 데미지 무효
    "guts": {"brn": 0.5 + 0.5, "tox": 0.5, "psn": 0.5, "par": 0.5},   # 공격 1.5배 (화상 반감도 무시)
    "toxicboost": {"tox": 0.5, "psn": 0.5}, "flareboost": {"brn": 0.5},
    "marvelscale": {"brn": 0.5, "tox": 0.5, "psn": 0.5, "par": 0.5}, "quickfeet": {"brn": 0.5, "tox": 0.5, "psn": 0.5, "par": 0.5},
}
STATUS_MOVES = {"facade": 1.0}      # 상태이상이면 위력 2배
CONSUMABLE_ABILITY = {"unburden": 1.0, "harvest": 0.5, "cheekpouch": 0.3, "ripen": 0.5, "gluttony": 0.3}
RECOIL_ITEM_ABILITY = {("magicguard", "lifeorb"): 0.1, ("sheerforce", "lifeorb"): 0.1}


def ability_links(dex: Dex, s: dict) -> dict:
    """특성과 도구·기술의 맞물림 (데미지 엔진이 안 재는 것). 값이 클수록 세트가 서로 맞물림."""
    ab = to_id(s.get("ability") or "")
    item = to_id(s.get("item") or "")
    moves = [to_id(m) for m in s.get("moves") or []]
    it = dex.items.get(item) or {}
    status = SELF_STATUS_ITEM.get(item)
    link, why = 0.0, []
    if status:
        b = STATUS_BENEFIT.get(ab, {}).get(status, 0.0)
        mv = sum(STATUS_MOVES.get(m, 0.0) for m in moves)
        if b or mv:
            link += b + 0.5 * mv
            why.append(f"{ab}+{item}" + ("+facade" if mv else ""))
        else:
            link -= 0.3          # 이득 없이 스스로 상태이상 = 손해
            why.append(f"{item}(이득 없음)")
    if ab in CONSUMABLE_ABILITY and (it.get("isBerry") or item in {"focussash", "whiteherb", "mentalherb",
                                                                    "electricseed", "psychicseed", "grassyseed", "mistyseed"}):
        link += CONSUMABLE_ABILITY[ab]
        why.append(f"{ab}+{item}")
    if (ab, item) in RECOIL_ITEM_ABILITY:
        link += RECOIL_ITEM_ABILITY[(ab, item)]
        why.append(f"{ab}+{item}")
    if ab == "klutz" and item and item not in {"", "toxicorb", "flameorb"}:
        link -= 0.2
    return {"link": link, "why": why, "selfStatus": status or ""}


# ======================================================================== 랭크업 · 장판 · 선공기 (스피드와 무관한 화력 구분)
TEMPO_NAMES = ["boost_atk", "boost_spa", "boost_spe", "boost_def", "boost_spd", "setup_power",
               "hz_rocks", "hz_spikes", "hz_tspikes", "hz_web", "hz_remove",
               "prio_power", "prio_level", "speed_power", "pivot", "drain_heal", "protect", "prankster_status"]
HAZARDS = {"stealthrock": 6, "spikes": 7, "toxicspikes": 8, "stickyweb": 9}
HAZARD_REMOVAL = {"rapidspin", "defog", "tidyup", "courtchange", "mortalspin"}


def tempo_profile(dex: Dex, s: dict, atk_mon: Mon) -> list:
    """랭크업 기술(칼춤·용춤·배북·껍질깨기…)의 양과 그걸로 늘어나는 화력, 장판(스텔스록·끈적끈적네트…),
    선공기 화력(불릿펀치·신속 — 스피드 조절·끈적끈적네트와 무관), 일반 기술 화력(스피드에 좌우),
    교체 기술·흡수/회복·방어, 짓궂은마음 변화기."""
    f = [0.0] * len(TEMPO_NAMES)
    ab = to_id(s.get("ability") or "")
    best = {"Physical": 0.0, "Special": 0.0}
    prio_best, prio_lv, speed_best = 0.0, 0, 0.0
    for mid in atk_mon.moves:
        mo = dex.moves.get(mid)
        if not mo:
            continue
        if mo["category"] != "Status" and mo.get("basePower"):
            stab = 1.5 if mo["type"] in atk_mon.types else 1.0
            ra, ri = move_multipliers(dex, atk_mon, mid)
            st = atk_mon.stats["atk" if mo["category"] == "Physical" else "spa"]
            val = mo["basePower"] * stab * ra * ri * st / (100 * 150)
            best[mo["category"]] = max(best[mo["category"]], val)
            if mo.get("priority", 0) > 0:
                if val > prio_best:
                    prio_best, prio_lv = val, mo["priority"]
            else:
                speed_best = max(speed_best, val)
        b = mo.get("boosts") if mo.get("target") == "self" else None
        b = b or ((mo.get("self") or {}).get("boosts") if mo["category"] == "Status" else None)
        if mid == "bellydrum":
            b = {"atk": 6}
        if b:
            for k, idx in (("atk", 0), ("spa", 1), ("spe", 2), ("def", 3), ("spd", 4)):
                f[idx] += b.get(k, 0)
        if mo.get("sideCondition") in HAZARDS and mo.get("target") == "foeSide":
            f[HAZARDS[mo["sideCondition"]]] = 1.0
        if mid in HAZARD_REMOVAL:
            f[10] = 1.0
        if mo.get("selfSwitch"):
            f[14] += 1.0
        if mo.get("drain") or (mo.get("flags") or {}).get("heal"):
            f[15] += 1.0
        if mo.get("stallingMove"):
            f[16] = 1.0
        if ab == "prankster" and mo["category"] == "Status":
            f[17] += 1.0
    # 랭크업으로 늘어나는 화력: 공격 랭크 n 단계 = (2+n)/2 배
    stage = lambda n: (2 + min(6, max(0, n))) / 2.0 - 1.0
    f[5] = stage(f[0]) * best["Physical"] + stage(f[1]) * best["Special"]
    f[11], f[12], f[13] = prio_best, float(prio_lv), speed_best
    return f
