# -*- coding: utf-8 -*-
"""벤치마크 — 아키타입별 고정 팀과 통계적으로 의미 있는 평가 규약 (싱글 6→3).

"10개 메타 팀 상대 승률"로는 전체를 대변할 수 없고, 몇십 판으로는 차이를 말할 수 없다. 그래서
  1. 아키타입별 고정 팀: 비·쾌청·모래·설경(오로라베일)·트릭룸·막이(맹독·회복)·바톤터치·끈적끈적네트·하이퍼오펜스·스탠다드
     — 대전 AI 가 '이런 식으로 두는 상대'에 다 대응하는지 보기 위한 최소 집합. 팀은 데이터로 검증한다(배우는 기술·특성·도구 클로즈·SP).
  2. 거울전(mirror): 같은 팀을 양쪽이 쓰면 팀 유불리가 사라져 '두는 실력'만 남는다. 교차전(cross)은 팀 유불리까지 포함.
  3. 양쪽 자리를 반씩 바꿔 두고, 승률은 윌슨 95% 신뢰구간과 함께 보고한다.
     n판에서 두 AI 의 차이를 말하려면 (대략) 50% 근처에서 ±1.96·0.5/√n: 100판 ±10%p, 400판 ±5%p, 1600판 ±2.5%p.
"""
from __future__ import annotations

import math
import time

from .dex import Dex, to_id
from .teamgen import normalize_set

# ---- 팀 정의: (종, 특성, 도구, 성격, SP, 기술 4)
SINGLES = {
    "rain": ("비", [
        ("pelipper", "drizzle", "damprock", "modest", {"hp": 32, "spa": 32, "def": 2}, ["hurricane", "weatherball", "uturn", "roost"]),
        ("archaludon", "stamina", "leftovers", "modest", {"hp": 32, "spa": 32, "spd": 2}, ["electroshot", "dracometeor", "flashcannon", "bodypress"]),
        ("basculegion", "swiftswim", "lifeorb", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["wavecrash", "lastrespects", "aquajet", "flipturn"]),
        ("swampert", "torrent", "swampertite", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["liquidation", "earthquake", "icepunch", "stealthrock"]),
        ("politoed", "drizzle", "wiseglasses", "modest", {"hp": 32, "spa": 32, "spe": 2}, ["hydropump", "icebeam", "weatherball", "focusblast"]),
        ("corviknight", "pressure", "sitrusberry", "impish", {"hp": 32, "def": 32, "spd": 2}, ["bravebird", "bodypress", "roost", "uturn"]),
    ]),
    "sun": ("쾌청", [
        ("torkoal", "drought", "heatrock", "bold", {"hp": 32, "def": 32, "spd": 2}, ["lavaplume", "stealthrock", "yawn", "rapidspin"]),
        ("venusaur", "chlorophyll", "lifeorb", "modest", {"spa": 32, "spe": 32, "hp": 2}, ["gigadrain", "weatherball", "sludgebomb", "growth"]),
        ("charizard", "blaze", "charizarditey", "timid", {"spa": 32, "spe": 32, "hp": 2}, ["fireblast", "solarbeam", "focusblast", "roost"]),
        ("scovillain", "chlorophyll", "wiseglasses", "modest", {"spa": 32, "spe": 32, "hp": 2}, ["solarbeam", "overheat", "flamethrower", "leafstorm"]),
        ("garchomp", "roughskin", "choicescarf", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["earthquake", "outrage", "rockslide", "poisonjab"]),
        ("kingambit", "supremeoverlord", "leftovers", "adamant", {"hp": 32, "atk": 32, "spd": 2}, ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"]),
    ]),
    "sand": ("모래", [
        ("hippowdon", "sandstream", "leftovers", "impish", {"hp": 32, "def": 32, "spd": 2}, ["earthquake", "stealthrock", "slackoff", "whirlwind"]),
        ("excadrill", "sandrush", "lifeorb", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["earthquake", "ironhead", "rockslide", "swordsdance"]),
        ("tyranitar", "sandstream", "expertbelt", "adamant", {"hp": 32, "atk": 32, "spd": 2}, ["stoneedge", "crunch", "earthquake", "icepunch"]),
        ("garchomp", "roughskin", "yacheberry", "jolly", {"hp": 32, "spe": 32, "atk": 2}, ["earthquake", "dragontail", "stealthrock", "swordsdance"]),
        ("lycanroc", "sandrush", "focussash", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["accelerock", "stoneedge", "closecombat", "swordsdance"]),
        ("corviknight", "pressure", "sitrusberry", "impish", {"hp": 32, "def": 32, "spd": 2}, ["bravebird", "bodypress", "roost", "defog"]),
    ]),
    "snowveil": ("설경·오로라베일", [
        ("ninetalesalola", "snowwarning", "lightclay", "timid", {"hp": 32, "spe": 32, "spa": 2}, ["auroraveil", "blizzard", "moonblast", "encore"]),
        ("weavile", "pressure", "muscleband", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["tripleaxel", "knockoff", "iceshard", "lowkick"]),
        ("dragonite", "multiscale", "lumberry", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["dragondance", "extremespeed", "earthquake", "outrage"]),
        ("kingambit", "supremeoverlord", "leftovers", "adamant", {"hp": 32, "atk": 32, "spd": 2}, ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"]),
        ("gholdengo", "goodasgold", "chopleberry", "timid", {"spa": 32, "spe": 32, "hp": 2}, ["nastyplot", "makeitrain", "shadowball", "recover"]),
        ("garchomp", "roughskin", "lifeorb", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["swordsdance", "earthquake", "scaleshot", "firefang"]),
    ]),
    "trickroom": ("트릭룸", [
        ("hatterene", "magicbounce", "lifeorb", "quiet", {"hp": 32, "spa": 32, "def": 2}, ["trickroom", "psychic", "dazzlinggleam", "mysticalfire"]),
        ("farigiraf", "armortail", "sitrusberry", "quiet", {"hp": 32, "spa": 32, "def": 2}, ["trickroom", "psychic", "hypervoice", "protect"]),
        ("reuniclus", "magicguard", "leftovers", "quiet", {"hp": 32, "spa": 32, "def": 2}, ["trickroom", "psychic", "focusblast", "recover"]),
        ("conkeldurr", "ironfist", "muscleband", "brave", {"hp": 32, "atk": 32, "def": 2}, ["drainpunch", "machpunch", "icepunch", "knockoff"]),
        ("mawile", "intimidate", "mawilite", "brave", {"hp": 32, "atk": 32, "def": 2}, ["playrough", "suckerpunch", "ironhead", "swordsdance"]),
        ("torkoal", "drought", "wiseglasses", "quiet", {"hp": 32, "spa": 32, "def": 2}, ["eruption", "overheat", "earthpower", "solarbeam"]),
    ]),
    "stall": ("막이(맹독·회복)", [
        ("toxapex", "regenerator", "leftovers", "bold", {"hp": 32, "def": 32, "spd": 2}, ["toxic", "recover", "haze", "banefulbunker"]),
        ("corviknight", "pressure", "lumberry", "impish", {"hp": 32, "def": 32, "spd": 2}, ["bodypress", "irondefense", "roost", "bravebird"]),
        ("gliscor", "hypercutter", "yacheberry", "impish", {"hp": 32, "def": 32, "spd": 2}, ["earthquake", "toxic", "protect", "roost"]),
        ("skarmory", "sturdy", "shedshell", "impish", {"hp": 32, "def": 32, "spd": 2}, ["spikes", "roost", "bodypress", "irondefense"]),
        ("garganacl", "purifyingsalt", "chestoberry", "careful", {"hp": 32, "spd": 32, "def": 2}, ["saltcure", "recover", "irondefense", "bodypress"]),
        ("umbreon", "synchronize", "sitrusberry", "calm", {"hp": 32, "spd": 32, "def": 2}, ["foulplay", "wish", "protect", "toxic"]),
    ]),
    "batonpass": ("바톤터치", [
        ("scolipede", "speedboost", "focussash", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["swordsdance", "batonpass", "protect", "megahorn"]),
        ("espeon", "magicbounce", "lightclay", "timid", {"spa": 32, "spe": 32, "hp": 2}, ["calmmind", "batonpass", "psychic", "reflect"]),
        ("blaziken", "speedboost", "lifeorb", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["swordsdance", "batonpass", "flareblitz", "closecombat"]),
        ("kingambit", "supremeoverlord", "leftovers", "adamant", {"hp": 32, "atk": 32, "spd": 2}, ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"]),
        ("dragonite", "multiscale", "lumberry", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["extremespeed", "earthquake", "dragondance", "roost"]),
        ("gholdengo", "goodasgold", "choicescarf", "timid", {"spa": 32, "spe": 32, "hp": 2}, ["makeitrain", "shadowball", "focusblast", "trick"]),
    ]),
    "web": ("끈적끈적네트", [
        ("ariados", "insomnia", "focussash", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["stickyweb", "toxicspikes", "suckerpunch", "megahorn"]),
        ("kingambit", "supremeoverlord", "leftovers", "adamant", {"hp": 32, "atk": 32, "spd": 2}, ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"]),
        ("gyarados", "intimidate", "lumberry", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["dragondance", "waterfall", "earthquake", "crunch"]),
        ("ceruledge", "flashfire", "lifeorb", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["swordsdance", "bitterblade", "shadowsneak", "closecombat"]),
        ("garchomp", "roughskin", "yacheberry", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["swordsdance", "earthquake", "scaleshot", "stealthrock"]),
        ("mimikyu", "disguise", "spelltag", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["swordsdance", "shadowsneak", "playrough", "shadowclaw"]),
    ]),
    "hyperoffense": ("하이퍼오펜스", [
        ("garchomp", "roughskin", "focussash", "jolly", {"atk": 32, "spe": 32, "hp": 2}, ["stealthrock", "earthquake", "dragontail", "swordsdance"]),
        ("kingambit", "supremeoverlord", "blackglasses", "adamant", {"hp": 32, "atk": 32, "spd": 2}, ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"]),
        ("dragonite", "multiscale", "lumberry", "adamant", {"atk": 32, "spe": 32, "hp": 2}, ["dragondance", "extremespeed", "earthquake", "outrage"]),
        ("gholdengo", "goodasgold", "choicescarf", "timid", {"spa": 32, "spe": 32, "hp": 2}, ["makeitrain", "shadowball", "focusblast", "trick"]),
        ("dragapult", "infiltrator", "wiseglasses", "timid", {"spa": 32, "spe": 32, "hp": 2}, ["dracometeor", "shadowball", "flamethrower", "uturn"]),
        ("volcarona", "flamebody", "sitrusberry", "timid", {"spa": 32, "spe": 32, "hp": 2}, ["quiverdance", "fierydance", "bugbuzz", "gigadrain"]),
    ]),
    "standard": ("스탠다드(메타 밸런스)", [
        ("primarina", "torrent", "sitrusberry", "modest", {"hp": 32, "spa": 32, "spd": 2}, ["moonblast", "hydropump", "calmmind", "drainingkiss"]),
        ("gholdengo", "goodasgold", "leftovers", "bold", {"hp": 32, "def": 32, "spa": 2}, ["makeitrain", "shadowball", "nastyplot", "recover"]),
        ("garchomp", "roughskin", "yacheberry", "jolly", {"hp": 32, "spe": 32, "atk": 2}, ["earthquake", "dragontail", "stealthrock", "spikes"]),
        ("hippowdon", "sandstream", "lumberry", "impish", {"hp": 32, "def": 32, "spd": 2}, ["earthquake", "stealthrock", "slackoff", "yawn"]),
        ("archaludon", "stamina", "expertbelt", "modest", {"hp": 32, "spa": 32, "spd": 2}, ["electroshot", "dracometeor", "flashcannon", "bodypress"]),
        ("kingambit", "supremeoverlord", "chopleberry", "adamant", {"hp": 32, "atk": 32, "spd": 2}, ["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"]),
    ]),
}


def _mk(dex: Dex, row):
    sid, ab, it, nat, sp, mv = row
    return normalize_set(dex, {"species": sid, "ability": ab, "item": it, "nature": nat, "sp": dict(sp), "moves": list(mv)})


def teams(dex: Dex, game_type="singles") -> dict:
    src = SINGLES if game_type == "singles" else {}
    return {k: {"ko": ko, "team": [_mk(dex, r) for r in rows]} for k, (ko, rows) in src.items()}


def validate(dex: Dex, game_type="singles") -> list:
    """팀 정의 오류 목록 (비어 있으면 정상): 종·특성·도구·기술·SP·도구 클로즈·메가 1마리."""
    errs = []
    for key, (ko, rows) in (SINGLES if game_type == "singles" else {}).items():
        items, megas = [], 0
        for sid, ab, it, nat, sp, mv in rows:
            spd = dex.species.get(sid)
            if not spd:
                errs.append(f"{key}: 없는 종 {sid}")
                continue
            abs_ = {to_id(a) for a in spd["abilities"].values()}
            if ab not in abs_:
                errs.append(f"{key}/{sid}: 특성 {ab} 아님 ({sorted(abs_)})")
            if it and it not in dex.items:
                errs.append(f"{key}/{sid}: 없는 도구 {it}")
            if it and dex.items.get(it, {}).get("megaStone"):
                megas += 1
                if not dex.mega_for(sid, it):
                    errs.append(f"{key}/{sid}: {it} 로 메가진화 불가")
            if nat not in dex.natures:
                errs.append(f"{key}/{sid}: 없는 성격 {nat}")
            learn = set(dex.learnsets.get(sid) or dex.learnsets.get(to_id(spd.get("baseSpecies") or "")) or [])
            for m in mv:
                if m not in dex.moves:
                    errs.append(f"{key}/{sid}: 없는 기술 {m}")
                elif learn and m not in learn:
                    errs.append(f"{key}/{sid}: {m} 못 배움")
            if sum(sp.values()) > 66 or any(v > 32 or v < 0 for v in sp.values()):
                errs.append(f"{key}/{sid}: SP {sp}")
            if it:
                items.append(it)
        if len(items) != len(set(items)):
            errs.append(f"{key}: 같은 도구 둘 (도구 클로즈)")
        if megas > 1:
            errs.append(f"{key}: 메가스톤 {megas}개 — 한 판에 한 번만이라 하나만")
        if len(rows) != 6:
            errs.append(f"{key}: {len(rows)}마리")
    return errs


# ======================================================================== 통계
def wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def games_needed(delta: float = 0.05, z: float = 1.96) -> int:
    """50% 근처에서 승률 차이 delta 를 95% 신뢰로 구분하는 데 필요한 판 수."""
    return int(math.ceil((z * 0.5 / delta) ** 2))


# ======================================================================== 평가
def evaluate(dex: Dex, bridge, fmt: str, me_factory, opp_factory, game_type="singles", games=100, mode="mirror",
             keys=None, seed=1, log=print, max_turns=120) -> dict:
    """me_factory(seed)/opp_factory(seed) → 정책. mode 'mirror': 같은 팀 양쪽 / 'cross': 상대는 벤치마크 팀, 나는 keys 의 팀을 돌아가며.
    돌려줌: 팀별 {wins, played, winrate, ci}, 전체."""
    from .selfplay import play_game
    B = teams(dex, game_type)
    keys = keys or list(B)
    out, tw, tn = {}, 0, 0
    g_id = 0
    t0 = time.time()
    for k in keys:
        w = n = 0
        for i in range(games):
            my_team = B[k]["team"]
            opp_team = B[k]["team"] if mode == "mirror" else B[keys[(keys.index(k) + 1 + i) % len(keys)]]["team"]
            me, opp = me_factory(seed * 1000 + g_id), opp_factory(seed * 1000 + g_id + 500)
            if i % 2 == 0:
                pols, teams_ = {"p1": me, "p2": opp}, {"p1": my_team, "p2": opp_team}
                mine = "p1"
            else:
                pols, teams_ = {"p1": opp, "p2": me}, {"p1": opp_team, "p2": my_team}
                mine = "p2"
            try:
                rec = play_game(bridge, dex, fmt, teams_, pols, seed * 100000 + g_id, f"bm{g_id}", max_turns=max_turns, record=False)
            except Exception as e:
                log(f"  {k} {i}: 오류 {type(e).__name__}: {e}")
                g_id += 1
                continue
            g_id += 1
            if rec["winner"]:
                n += 1
                w += rec["winner"] == mine
        p, lo, hi = wilson(w, n)
        out[k] = {"ko": B[k]["ko"], "wins": w, "played": n, "winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)]}
        tw += w
        tn += n
        log(f"  {B[k]['ko']:14s} {w:3d}/{n:<3d} {p:6.1%}  [{lo:.0%}–{hi:.0%}]")
    p, lo, hi = wilson(tw, tn)
    out["_all"] = {"wins": tw, "played": tn, "winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
                   "seconds": round(time.time() - t0)}
    log(f"  전체 {tw}/{tn} {p:.1%} [{lo:.0%}–{hi:.0%}] · {time.time() - t0:.0f}초")
    return out
