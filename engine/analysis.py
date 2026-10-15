# -*- coding: utf-8 -*-
"""대전 후 분석 — 모델 예측이 크게 빗나간 결정에 어떤 상황이 자주 끼어 있는지 찾는다.

빗나감(surprise) = |실제 결과 − 예측 승률|. 상황 태그별로 '크게 빗나간 비율'을 전체와 비교해
배율(lift)이 높고 표본이 충분하면 '다음에 넣을 변수 후보'로 올린다.
운(급소·빗나감)처럼 변수로 고칠 수 없는 것은 따로 표시한다.
"""
from __future__ import annotations

from collections import defaultdict

from . import features as F
from .policies import HeuristicPolicy

TAGS = {
    # 태그: (변수 후보 이름, 설명, 운 요소인가)
    "opp_sub": ("대타출동 상태", "상대가 대타출동 중", False),
    "my_sub": ("내 대타출동 상태", "내가 대타출동 중", False),
    "confusion": ("혼란 상태", "활성 포켓몬이 혼란", False),
    "taunt_encore": ("도발·앵콜·사슬묶기", "행동 제한 상태가 걸림", False),
    "leechseed": ("씨뿌리기·저주 지속 데미지", "매 턴 깎이는 상태", False),
    "protect_last": ("직전 방어 사용", "상대가 직전 턴에 방어 계열 사용 (연속 방어 실패 확률)", False),
    "opp_item_unknown_full": ("기합의띠 가능성", "상대 HP 가득 + 도구 미공개", False),
    "opp_priority": ("상대 선공기 보유", "상대가 우선도 기술을 드러냄", False),
    "low_hp_both": ("서로 빈사", "양쪽 활성 HP 25% 이하", False),
    "big_boost": ("큰 랭크업", "어느 쪽이든 +2 이상", False),
    "long_game": ("장기전", "15턴 이후", False),
    "opp_mega_pending": ("상대 메가진화 대기", "상대가 메가진화 가능 종을 냈지만 아직 안 함", False),
    "screens": ("벽 남은 턴", "벽이 깔려 있음", False),
    "weather_end": ("날씨 남은 턴", "날씨 4턴 이상 경과 (곧 끝남)", False),
    "trickroom": ("트릭룸 남은 턴", "트릭룸 중", False),
    "doubles_ally_low": ("더블 파트너 상태", "더블에서 내 파트너가 빈사", False),
    "recent_crit": ("급소", "직전 턴에 급소 발생", True),
    "recent_miss": ("빗나감", "직전 턴에 기술 빗나감", True),
}


def tags_of(view: dict) -> set:
    me = view["me"]
    opp = "p2" if me == "p1" else "p1"
    act = lambda s: [m for m in view["mons"][s] if m["slot"] and not m["fainted"]]
    ma, oa = act(me), act(opp)
    out = set()
    vol = lambda ms: set().union(*[set(m.get("volatiles") or []) for m in ms]) if ms else set()
    if "substitute" in vol(oa):
        out.add("opp_sub")
    if "substitute" in vol(ma):
        out.add("my_sub")
    if "confusion" in vol(ma) | vol(oa):
        out.add("confusion")
    if {"taunt", "encore", "disable"} & (vol(ma) | vol(oa)):
        out.add("taunt_encore")
    if {"leechseed", "curse", "saltcure"} & (vol(ma) | vol(oa)):
        out.add("leechseed")
    if any(m.get("lastMove") in ("protect", "detect", "spikyshield", "kingsshield", "banefulbunker") for m in oa):
        out.add("protect_last")
    if any(m["hpPct"] >= 99.9 and m.get("item") is None for m in oa):
        out.add("opp_item_unknown_full")
    if ma and oa and all(m["hpPct"] <= 25 for m in ma[:1] + oa[:1]):
        out.add("low_hp_both")
    if any(abs(v) >= 2 for m in ma + oa for v in (m.get("boosts") or {}).values()):
        out.add("big_boost")
    if view["turn"] >= 15:
        out.add("long_game")
    if not view["megaUsed"].get(opp) and any("mega" not in m["species"] and m.get("item") is None for m in oa):
        out.add("opp_mega_pending")
    sc = view["sideConds"]
    if any(k in sc.get(s, {}) for s in (me, opp) for k in ("reflect", "lightscreen", "auroraveil")):
        out.add("screens")
    if view["weather"] and view.get("weatherTurns", 0) >= 4:
        out.add("weather_end")
    if "trickroom" in view["pseudo"]:
        out.add("trickroom")
    if view["gameType"] == "doubles" and len(ma) == 2 and min(m["hpPct"] for m in ma) <= 25:
        out.add("doubles_ally_low")
    turn = view["turn"]
    for ev in view.get("recent") or []:
        if ev[0] >= turn - 1:
            if ev[1] == "crit":
                out.add("recent_crit")
            if ev[1] == "miss":
                out.add("recent_miss")
    for m in oa:
        for mv in m.get("moves") or []:
            if mv in ("fakeout", "extremespeed", "suckerpunch", "aquajet", "machpunch", "bulletpunch",
                      "iceshard", "shadowsneak", "quickattack", "vacuumwave", "jetpunch", "grassyglide"):
                out.add("opp_priority")
    return out


def analyze_surprises(dex, model, games, game_type, max_games=300, thresh=0.6, min_count=25, min_lift=1.3):
    calc = HeuristicPolicy(dex)
    total = big = 0
    tag_n = defaultdict(int)
    tag_big = defaultdict(int)
    n = 0
    for g in games:
        if g.get("gameType") != game_type or g.get("winner") not in ("p1", "p2"):
            continue
        n += 1
        if n > max_games:
            break
        calc.beliefs = {}
        for d in g["decisions"]:
            if d["kind"] == "preview":
                continue
            s = d["side"]
            x = F.FCtx(dex, d["view"], d["req"], g["teams"][s], calc)
            try:
                _, v = model.evaluate(x, [])
            except Exception:
                continue
            y = 1.0 if g["winner"] == s else 0.0
            surprise = abs(y - v) > thresh
            total += 1
            big += surprise
            for t in tags_of(d["view"]):
                tag_n[t] += 1
                tag_big[t] += surprise
    if not total or not big:
        return []
    base = big / total
    out = []
    for t, cnt in tag_n.items():
        if cnt < min_count:
            continue
        rate = tag_big[t] / cnt
        lift = rate / base if base else 0
        if lift >= min_lift:
            name, why, luck = TAGS[t]
            out.append({"name": name + (" (운 요소 — 변수로 개선 어려움)" if luck else ""), "tag": t,
                        "why": f"{why} 상황에서 예측이 크게 빗나간 비율이 평소의 {lift:.1f}배",
                        "evidence": f"{game_type} {cnt}개 결정 중 {tag_big[t]}개 (평소 {base:.0%})",
                        "lift": round(lift, 2), "luck": luck})
    return sorted(out, key=lambda s: -s["lift"])
