# -*- coding: utf-8 -*-
"""셀프테스트: 쇼다운 시뮬레이터 쪽 (Node.js 와 쇼다운 폴더가 있을 때만).

  1. 끝까지 대전: 자리를 바꿔도 승률이 대칭인가 (승자 집계 버그 방지)
  2. 수읽기: 상대 마지막 한 마리를 확정으로 쓰러뜨리는 수의 평가가 1(승리)에 가까운가
  3. 선출 승률표: 값이 모두 같지(0.5) 않고 정상적으로 채워지는가
"""
from __future__ import annotations

import random

from .dex import Dex
from .psbridge import Bridge, find_ps_dir
from .teamgen import to_showdown_set

TEAM_A = [
    {"species": "garchomp", "ability": "roughskin", "item": "choicescarf", "nature": "jolly", "sp": {"atk": 32, "spe": 32, "hp": 2},
     "moves": ["earthquake", "outrage", "rockslide", "poisonjab"]},
    {"species": "rotomwash", "ability": "levitate", "item": "sitrusberry", "nature": "bold", "sp": {"hp": 32, "def": 32, "spd": 2},
     "moves": ["hydropump", "voltswitch", "willowisp", "protect"]},
    {"species": "kingambit", "ability": "supremeoverlord", "item": "blackglasses", "nature": "adamant", "sp": {"hp": 32, "atk": 32, "spd": 2},
     "moves": ["kowtowcleave", "ironhead", "suckerpunch", "swordsdance"]},
]
TEAM_B = [
    {"species": "dragonite", "ability": "multiscale", "item": "lumberry", "nature": "adamant", "sp": {"hp": 32, "atk": 32, "spe": 2},
     "moves": ["extremespeed", "dragondance", "earthquake", "outrage"]},
    {"species": "tyranitar", "ability": "sandstream", "item": "leftovers", "nature": "careful", "sp": {"hp": 32, "spd": 32, "atk": 2},
     "moves": ["stoneedge", "crunch", "stealthrock", "earthquake"]},
    {"species": "primarina", "ability": "liquidvoice", "item": "sitrusberry", "nature": "modest", "sp": {"hp": 32, "spa": 32, "spd": 2},
     "moves": ["moonblast", "hypervoice", "psychicnoise", "protect"]},
]


def run() -> dict:
    if find_ps_dir() is None:
        return {"ok": True, "skipped": "쇼다운 폴더 없음 (PS_DIR) — 시뮬레이터 검사 생략"}
    dex = Dex()
    errs, notes = [], []
    try:
        with Bridge() as br:
            pa = br.pack([to_showdown_set(dex, s) for s in TEAM_A])
            pb = br.pack([to_showdown_set(dex, s) for s in TEAM_B])
            fmt = "gen9championscustomgame"
            # 1. 대칭
            r1 = br.call(op="matrix", format=fmt, A={"packed": pa}, B={"worlds": [pb]}, ordersA=["123"], ordersB=["123"], games=40, seed=5)
            r2 = br.call(op="matrix", format=fmt, A={"packed": pb}, B={"worlds": [pa]}, ordersA=["123"], ordersB=["123"], games=40, seed=6)
            w1, w2 = r1["M"][0][0], r2["M"][0][0]
            notes.append(f"A 팀 승률 {w1:.0%} / 자리 바꿔 B 팀 승률 {w2:.0%}")
            if abs((w1 + w2) - 1.0) > 0.3:
                errs.append(f"자리를 바꾸면 승률이 대칭이 아님: {w1:.2f} + {w2:.2f}")
            if w1 in (0.0, 0.5) and w2 in (0.0, 0.5):
                errs.append("승자가 집계되지 않음 (전부 0 또는 0.5)")
            # 2. 수읽기: 상대 마지막 1마리, HP 1% → 확정 처치 수의 평가
            sc = {
                "gameType": "singles", "mySide": "p1", "oppSide": "p2",
                "field": {"weather": "", "terrain": "", "trickRoom": False, "gravity": False},
                "sides": {}, "megaUsed": {"p1": False, "p2": False}, "turn": 5,
                "options": ["move 1", "move 2"], "rollouts": 2, "depth": 2, "seed": 3, "leaf": "mix",
                "p1": {"packed": pa, "state": [{"hp": 150}, {"hp": 157}, {"hp": 207}]},
                "worlds": [{"packed": pb, "state": [{"hpPct": 1}, {"fainted": True}, {"fainted": True}]}],
            }
            sc["p2"] = sc["worlds"][0]
            res = br.call(op="search", **sc)
            v = [x["value"] for x in res["results"]]
            notes.append(f"마지막 한 마리 처치 수 평가 {v}")
            if not v or v[0] is None or max(x for x in v if x is not None) < 0.9:
                errs.append(f"이기는 수를 승리로 평가하지 못함: {v}")
            # 3. 선출 승률표
            from . import bring
            six = TEAM_A + [
                {"species": "sylveon", "ability": "pixilate", "item": "wiseglasses", "nature": "modest", "sp": {"hp": 32, "spa": 32, "spd": 2},
                 "moves": ["hypervoice", "moonblast", "calmmind", "protect"]},
                {"species": "gengar", "ability": "cursedbody", "item": "focussash", "nature": "timid", "sp": {"spa": 32, "spe": 32, "hp": 2},
                 "moves": ["shadowball", "sludgebomb", "thunderbolt", "willowisp"]},
                {"species": "charizard", "ability": "blaze", "item": "charizarditey", "nature": "timid", "sp": {"spa": 32, "spe": 32, "hp": 2},
                 "moves": ["heatwave", "solarbeam", "airslash", "protect"]},
            ]
            rb = bring.recommend(dex, br, "gen9championsbssregmb", six,
                                 ["dragonite", "tyranitar", "gyarados", "corviknight", "volcarona", "hippowdon"],
                                 k_mine=4, k_opp=4, worlds=2, games=4, seed=random.randint(1, 999), budget_ms=20000)
            flat = [x for row in rb["matrix"] for x in row]
            notes.append(f"선출 승률표 {len(flat)}칸, 범위 {min(flat):.2f}~{max(flat):.2f}")
            if len(set(flat)) <= 1:
                errs.append("선출 승률표가 모두 같은 값")
    except Exception as e:
        errs.append(f"{type(e).__name__}: {e}")
    return {"ok": not errs, "errors": errs, "summary": " · ".join(notes) if notes else "실행 실패"}
