# -*- coding: utf-8 -*-
"""변수(feature) 등록부.

변수는 '묶음' 단위로 등록한다. 묶음마다 이름·한글 설명·추가 날짜·종류(state/option)가 있고,
학습(engine/model.py)은 등록부에 있는 묶음 중 '채택'된 것만 쓴다.
새 묶음은 '후보'로 들어오고, 변수 장부(features/ledger.json)의 제거 실험(ablation)에서
효과가 확인되면 채택된다 — tools/train.py ablation.

모든 변수는 BattleView 스냅샷(그 플레이어가 볼 수 있는 정보) + 내 팀 세트에서만 계산한다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .belief import OppBelief, mon_from_view
from .dex import Dex, to_id
from .mon import Mon
from .policies import (HAZARDS, PROTECT, RECOVERY, SCREENS, SETUP_BOOST_MOVES, STATUS_INFLICT,
                       HeuristicPolicy)
from .protocol import split_ident

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "features" / "ledger.json"

WEATHERS = ["sunnyday", "raindance", "sandstorm", "snowscape"]
TERRAINS = ["electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"]
SIDE_KEYS = ["reflect", "lightscreen", "auroraveil", "tailwind", "stealthrock", "spikes", "toxicspikes", "stickyweb"]
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]
SPEED_CONTROL = {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "rocktomb", "bulldoze"}


@dataclass
class Group:
    key: str
    ko: str
    kind: str          # state | option
    size: int
    fn: object
    added: str
    desc: str


REGISTRY: list[Group] = []


def group(key, ko, kind, size, added, desc=""):
    def deco(fn):
        REGISTRY.append(Group(key, ko, kind, size, fn, added, desc))
        return fn
    return deco


# ======================================================================== 계산 문맥
class FCtx:
    """결정 하나에 대한 계산 문맥 (캐시 포함)."""

    def __init__(self, dex: Dex, view: dict, req: dict, my_sets: list, calc: HeuristicPolicy | None = None):
        self.dex, self.v, self.req = dex, view, req or {}
        self.me = view["me"]
        self.opp = "p2" if self.me == "p1" else "p1"
        self.doubles = view.get("gameType") == "doubles"
        self.kind = "preview" if self.req.get("teamPreview") else ("force" if self.req.get("forceSwitch") else "turn")
        if calc is None:
            calc = HeuristicPolicy(dex)
        self.c = calc
        self.c.sets = {dex.species[s["species"]]["name"]: s for s in my_sets if s["species"] in dex.species}
        self.field = calc.field_of(view)
        self.pick = view.get("pickSize") or (4 if self.doubles else 3)
        self._mine = {}
        self._cache = {}

    # ---- 포켓몬
    def my_views(self):
        """내 포켓몬 (선출 후에는 실제로 데려온 포켓몬만)."""
        mons = self.v["mons"][self.me]
        if self.kind != "preview" and self.req.get("side"):
            names = {split_ident(p["ident"])[2] for p in self.req["side"]}
            mons = [m for m in mons if m["name"] in names]
        return mons

    def opp_views(self):
        return self.v["mons"][self.opp]

    def my_active(self):
        act = sorted([m for m in self.my_views() if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
        return act

    def opp_active(self):
        return sorted([m for m in self.opp_views() if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])

    def mon(self, mv, mine: bool):
        key = (mine, mv["name"], mv.get("species"), round(mv.get("hpPct", 100)), mv.get("status"),
               json.dumps(mv.get("boosts") or {}, sort_keys=True), mv.get("item"))
        if key not in self._mine:
            try:
                self._mine[key] = self.c.my_mon(mv) if mine else self.c.opp_mon(mv)
            except Exception:
                self._mine[key] = None
        return self._mine[key]

    def bench_mon_by_req(self, k):
        side = self.req.get("side") or []
        if not 0 < k <= len(side):
            return None, None
        p = side[k - 1]
        _, _, name = split_ident(p["ident"])
        mv = next((m for m in self.my_views() if m["name"] == name), None)
        if mv is None:
            s = self.c.sets.get(name)
            if not s:
                return None, None
            mv = {"name": name, "species": s["species"], "hpPct": 100, "boosts": {}}
        return mv, self.mon(mv, True)

    def best(self, a: Mon, d: Mon, a_side: int):
        if a is None or d is None:
            return (0.0, 0.0, "")
        key = ("best", id(a), id(d), a_side)
        if key not in self._cache:
            self._cache[key] = self.c.best_vs(a, d, self.field, a_side)
        return self._cache[key]

    def teacher_part(self, slot: int, raw: str) -> float:
        """선생님 정책이 이 자리의 행동에 매기는 점수 (기본기 · 정책 사전확률 역할)."""
        key = ("teacher", slot, raw)
        if key in self._cache:
            return self._cache[key]
        if "tctx" not in self._cache:
            my = [self.mon(m, True) for m in self.my_active()]
            op = [self.mon(m, False) for m in self.opp_active()]
            self._cache["tctx"] = (my, op)
        my, op = self._cache["tctx"]
        try:
            v = self.c._part(slot, raw, my, op, self.field, self.req.get("active") or [],
                             lambda k: self.bench_mon_by_req(k)[1], self.doubles)
        except Exception:
            v = 0.0
        self._cache[key] = v
        return v

    def speed(self, m: Mon, side: int):
        if m is None:
            return 0
        key = ("spe", id(m), side)
        if key not in self._cache:
            self._cache[key] = self.c.speed(m, self.field, side)
        return self._cache[key]


def _pad(xs, n, fill=0.0):
    xs = list(xs)[:n]
    return xs + [fill] * (n - len(xs))


# ======================================================================== 상태 변수
@group("s_hp", "남은 HP", "state", 10, "2026-10-15", "내·상대 활성 HP, 살아있는 수, 팀 HP 합")
def f_hp(x: FCtx):
    mine = [m for m in x.my_views()]
    opp = x.opp_views()
    my_alive = [m["hpPct"] / 100 for m in mine if not m["fainted"]]
    if x.kind == "preview":
        my_alive = [1.0] * x.pick
    opp_rev = [m for m in opp if m.get("revealed")]
    unrevealed = max(0, x.pick - len(opp_rev))
    opp_alive = [m["hpPct"] / 100 for m in opp_rev if not m["fainted"]] + [1.0] * unrevealed
    ma = [m["hpPct"] / 100 for m in x.my_active()]
    oa = [m["hpPct"] / 100 for m in x.opp_active()]
    my_sum = sum(my_alive[:x.pick]) / x.pick
    op_sum = sum(opp_alive) / x.pick
    return [my_sum, op_sum, my_sum - op_sum, len(my_alive) / x.pick, len(opp_alive) / x.pick,
            (len(my_alive) - len(opp_alive)) / x.pick] + _pad(ma, 2) + _pad(oa, 2)


@group("s_status", "상태이상", "state", 18, "2026-10-15", "활성 포켓몬의 화상·마비·독·잠듦, 팀 상태이상 수")
def f_status(x: FCtx):
    def enc(m):
        s = m.get("status") if m else ""
        return [s == "brn", s == "par", s in ("psn", "tox"), s in ("slp", "frz")]
    out = []
    for act in (x.my_active(), x.opp_active()):
        act = _pad(act, 2, None)
        for m in act:
            out += enc(m)
    out.append(sum(1 for m in x.my_views() if m.get("status") and not m["fainted"]) / 3)
    out.append(sum(1 for m in x.opp_views() if m.get("status") and not m["fainted"]) / 3)
    return [float(v) for v in out]


@group("s_boosts", "랭크 변화", "state", 20, "2026-10-15", "활성 포켓몬의 공격·방어·특공·특방·스피드 랭크")
def f_boosts(x: FCtx):
    out = []
    for act in (x.my_active(), x.opp_active()):
        for m in _pad(act, 2, None):
            b = (m or {}).get("boosts") or {}
            out += [b.get(k, 0) / 6 for k in BOOST_KEYS]
    return out


@group("s_field", "날씨·필드", "state", 12, "2026-10-15", "날씨·필드 종류와 경과 턴, 트릭룸, 중력")
def f_field(x: FCtx):
    v = x.v
    out = [float(v["weather"] == w) for w in WEATHERS] + [min(v.get("weatherTurns", 0), 8) / 8]
    out += [float(v["terrain"] == t) for t in TERRAINS] + [min(v.get("terrainTurns", 0), 8) / 8]
    out += [float("trickroom" in v["pseudo"]), float("gravity" in v["pseudo"])]
    return out


@group("s_side", "진영 효과", "state", 16, "2026-10-15", "벽·순풍·스텔스록·압정 등 (내/상대)")
def f_side(x: FCtx):
    out = []
    for s in (x.me, x.opp):
        sc = x.v["sideConds"].get(s, {})
        for k in SIDE_KEYS:
            e = sc.get(k)
            out.append((e["layers"] / 3 if k in ("spikes", "toxicspikes") else 1.0) if e else 0.0)
    return out


@group("s_speed", "스피드 비교", "state", 4, "2026-10-15", "활성끼리 누가 먼저 움직이는가 (트릭룸·순풍·스카프·마비 반영)")
def f_speed(x: FCtx):
    out = []
    oa = [x.mon(m, False) for m in x.opp_active()]
    for m in _pad(x.my_active(), 2, None):
        if not m or not oa:
            out += [0.0, 0.0]
            continue
        me = x.mon(m, True)
        s = x.speed(me, 0)
        so = [x.speed(o, 1) for o in oa if o]
        if not so or me is None:
            out += [0.0, 0.0]
            continue
        out += [1.0 if s > max(so) else (-1.0 if s < min(so) else 0.0),
                math.tanh((abs(s) - abs(max(so))) / 50)]
    return out


@group("s_matchup", "데미지 상성", "state", 8, "2026-10-15", "데미지 엔진으로 계산한 내→상대, 상대→나 최대 기대 데미지와 처치 확률")
def f_matchup(x: FCtx):
    out = []
    oa = [x.mon(m, False) for m in x.opp_active()]
    for m in _pad(x.my_active(), 2, None):
        me = x.mon(m, True) if m else None
        if me is None or not oa:
            out += [0.0] * 4
            continue
        give = [x.best(me, o, 0) for o in oa if o]
        take = [x.best(o, me, 1) for o in oa if o]
        out += [max((g[0] for g in give), default=0), max((g[1] for g in give), default=0),
                max((t[0] for t in take), default=0), max((t[1] for t in take), default=0)]
    return out


@group("s_bench", "교체 후보", "state", 4, "2026-10-15", "대기 중인 내 포켓몬의 최고 교체 상성, 상대 미공개 수")
def f_bench(x: FCtx):
    oa = [x.mon(m, False) for m in x.opp_active()]
    best = -1.0
    n = 0
    for k, p in enumerate(x.req.get("side") or [], 1):
        if p.get("active") or p["condition"].endswith("fnt"):
            continue
        mv, m = x.bench_mon_by_req(k)
        if m is None or not oa:
            continue
        n += 1
        sc = 0.0
        for o in oa:
            if o:
                g, t = x.best(m, o, 0), x.best(o, m, 1)
                sc += (g[0] + g[1]) - (t[0] + t[1])
        best = max(best, sc / len(oa))
    opp_rev = sum(1 for m in x.opp_views() if m.get("revealed"))
    return [best if n else 0.0, n / 3, max(0, x.pick - opp_rev) / x.pick, float(n > 0)]


@group("s_types", "타입 상성", "state", 4, "2026-10-15", "활성 포켓몬 자속 타입끼리의 상성 (log2)")
def f_types(x: FCtx):
    dex = x.dex

    def eff(t, defender_types):
        if any(dex.immune(t, d) for d in defender_types):
            return -3.0
        return float(sum(dex.effectiveness(t, d) for d in defender_types))

    def types_of(m):
        sp = dex.species.get(m["species"]) if m else None
        return sp["types"] if sp else []
    ma = [types_of(m) for m in x.my_active()]
    oa = [types_of(m) for m in x.opp_active()]
    if not ma or not oa:
        return [0.0] * 4
    give = [max(eff(t, d) for t in a) for a in ma for d in oa if a and d]
    take = [max(eff(t, d) for t in o) for o in oa for d in ma if o and d]
    return [max(give, default=0) / 2, min(give, default=0) / 2, max(take, default=0) / 2, min(take, default=0) / 2]


@group("s_mega", "메가진화", "state", 4, "2026-10-15", "메가진화 사용 여부와 지금 가능한지")
def f_mega(x: FCtx):
    can = any(a.get("canMegaEvo") for a in x.req.get("active") or [])
    opp_mega_possible = any(
        any(x.dex.species.get(f, {}).get("isMega") for f in [to_id(o) for o in x.dex.species.get(s, {}).get("otherFormes", [])])
        for s in x.v["preview"].get(x.opp, []))
    return [float(x.v["megaUsed"].get(x.me)), float(x.v["megaUsed"].get(x.opp)), float(can), float(opp_mega_possible)]


@group("s_info", "정보량", "state", 3, "2026-10-15", "상대에 대해 드러난 기술·도구·특성 비율")
def f_info(x: FCtx):
    rev = [m for m in x.opp_views() if m.get("revealed")]
    if not rev:
        return [0.0, 0.0, 0.0]
    return [sum(len(m.get("moves") or []) for m in rev) / (4 * len(rev)),
            sum(1 for m in rev if m.get("item") is not None) / len(rev),
            sum(1 for m in rev if m.get("ability")) / len(rev)]


@group("s_turn", "턴·결정 종류", "state", 4, "2026-10-15", "턴 수, 선출/강제교체/일반 턴")
def f_turn(x: FCtx):
    return [min(x.v["turn"], 40) / 40, float(x.kind == "preview"), float(x.kind == "force"), float(x.kind == "turn")]


@group("s_stats", "종족값", "state", 12, "2026-10-15", "첫 번째 활성 포켓몬(나·상대)의 종족값")
def f_stats(x: FCtx):
    out = []
    for act in (x.my_active(), x.opp_active()):
        sp = x.dex.species.get(act[0]["species"]) if act else None
        out += [sp["baseStats"][k] / 150 for k in ("hp", "atk", "def", "spa", "spd", "spe")] if sp else [0.0] * 6
    return out


# ======================================================================== 선택지 변수 (자리마다)
PART_GROUPS = []


def part_group(key, ko, size, added, desc=""):
    def deco(fn):
        PART_GROUPS.append(Group(key, ko, "option", size, fn, added, desc))
        return fn
    return deco


def _parse_part(part: str):
    toks = part.split()
    if not toks or toks[0] == "pass":
        return {"kind": "pass"}
    if toks[0] == "switch":
        return {"kind": "switch", "k": int(toks[1])}
    if toks[0] == "team":
        return {"kind": "team"}
    out = {"kind": "move", "j": int(toks[1]), "mega": toks[-1] == "mega", "tgt": None}
    if len(toks) >= 3 and toks[2].lstrip("-").isdigit():
        out["tgt"] = int(toks[2])
    return out


def _move_of(x: FCtx, slot: int, j: int):
    act = x.req.get("active") or []
    if slot >= len(act) or j - 1 >= len(act[slot].get("moves", [])):
        return None
    return x.dex.moves.get(to_id(act[slot]["moves"][j - 1].get("id", "")))


@part_group("o_kind", "행동 종류", 4, "2026-10-15", "기술/교체/메가진화/패스")
def p_kind(x, slot, p):
    return [float(p["kind"] == "move"), float(p["kind"] == "switch"), float(p.get("mega", False)), float(p["kind"] == "pass")]


@part_group("o_move", "기술 기본 정보", 10, "2026-10-15", "위력·명중·우선도·분류·범위·자속·접촉·반동")
def p_move(x, slot, p):
    if p["kind"] != "move":
        return [0.0] * 10
    mv = _move_of(x, slot, p["j"])
    if not mv:
        return [0.0] * 10
    act = x.my_active()
    sp = x.dex.species.get(act[slot]["species"]) if slot < len(act) else None
    acc = mv.get("accuracy")
    return [(mv.get("basePower") or 0) / 150, 1.0 if acc is True else (acc or 100) / 100, mv.get("priority", 0) / 3,
            float(mv["category"] == "Physical"), float(mv["category"] == "Special"), float(mv["category"] == "Status"),
            float(mv.get("target") in ("allAdjacentFoes", "allAdjacent")),
            float(bool(sp) and mv["type"] in sp["types"]), float(bool((mv.get("flags") or {}).get("contact"))),
            float(bool(mv.get("recoil"))) - float(bool(mv.get("drain")))]


@part_group("o_effect", "기술 효과 계산", 5, "2026-10-15", "데미지 엔진: 대상 상성·기대 데미지·처치 확률·아군 공격·무효")
def p_effect(x, slot, p):
    if p["kind"] != "move":
        return [0.0] * 5
    mv = _move_of(x, slot, p["j"])
    act = x.my_active()
    if not mv or mv["category"] == "Status" or slot >= len(act):
        return [0.0] * 5
    me = x.mon(act[slot], True)
    if me is None:
        return [0.0] * 5
    if p.get("mega") and not me.mega:
        me = Mon.from_dict(x.dex, me.to_dict())
        me.mega_evolve()
    opps = [x.mon(m, False) for m in x.opp_active()]
    if p["tgt"] is not None and p["tgt"] < 0:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    if p["tgt"] and 0 < p["tgt"] <= len(opps) and mv.get("target") not in ("allAdjacentFoes", "allAdjacent"):
        opps = [opps[p["tgt"] - 1]]
    fr, ko, te, imm = 0.0, 0.0, -9.0, 1.0
    for o in opps:
        if o is None:
            continue
        f, k = x.c.exp_frac(me, o, mv["id"], x.field, 0,
                            spread=x.doubles and mv.get("target") in ("allAdjacentFoes", "allAdjacent") and len(opps) > 1)
        fr, ko = max(fr, f), max(ko, k)
        e = sum(x.dex.effectiveness(mv["type"], t) for t in o.types)
        te = max(te, e)
        if f > 0:
            imm = 0.0
    return [max(te, -2) / 2, fr, ko, 0.0, imm]


@part_group("o_status", "변화기 종류", 7, "2026-10-15", "능력 올리기·회복·방어·상태이상·설치·벽·스피드 조절")
def p_status(x, slot, p):
    if p["kind"] != "move":
        return [0.0] * 7
    mv = _move_of(x, slot, p["j"])
    if not mv:
        return [0.0] * 7
    i = mv["id"]
    return [float(i in SETUP_BOOST_MOVES), float(i in RECOVERY), float(i in PROTECT), float(i in STATUS_INFLICT),
            float(i in HAZARDS), float(i in SCREENS), float(i in SPEED_CONTROL)]


@part_group("o_switch", "교체 상성", 4, "2026-10-15", "들어올 포켓몬이 받을/줄 데미지, HP, 스피드")
def p_switch(x, slot, p):
    if p["kind"] != "switch":
        return [0.0] * 4
    mv, m = x.bench_mon_by_req(p["k"])
    opps = [x.mon(o, False) for o in x.opp_active()]
    if m is None or not opps:
        return [0.0] * 4
    take = max((x.best(o, m, 1)[0] for o in opps if o), default=0)
    give = max((x.best(m, o, 0)[0] for o in opps if o), default=0)
    so = max((x.speed(o, 1) for o in opps if o), default=0)
    return [take, give, (mv or {}).get("hpPct", 100) / 100, float(x.speed(m, 0) > so)]


@part_group("o_risk", "행동 위험", 3, "2026-10-15", "행동하는 포켓몬이 먼저 맞고 쓰러질 위험")
def p_risk(x, slot, p):
    act = x.my_active()
    if slot >= len(act) or p["kind"] == "pass":
        return [0.0] * 3
    me = x.mon(act[slot], True)
    opps = [x.mon(o, False) for o in x.opp_active()]
    if me is None or not opps:
        return [0.0] * 3
    so = max((x.speed(o, 1) for o in opps if o), default=0)
    ko = max((x.best(o, me, 1)[1] for o in opps if o), default=0)
    return [float(x.speed(me, 0) < so), ko, act[slot]["hpPct"] / 100]


@part_group("o_preview", "선출 조합", 6, "2026-10-15", "선출한 포켓몬들의 평균·선봉·최저 상성, 메가 포함, 스피드")
def p_preview(x, slot, p):
    if p["kind"] != "team":
        return [0.0] * 6
    return list(x._cache.get(("preview_feats", p["order"]), [0.0] * 6))


@part_group("o_teacher", "선생님 판단", 1, "2026-10-15", "데미지 엔진 기반 선생님 정책의 행동 점수 (알파고의 지도학습 정책 역할)")
def p_teacher(x, slot, p):
    if p["kind"] in ("pass", "team") or "raw" not in p:
        return [0.0]
    return [max(-3.0, min(3.0, x.teacher_part(slot, p["raw"])))]


PART_SIZE = sum(g.size for g in PART_GROUPS)


# ======================================================================== 벡터 만들기
def active_groups(ledger: dict | None = None, exclude=()):
    """채택된 묶음 (장부에서 rejected 된 것과 exclude 는 뺀다)."""
    ledger = ledger if ledger is not None else load_ledger()
    rejected = {k for k, v in ledger.get("groups", {}).items() if v.get("status") == "rejected"}
    st = [g for g in REGISTRY if g.key not in rejected and g.key not in exclude]
    op = [g for g in PART_GROUPS if g.key not in rejected and g.key not in exclude]
    return st, op


def state_vector(x: FCtx, groups) -> list:
    out = []
    for g in groups:
        try:
            v = g.fn(x)
        except Exception:
            v = [0.0] * g.size
        if len(v) != g.size:
            v = _pad(v, g.size)
        out += [float(a) for a in v]
    return out


def option_vector(x: FCtx, option: str, groups) -> list:
    parts = [p.strip() for p in option.split(",")]
    if option.startswith("team"):
        order = tuple(int(c) for c in option.split()[1])
        _preview_feats(x, order)
        pp = [{"kind": "team", "order": order}, {"kind": "pass"}]
    else:
        pp = []
        for p in parts:
            d = _parse_part(p)
            d["raw"] = p
            pp.append(d)
    pp = _pad(pp, 2, {"kind": "pass"})
    out = []
    for slot, p in enumerate(pp[:2]):
        for g in groups:
            try:
                v = g.fn(x, slot, p)
            except Exception:
                v = [0.0] * g.size
            out += [float(a) for a in _pad(v, g.size)]
    return out


def _preview_feats(x: FCtx, order):
    key = ("preview_feats", order)
    if key in x._cache:
        return
    side = x.req.get("side") or []
    opp = [x.c._belief(s).assumed_mon() for s in x.v["preview"].get(x.opp, []) if s in x.dex.species]
    scores, speeds, megas = [], [], 0
    for k in order:
        if k - 1 >= len(side):
            continue
        _, _, name = split_ident(side[k - 1]["ident"])
        s = x.c.sets.get(name)
        if not s:
            continue
        mkey = ("pm", name)
        if mkey not in x._cache:
            m = Mon(species=s["species"], ability=s["ability"], item=s["item"], nature=s["nature"],
                    sp=s["sp"], moves=s["moves"]).setup(x.dex)
            if s.get("megaTarget"):
                m.mega_evolve()
            v = 0.0
            for o in opp:
                g, t = x.best(m, o, 0), x.best(o, m, 1)
                v += (g[0] + g[1]) - (t[0] + t[1])
            x._cache[mkey] = (v / max(1, len(opp)), m.stats["spe"], bool(s.get("megaTarget")))
        sc, spe, mg = x._cache[mkey]
        scores.append(sc)
        speeds.append(spe)
        megas += mg
    if not scores:
        x._cache[key] = [0.0] * 6
        return
    lead_n = 1 if len(order) <= 3 else 2
    x._cache[key] = [sum(scores) / len(scores), sum(scores[:lead_n]) / lead_n, min(scores),
                     sum(1 for s in scores if s > 0) / len(scores), float(megas > 0), sum(speeds) / len(speeds) / 200]


def dims(st_groups, op_groups):
    return sum(g.size for g in st_groups), 2 * sum(g.size for g in op_groups)


# ======================================================================== 장부
def load_ledger() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"groups": {}, "suggestions": [], "history": []}


def save_ledger(led: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")


def sync_ledger() -> dict:
    """등록부에 새로 생긴 묶음을 장부에 '후보'로 올린다."""
    led = load_ledger()
    for g in REGISTRY + PART_GROUPS:
        e = led["groups"].setdefault(g.key, {"ko": g.ko, "kind": g.kind, "size": g.size, "added": g.added,
                                             "desc": g.desc, "status": "candidate", "tests": []})
        e.update({"ko": g.ko, "kind": g.kind, "size": g.size, "desc": g.desc})
    save_ledger(led)
    return led


# ======================================================================== 셀프테스트
def selfcheck(dex: Dex) -> dict:
    """가짜 대전 스냅샷 하나로 모든 묶음이 정해진 크기의 유한한 숫자를 내는지 확인."""
    from .protocol import BattleView
    errs = []
    v = BattleView("p1")
    v.feed(["|gametype|singles", "|poke|p1|Garchomp, L50, M|", "|poke|p2|Charizard, L50, M|", "|teampreview|3",
            "|switch|p1a: Garchomp|Garchomp, L50, M|183/183", "|switch|p2a: Charizard|Charizard, L50, M|48/48",
            "|turn|1", "|move|p2a: Charizard|Flamethrower|p1a: Garchomp", "|-damage|p1a: Garchomp|120/183",
            "|-weather|SunnyDay", "|-boost|p1a: Garchomp|atk|2"])
    team = [{"species": "garchomp", "ability": "roughskin", "item": "lifeorb", "nature": "jolly",
             "sp": {"atk": 32, "spe": 32, "hp": 2}, "moves": ["earthquake", "dragonclaw", "stoneedge", "swordsdance"]}]
    req = {"side": [{"ident": "p1: Garchomp", "details": "Garchomp, L50, M", "condition": "120/183", "active": True}],
           "active": [{"moves": [{"id": m} for m in team[0]["moves"]]}]}
    x = FCtx(dex, v.snapshot(), req, team)
    st, op = active_groups({"groups": {}})
    try:
        s = state_vector(x, st)
        o = option_vector(x, "move 1", op)
        sd, od = dims(st, op)
        if len(s) != sd or len(o) != od:
            errs.append(f"크기 불일치 state {len(s)}/{sd} option {len(o)}/{od}")
        if not all(math.isfinite(a) for a in s + o):
            errs.append("유한하지 않은 값")
        for g in st:
            vv = g.fn(x)
            if len(vv) != g.size:
                errs.append(f"{g.key}: 크기 {len(vv)} ≠ {g.size}")
        # 지진은 리자몽(비행)에게 무효 → o_effect 의 '무효' 칸이 1 이어야 한다
        off = sum(g.size for g in op[:[g.key for g in op].index("o_effect")])
        if o[off + 4] != 1.0:
            errs.append("지진→리자몽 무효가 변수에 반영되지 않음")
        o2 = option_vector(x, "move 2", op)
        if not o2[off + 1] > 0:
            errs.append("드래곤클로 기대 데미지가 0")
    except Exception as e:
        errs.append(f"{type(e).__name__}: {e}")
    return {"ok": not errs, "errors": errs,
            "summary": f"상태 묶음 {len(st)}개 · 선택지 묶음 {len(op)}개 · 입력 {sum(dims(st, op))}칸"}
