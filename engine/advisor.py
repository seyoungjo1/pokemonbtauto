# -*- coding: utf-8 -*-
"""대전 중 추천 — 사용자가 화면에서 입력한 상황(보이는 것) → 행동별 예상 승률과 근거.

입력 상태 (앱이 보내는 JSON)
  gameType, turn, myTeam[세트], brought[내 팀 번호, 앞이 선봉], oppPreview[종족]
  mine[{idx, hpPct, status, boosts, slot('a'/'b'/None), fainted, mega, item, volatiles}]
  opp[{species, hpPct, status, boosts, slot, fainted, moves[], item(None=모름), ability, mega, volatiles}]
  field{weather, weatherTurns, terrain, terrainTurns, trickRoom, gravity}
  sides{me{reflect,lightscreen,auroraveil,tailwind,...}, opp{...}}
  megaUsed{me, opp}, observations[{attacker:'me'|'opp', who, target, move, pct}]

난수 결과(데미지·급소·명중)와 상대가 쓴 기술, 내 HP·상태이상은 사용자가 보이는 대로 고쳐 넣는다.
"""
from __future__ import annotations

from .belief import OppBelief
from .damage import Ctx, calc, ko_chance, percent_range
from .dex import Dex, to_id
from .mon import Mon
from . import features as F
from . import model as M
from .policies import HeuristicPolicy
from .protocol import legal_choices

SIDE_KEYS = ("reflect", "lightscreen", "auroraveil", "tailwind", "stealthrock", "spikes", "toxicspikes", "stickyweb")


# ======================================================================== 입력 → 스냅샷·요청
def to_snapshot(dex: Dex, st: dict):
    gt = st.get("gameType", "singles")
    team = st.get("myTeam") or []
    brought = st.get("brought") or list(range(len(team)))
    pick = 4 if gt == "doubles" else 3
    mine_in = {m.get("idx"): m for m in st.get("mine") or []}
    me_views, side_req = [], []
    order = list(brought) + [i for i in range(len(team)) if i not in brought]
    for idx in order:
        s = team[idx]
        sp = dex.species.get(s["species"])
        if not sp:
            continue
        cur = mine_in.get(idx, {})
        species = s["species"]
        if cur.get("mega") and s.get("item"):
            mg = dex.mega_for(s["species"], s["item"])
            species = mg or species
        name = sp["name"]
        hp_pct = float(cur.get("hpPct", 100))
        mon = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""),
                  nature=s.get("nature", "serious"), sp=s.get("sp") or {}, moves=s.get("moves") or []).setup(dex)
        if cur.get("mega"):
            mon.mega_evolve()
        mon.set_hp_pct(hp_pct)
        fainted = bool(cur.get("fainted")) or hp_pct <= 0
        in_battle = idx in brought[:pick]
        if not in_battle:
            continue
        me_views.append({
            "name": name, "species": species, "hpPct": 0.0 if fainted else hp_pct, "status": cur.get("status", ""),
            "fainted": fainted, "boosts": cur.get("boosts") or {}, "volatiles": cur.get("volatiles") or [],
            "slot": cur.get("slot") if not fainted else None, "moves": list(s.get("moves") or []),
            "item": cur.get("item", s.get("item", "")), "ability": s.get("ability"), "mega": bool(cur.get("mega")),
            "revealed": True, "lastMove": cur.get("lastMove", ""), "turnsOut": cur.get("turnsOut", 1),
            "timesAttacked": cur.get("timesAttacked", 0), "hp": mon.hp if not fainted else 0, "maxhp": mon.maxhp,
            "_idx": idx, "_inBattle": in_battle,
        })
        if in_battle:
            side_req.append({"ident": f"p1: {name}", "details": f"{dex.species[species]['name']}, L50",
                             "condition": "0 fnt" if fainted else f"{mon.hp}/{mon.maxhp}" + (f" {cur.get('status')}" if cur.get("status") else ""),
                             "active": bool(cur.get("slot")) and not fainted, "_idx": idx})
    opp_views = []
    for o in st.get("opp") or []:
        sid = dex.find("species", o.get("species"))
        if not sid:
            continue
        hp_pct = float(o.get("hpPct", 100))
        fainted = bool(o.get("fainted")) or hp_pct <= 0
        # 아직 나오지 않은 상대(프리뷰에만 있음)는 시점에 넣지 않는다 (실제 대전 기록과 같게)
        if not (o.get("seen") or o.get("slot") or fainted or o.get("moves")):
            continue
        opp_views.append({
            "name": dex.species[sid]["name"], "species": sid, "hpPct": 0.0 if fainted else hp_pct,
            "status": o.get("status", ""), "fainted": fainted, "boosts": o.get("boosts") or {},
            "volatiles": o.get("volatiles") or [], "slot": o.get("slot") if not fainted else None,
            "moves": [dex.find("moves", m) for m in o.get("moves") or [] if dex.find("moves", m)],
            "item": (dex.find("items", o["item"]) or "") if o.get("item") not in (None, "?") else None,
            "ability": dex.find("abilities", o.get("ability")) if o.get("ability") else None,
            "mega": bool(o.get("mega")), "revealed": True, "lastMove": o.get("lastMove", ""),
            "turnsOut": o.get("turnsOut", 1), "timesAttacked": o.get("timesAttacked", 0),
        })
    f = st.get("field") or {}
    sides = st.get("sides") or {}
    sc = {}
    for who, key in (("p1", "me"), ("p2", "opp")):
        d = sides.get(key) or {}
        sc[who] = {k: {"turns": 0, "layers": int(d[k]) if isinstance(d[k], (int, float)) and not isinstance(d[k], bool) else 1}
                   for k in SIDE_KEYS if d.get(k)}
    pseudo = {}
    if f.get("trickRoom"):
        pseudo["trickroom"] = 0
    if f.get("gravity"):
        pseudo["gravity"] = 0
    mu = st.get("megaUsed") or {}
    snap = {
        "me": "p1", "gameType": gt, "turn": int(st.get("turn") or 1),
        "weather": f.get("weather", ""), "weatherTurns": int(f.get("weatherTurns") or 0),
        "terrain": f.get("terrain", ""), "terrainTurns": int(f.get("terrainTurns") or 0),
        "pseudo": pseudo, "sideConds": sc,
        "megaUsed": {"p1": bool(mu.get("me")) or any(m["mega"] for m in me_views),
                     "p2": bool(mu.get("opp")) or any(o["mega"] for o in opp_views)},
        "fainted": {"p1": sum(1 for m in me_views if m["fainted"] and m["_inBattle"]),
                    "p2": sum(1 for o in opp_views if o["fainted"])},
        "preview": {"p1": [team[i]["species"] for i in range(len(team))],
                    "p2": [dex.find("species", s) for s in st.get("oppPreview") or [] if dex.find("species", s)]},
        "pickSize": pick, "mons": {"p1": me_views, "p2": opp_views}, "recent": [],
    }
    return snap, side_req


def make_request(dex: Dex, st: dict, snap: dict, side_req: list) -> dict:
    """앱 입력 → 쇼다운 요청 모양 (선택지 생성용)."""
    gt = snap["gameType"]
    team = st.get("myTeam") or []
    req = {"side": {"pokemon": [{k: v for k, v in p.items() if not k.startswith("_")} for p in side_req]}}
    actives = sorted([m for m in snap["mons"]["p1"] if m["slot"] and not m["fainted"] and m["_inBattle"]],
                     key=lambda m: m["slot"])
    # 쓰러져서 비어 있는 자리 → 강제 교체
    n_slots = 2 if gt == "doubles" else 1
    if len(actives) < n_slots and st.get("forceSwitch"):
        req["forceSwitch"] = [True] * (n_slots - len(actives)) + [False] * len(actives)
        return req
    act_req = []
    can_mega = not snap["megaUsed"]["p1"]
    for m in actives:
        s = team[m["_idx"]]
        mvs = []
        for mid in s.get("moves") or []:
            mv = dex.moves.get(mid)
            if not mv:
                continue
            disabled = mid in (st.get("disabledMoves") or {}).get(str(m["_idx"]), [])
            mvs.append({"id": mid, "target": mv.get("target"), "pp": 1, "disabled": disabled})
        mega_ok = can_mega and not m["mega"] and bool(dex.mega_for(s["species"], m.get("item") or s.get("item")))
        act_req.append({"moves": mvs, "canMegaEvo": mega_ok, "trapped": bool(st.get("trapped"))})
    req["active"] = act_req
    return req


# ======================================================================== 관측으로 상대 세트 좁히기
SPREAD_GRID = [
    ({"atk": 32, "spe": 32, "hp": 2}, ["jolly", "adamant"]),
    ({"spa": 32, "spe": 32, "hp": 2}, ["timid", "modest"]),
    ({"hp": 32, "atk": 32, "def": 2}, ["adamant", "careful", "impish"]),
    ({"hp": 32, "spa": 32, "spd": 2}, ["modest", "calm", "bold"]),
    ({"hp": 32, "def": 17, "spd": 17}, ["bold", "careful", "impish", "calm"]),
    ({"hp": 2, "def": 32, "spd": 32}, ["serious"]),
    ({}, ["serious"]),
]
OBS_ITEMS = ["", "lifeorb", "expertbelt", "muscleband", "wiseglasses", "TYPE", "RESIST", "focussash"]


def narrow_by_observations(dex: Dex, species: str, observations: list, my_mons: dict, field) -> dict:
    """관측된 데미지(%)와 맞는 SP·성격·도구 조합만 남긴다.
    observations: [{dir:'opp2me'|'me2opp', move, pct, my_idx}]
    돌려줌: {candidates: [...], best: {...}, total, consistent}
    """
    from .teamgen import TYPE_ITEM, RESIST_BERRY
    sp = dex.species[species]
    cands = []
    for spread, natures in SPREAD_GRID:
        for nat in natures:
            for it in OBS_ITEMS:
                cands.append((spread, nat, it))
    keep = []
    for spread, nat, it in cands:
        ok = True
        for ob in observations:
            mv = dex.find("moves", ob["move"])
            if not mv or dex.moves[mv]["category"] == "Status":
                continue
            mine = my_mons.get(ob.get("my_idx"))
            if mine is None:
                continue
            item = it
            if it == "TYPE":
                item = TYPE_ITEM.get(dex.moves[mv]["type"], "")
            opp = Mon(species=species, ability=ob.get("oppAbility") or to_id(sp["abilities"].get("0", "")),
                      item=item if it != "RESIST" else "", nature=nat, sp=dict(spread), moves=[mv]).setup(dex)
            if ob.get("oppHpPctBefore") is not None:
                opp.set_hp_pct(ob["oppHpPctBefore"])
            if ob["dir"] == "opp2me":
                r = calc(dex, opp, mine, mv, field, Ctx(atk_side=1, def_side=0))
                tgt_max = mine.maxhp
                tol = 0.6  # 내 HP 는 정확
            else:
                if it == "RESIST":
                    t = dex.moves[mv]["type"]
                    opp.item = RESIST_BERRY.get(t, "")
                r = calc(dex, mine, opp, mv, field, Ctx(atk_side=0, def_side=1))
                tgt_max = opp.maxhp
                tol = 2.2  # 상대 HP 는 48칸 체력바 → 약 ±2%
            if r.immune or not r.rolls:
                ok = ob["pct"] <= 0.5
            else:
                lo = 100 * min(r.rolls[0]) / tgt_max
                hi = 100 * max(r.crit_rolls[0] if ob.get("crit") else r.rolls[0]) / tgt_max
                if ob.get("crit"):
                    lo = 100 * min(r.crit_rolls[0]) / tgt_max
                ok = lo - tol <= ob["pct"] <= hi + tol
            if not ok:
                break
        if ok:
            keep.append({"sp": spread, "nature": nat, "item": it})
    return {"total": len(cands), "consistent": len(keep), "candidates": keep[:40],
            "best": keep[0] if keep else None}


# ======================================================================== 추천
def advise(dex: Dex, st: dict) -> dict:
    snap, side_req = to_snapshot(dex, st)
    req = make_request(dex, st, snap, side_req)
    gt = snap["gameType"]
    opts = legal_choices(req)
    team = st.get("myTeam") or []
    calc_pol = HeuristicPolicy(dex, temperature=0.12)
    calc_pol.start("p1", team, dex)
    x = F.FCtx(dex, snap, _compact(req), team, calc_pol)

    # 관측으로 상대 세트 좁히기 → 계산에 반영
    narrowed = {}
    my_mons_by_idx = {}
    for m in snap["mons"]["p1"]:
        mm = x.mon(m, True)
        if mm is not None:
            my_mons_by_idx[m["_idx"]] = mm
    for o in snap["mons"]["p2"]:
        obs = [ob for ob in st.get("observations") or [] if dex.find("species", ob.get("oppSpecies")) == o["species"]]
        if obs:
            res = narrow_by_observations(dex, o["species"], obs, my_mons_by_idx, x.field)
            narrowed[o["species"]] = res
            if res["best"]:
                b = calc_pol._belief(o["species"])
                b.forced = res["best"]
    _apply_forced_beliefs(calc_pol)

    model = M.load(gt)
    q = v = None
    source = "선생님 정책 (학습 모델 없음)"
    if model and opts:
        try:
            q, v = model.evaluate(x, [o["choice"] for o in opts])
            source = f"학습 모델 ({model.info.get('games')}판 학습)"
        except Exception as e:
            source = f"선생님 정책 (모델 오류: {e})"
    heur = calc_pol._turn_scores(snap, req, opts) if opts and not req.get("teamPreview") else [0] * len(opts)
    ranked = []
    for i, o in enumerate(opts):
        ranked.append({"choice": o["choice"], "label": describe_choice(dex, o["choice"], req, snap, team),
                       "win": float(q[i]) if q is not None else None, "score": round(float(heur[i]), 4),
                       "why": _why(x, o["choice"], req, snap)})
    if q is not None:
        ranked.sort(key=lambda r: -(r["win"] or 0))
    else:
        ranked.sort(key=lambda r: -r["score"])
    return {
        "source": source, "winNow": v, "options": ranked[:40], "nOptions": len(ranked),
        "damage": damage_table(dex, x, snap, team), "threats": threat_table(dex, x, snap),
        "speed": speed_order(dex, x, snap), "narrowed": narrowed,
        "oppSets": {o["species"]: _belief_summary(dex, calc_pol, o) for o in snap["mons"]["p2"]},
    }


def _apply_forced_beliefs(pol: HeuristicPolicy):
    """관측으로 좁힌 SP·성격·도구를 상대 가정 포켓몬에 적용하도록 assumed_mon 을 감싼다."""
    for b in pol.beliefs.values():
        forced = getattr(b, "forced", None)
        if not forced or getattr(b, "_wrapped", False):
            continue
        orig = b.assumed_mon

        def wrapped(hp_pct=100.0, status="", boosts=None, item=None, mega=False, _b=b, _o=orig, _f=forced):
            m = _o(hp_pct=hp_pct, status=status, boosts=boosts, item=item, mega=mega)
            m.sp = dict(_f["sp"])
            m.nature = _f["nature"]
            if item is None and _f["item"] not in ("TYPE", "RESIST"):
                m.item = _f["item"]
            m.recalc()
            m.set_hp_pct(hp_pct)
            return m
        b.assumed_mon = wrapped
        b._wrapped = True


def _compact(req):
    from .selfplay import compact_req
    r = dict(req)
    r.setdefault("rqid", 0)
    return compact_req(r)


def describe_choice(dex, choice, req, snap, team):
    """'move 1 2, switch 4' → '한카리아스: 지진 → 리자몽 / 교체: 팬텀'"""
    if choice.startswith("team"):
        order = [int(c) for c in choice.split()[1]]
        names = []
        for k in order:
            p = req["side"]["pokemon"][k - 1]
            names.append(dex.species[dex.find("species", p["ident"].split(": ", 1)[1])]["ko"])
        return "선출: " + " · ".join(names)
    parts = [p.strip() for p in choice.split(",")]
    actives = sorted([m for m in snap["mons"]["p1"] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
    opps = sorted([m for m in snap["mons"]["p2"] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
    out = []
    for i, p in enumerate(parts):
        toks = p.split()
        who = dex.species[actives[i]["species"]]["ko"] if i < len(actives) else ""
        if toks[0] == "pass":
            continue
        if toks[0] == "switch":
            k = int(toks[1])
            name = req["side"]["pokemon"][k - 1]["ident"].split(": ", 1)[1]
            sid = dex.find("species", name)
            out.append(f"{who} → 교체: {dex.species[sid]['ko'] if sid else name}")
            continue
        j = int(toks[1])
        mid = req["active"][i]["moves"][j - 1]["id"]
        s = f"{who}: {dex.moves[mid]['ko']}"
        if len(toks) >= 3 and toks[2].lstrip("-").isdigit():
            t = int(toks[2])
            if t > 0 and t - 1 < len(opps):
                s += f" → {dex.species[opps[t - 1]['species']]['ko']}"
            elif t < 0:
                s += " → 아군"
        if toks[-1] == "mega":
            s = "메가진화 + " + s
        out.append(s)
    return " / ".join(out)


def _why(x: F.FCtx, choice: str, req, snap):
    if choice.startswith("team"):
        return ""
    reasons = []
    for i, p in enumerate([q.strip() for q in choice.split(",")]):
        part = F._parse_part(p)
        if part["kind"] == "move":
            eff = F.p_effect(x, i, part)
            if eff[4] == 1.0 and eff[1] == 0:
                reasons.append("효과 없음")
            elif eff[2] >= 0.99:
                reasons.append("확정 처치")
            elif eff[2] > 0:
                reasons.append(f"처치 {eff[2]:.0%}")
            elif eff[1] > 0:
                reasons.append(f"기대 데미지 {eff[1]:.0%}")
            risk = F.p_risk(x, i, part)
            if risk[0] and risk[1] > 0.5:
                reasons.append(f"먼저 맞고 쓰러질 위험 {risk[1]:.0%}")
        elif part["kind"] == "switch":
            sw = F.p_switch(x, i, part)
            reasons.append(f"교체 시 받는 데미지 약 {sw[0]:.0%}")
    return " · ".join(reasons)


def damage_table(dex, x: F.FCtx, snap, team):
    rows = []
    actives = sorted([m for m in snap["mons"]["p1"] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
    opps = sorted([m for m in snap["mons"]["p2"] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
    for m in actives:
        me = x.mon(m, True)
        if me is None:
            continue
        for mid in me.moves:
            mv = dex.moves.get(mid)
            if not mv or mv["category"] == "Status":
                continue
            for o in opps:
                om = x.mon(o, False)
                if om is None:
                    continue
                r = calc(dex, me, om, mid, x.field, Ctx(atk_side=0, def_side=1, actives=[me, om],
                                                          spread=x.doubles and mv.get("target") in ("allAdjacentFoes", "allAdjacent") and len(opps) > 1))
                lo, hi = percent_range(r, om)
                rows.append({"from": dex.species[m["species"]]["ko"], "to": dex.species[o["species"]]["ko"],
                             "move": mv["ko"], "type": r.move_type, "min": round(lo, 1), "max": round(hi, 1),
                             "ko1": round(ko_chance(r, om.hp, 1), 3), "ko2": round(ko_chance(r, om.hp, 2), 3),
                             "immune": r.immune or r.blocked, "notes": r.notes})
    return rows


def threat_table(dex, x: F.FCtx, snap):
    rows = []
    actives = sorted([m for m in snap["mons"]["p1"] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
    opps = sorted([m for m in snap["mons"]["p2"] if m["slot"] and not m["fainted"]], key=lambda m: m["slot"])
    for o in opps:
        om = x.mon(o, False)
        b = x.c._belief(o["species"])
        probs = b.move_probs()
        for mid, p in list(probs.items())[:8]:
            mv = dex.moves.get(mid)
            if not mv or mv["category"] == "Status":
                continue
            for m in actives:
                me = x.mon(m, True)
                if me is None or om is None:
                    continue
                r = calc(dex, om, me, mid, x.field, Ctx(atk_side=1, def_side=0, actives=[me, om]))
                lo, hi = percent_range(r, me)
                rows.append({"from": dex.species[o["species"]]["ko"], "to": dex.species[m["species"]]["ko"],
                             "move": mv["ko"], "prob": round(p, 2), "revealed": mid in (o.get("moves") or []),
                             "min": round(lo, 1), "max": round(hi, 1), "ko1": round(ko_chance(r, me.hp, 1), 3),
                             "immune": r.immune or r.blocked})
    rows.sort(key=lambda r: (-r["ko1"], -r["max"]))
    return rows[:24]


def speed_order(dex, x: F.FCtx, snap):
    out = []
    for side, mine in (("p1", True), ("p2", False)):
        for m in snap["mons"][side]:
            if not m["slot"] or m["fainted"]:
                continue
            mm = x.mon(m, mine)
            if mm is None:
                continue
            s = x.speed(mm, 0 if mine else 1)
            out.append({"who": dex.species[m["species"]]["ko"], "mine": mine, "speed": abs(s),
                        "estimated": not mine})
    tr = "trickroom" in snap["pseudo"]
    out.sort(key=lambda r: (r["speed"] if tr else -r["speed"]))
    return {"trickRoom": tr, "order": out}


def _belief_summary(dex, pol, o):
    b = pol._belief(o["species"])
    b.observe(o.get("moves") or [], o.get("item"), o.get("ability"))
    return {"role": b.likely_role(), "ability": dex.abilities.get(b.likely_ability(), {}).get("ko"),
            "moves": [{"move": dex.moves[m]["ko"], "id": m, "prob": round(p, 2)} for m, p in list(b.move_probs().items())[:10]
                      if m in dex.moves]}
