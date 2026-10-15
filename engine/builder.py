# -*- coding: utf-8 -*-
"""대전 준비 추천 — 선출 · 육성(SP·성격) · 기술 배치 · 도구.

메타(자주 만나는 상대)는 쇼다운 챔피언스 세트 목록의 티어로 가중하고, 사용자가 실제로 만난 상대
(userdata/battles 기록)가 쌓이면 그 빈도로 바꾼다.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from .belief import OppBelief
from .damage import Ctx, calc, ko_chance
from .dex import Dex, to_id
from .mon import FieldState, Mon
from .policies import HeuristicPolicy
from .teamgen import RESIST_BERRY, TYPE_ITEM

ROOT = Path(__file__).resolve().parent.parent
TIER_W = {"Uber": 3.0, "OU": 3.0, "(OU)": 2.0, "UUBL": 2.0, "UU": 1.5, "RU": 1.0, "NFE": 0.3}
NATURES_FOR = {
    "atk": ["adamant", "jolly"], "spa": ["modest", "timid"],
}


def mon_from_set(dex: Dex, s: dict) -> Mon:
    m = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""),
            nature=s.get("nature", "serious"), sp=dict(s.get("sp") or {}), moves=list(s.get("moves") or [])).setup(dex)
    if dex.mega_for(m.species, m.item):
        m.mega_evolve()
    return m


def meta_threats(dex: Dex, n=40) -> list:
    """자주 만날 상대 목록 [(종족, 가중치)]."""
    seen = Counter()
    bdir = ROOT / "userdata" / "battles"
    if bdir.is_dir():
        for f in bdir.glob("*.json"):
            try:
                b = json.loads(f.read_text(encoding="utf-8"))
                for s in b.get("oppPreview") or []:
                    sid = dex.find("species", s)
                    if sid:
                        seen[sid] += 1
            except Exception:
                pass
    out = []
    for sid in dex.sets:
        if sid not in dex.species:
            continue
        w = TIER_W.get(dex.species[sid].get("tier", ""), 0.5) + 2.0 * seen.get(sid, 0)
        out.append((sid, w))
    out.sort(key=lambda t: -t[1])
    return out[:n]


# ======================================================================== 선출
def recommend_bring(dex: Dex, team: list, opp_preview: list, game_type="singles", top=5) -> dict:
    pol = HeuristicPolicy(dex)
    field = FieldState(game_type=game_type)
    opps = [dex.find("species", s) for s in opp_preview if dex.find("species", s)]
    opp_mons = [OppBelief(dex, s).assumed_mon() for s in opps]
    mine = [mon_from_set(dex, s) for s in team]
    # 1:1 상성표
    grid = []
    for m in mine:
        row = []
        for o in opp_mons:
            g = pol.best_vs(m, o, field, 0)
            t = pol.best_vs(o, m, field, 1)
            faster = pol.speed(m, field, 0) > pol.speed(o, field, 1)
            sc = (g[0] + g[1]) - (t[0] + t[1]) + (0.15 if faster else -0.15)
            row.append({"score": round(sc, 3), "give": round(g[0], 2), "giveKo": round(g[1], 2),
                        "take": round(t[0], 2), "takeKo": round(t[1], 2), "faster": faster,
                        "move": dex.moves.get(g[2], {}).get("ko", ""), "oppMove": dex.moves.get(t[2], {}).get("ko", "")})
        grid.append(row)
    k = 4 if game_type == "doubles" else 3
    lead_n = 2 if game_type == "doubles" else 1
    combos = []
    for chosen in itertools.combinations(range(len(mine)), k):
        # 상대 각 포켓몬에 대해 내 선출 중 가장 잘 받는 포켓몬 점수 (커버 범위)
        cover = [max(grid[i][j]["score"] for i in chosen) for j in range(len(opp_mons))]
        avg = sum(grid[i][j]["score"] for i in chosen for j in range(len(opp_mons))) / (k * max(1, len(opp_mons)))
        worst = min(cover) if cover else 0
        megas = sum(1 for i in chosen if dex.mega_for(team[i]["species"], team[i].get("item", "")))
        mega_pen = 0.3 * max(0, megas - 1)  # 메가는 한 판에 한 번
        for leads in itertools.combinations(chosen, lead_n):
            lead_sc = sum(sum(grid[i][j]["score"] for j in range(len(opp_mons))) / max(1, len(opp_mons)) for i in leads) / lead_n
            total = 0.45 * sum(cover) / max(1, len(cover)) + 0.25 * avg + 0.2 * worst + 0.25 * lead_sc - mega_pen
            order = list(leads) + [i for i in chosen if i not in leads]
            combos.append({"order": order, "score": round(total, 4), "cover": [round(c, 2) for c in cover],
                           "worst": round(worst, 3), "lead": round(lead_sc, 3)})
    combos.sort(key=lambda c: -c["score"])
    for c in combos[:top]:
        c["names"] = [dex.species[team[i]["species"]]["ko"] for i in c["order"]]
        weak = [dex.species[opps[j]]["ko"] for j, v in enumerate(c["cover"]) if v < 0]
        c["why"] = ("선봉 " + " · ".join(c["names"][:lead_n]) + " 의 상성이 좋음") + \
                   (f" / 주의: {', '.join(weak)} 상대로 불리" if weak else " / 상대 6마리 모두 대응 가능")
    return {"combos": combos[:top], "grid": grid,
            "mine": [dex.species[s["species"]]["ko"] for s in team], "opp": [dex.species[s]["ko"] for s in opps]}


# ======================================================================== 육성 (SP · 성격)
def speed_benchmarks(dex: Dex, me: dict, threats=None) -> list:
    """주요 상대의 예상 최대 스피드와, 그걸 넘기 위한 내 최소 스피드 SP."""
    sp_me = dex.species[mon_from_set(dex, me).species]
    out = []
    for sid, _ in (threats or meta_threats(dex, 25)):
        bs = dex.species[sid]["baseStats"]["spe"]
        their = int((bs + 32 + 20) * 1.1)   # 최속(스피드 32, +성격)
        their_neutral = bs + 32 + 20
        need = None
        for plus in (False, True):
            for spv in range(0, 33):
                v = sp_me["baseStats"]["spe"] + spv + 20
                v = int(v * 1.1) if plus else v
                if v > their:
                    need = (spv, plus)
                    break
            if need:
                break
        out.append({"species": dex.species[sid]["ko"], "theirMax": their, "theirNeutral": their_neutral,
                    "needSp": need[0] if need else None, "needPlusNature": need[1] if need else None})
    return out


def recommend_spread(dex: Dex, me: dict, threats=None, top=3, goal="auto") -> dict:
    """SP 66 을 나누는 몇 가지 안을 계산해 비교한다.
    점수 = 메타 상대에게 주는 데미지 + 받는 공격을 버티는 정도 + 스피드 추월 수.
    """
    threats = threats or meta_threats(dex, 30)
    base = mon_from_set(dex, me)
    phys = sum(1 for m in base.moves if dex.moves.get(m, {}).get("category") == "Physical") >= \
        sum(1 for m in base.moves if dex.moves.get(m, {}).get("category") == "Special")
    a = "atk" if phys else "spa"
    n_dmg = sum(1 for m in base.moves if dex.moves.get(m, {}).get("category") not in (None, "Status"))
    if goal == "auto":
        goal = "offense" if n_dmg >= 3 else ("balanced" if n_dmg == 2 else "defense")
    templates = []
    for spe in (0, 16, 24, 32):
        for off in (0, 16, 32):
            rest = 66 - spe - off
            for hp in (0, 16, 32):
                if hp > rest:
                    continue
                r2 = rest - hp
                for dsplit in ((r2, 0), (r2 // 2, r2 - r2 // 2), (0, r2)):
                    d_, s_ = min(32, dsplit[0]), min(32, dsplit[1])
                    sp = {"spe": spe, a: off, "hp": hp, "def": d_, "spd": s_}
                    if sum(sp.values()) <= 66 and all(v <= 32 for v in sp.values()):
                        templates.append({k: v for k, v in sp.items() if v})
    natures = list(NATURES_FOR[a])
    if goal != "offense":
        natures += ["bold" if not phys else "impish", "calm" if not phys else "careful"]
    field = FieldState()
    opp_mons = [(OppBelief(dex, sid).assumed_mon(), w) for sid, w in threats[:14]]
    pol = HeuristicPolicy(dex)
    results = []
    seen = set()
    for sp in templates:
        for nat in natures:
            key = (tuple(sorted(sp.items())), nat)
            if key in seen:
                continue
            seen.add(key)
            m = Mon(species=base.species, ability=base.ability, item=base.item, nature=nat, sp=sp,
                    moves=base.moves).setup(dex)
            if base.mega:
                m.species = base.species
                m.recalc()
                m.mega = True
            give = take = outsp = wsum = 0.0
            for o, w in opp_mons:
                g = pol.best_vs(m, o, field, 0)
                t = pol.best_vs(o, m, field, 1)
                give += w * (g[0] + g[1])
                take += w * (t[0] + t[1])
                outsp += w * (pol.speed(m, field, 0) > pol.speed(o, field, 1))
                wsum += w
            give, take, outsp = give / wsum, take / wsum, outsp / wsum
            score = {"balanced": give - take + 0.6 * outsp, "offense": 1.5 * give - 0.5 * take + 0.8 * outsp,
                     "defense": 0.5 * give - 1.5 * take + 0.3 * outsp}[goal]
            results.append({"sp": sp, "nature": nat, "score": round(score, 4), "give": round(give, 3),
                            "take": round(take, 3), "outspeed": round(outsp, 3), "stats": m.stats})
    results.sort(key=lambda r: -r["score"])
    for r in results[:top]:
        r["natureKo"] = dex.natures[r["nature"]]["ko"]
        r["text"] = " / ".join(f"{v} {k.upper()}" for k, v in r["sp"].items())
    cur = next((r for r in results if r["sp"] == {k: v for k, v in (me.get("sp") or {}).items() if v}
                and r["nature"] == me.get("nature")), None)
    return {"options": results[:top], "goal": goal, "current": cur,
            "speed": speed_benchmarks(dex, me, threats[:15]), "considered": len(results)}


# ======================================================================== 기술 배치
def recommend_moves(dex: Dex, me: dict, threats=None, top=5) -> dict:
    threats = threats or meta_threats(dex, 30)
    base = mon_from_set(dex, me)
    sid = base.species
    sp = dex.species[sid]
    learn = dex.learnsets.get(to_id(me["species"])) or dex.learnsets.get(sid) or []
    field = FieldState()
    opp_mons = [(OppBelief(dex, s).assumed_mon(), w) for s, w in threats[:20]]
    pol = HeuristicPolicy(dex)
    # 기술 하나하나의 메타 상대 기대값
    single = {}
    for mid in learn:
        mv = dex.moves.get(mid)
        if not mv:
            continue
        if mv["category"] == "Status":
            continue
        vals = []
        for o, w in opp_mons:
            f, ko = pol.exp_frac(base, o, mid, field, 0)
            vals.append((f + 0.7 * ko) * w)
        single[mid] = vals
    cand = sorted(single, key=lambda m: -sum(single[m]))[:14]
    status_pool = [m for m in learn if dex.moves.get(m, {}).get("category") == "Status" and m in
                   {"protect", "swordsdance", "nastyplot", "dragondance", "calmmind", "willowisp", "thunderwave",
                    "toxic", "recover", "roost", "slackoff", "stealthrock", "tailwind", "trickroom", "taunt",
                    "followme", "ragepowder", "fakeout", "spore", "uturn", "voltswitch", "bulkup", "quiverdance",
                    "shellsmash", "substitute", "encore", "helpinghand", "wideguard", "reflect", "lightscreen"}]
    combos = []
    for n_status in (0, 1, 2):
        for dmg in itertools.combinations(cand, 4 - n_status):
            # 커버: 상대마다 가장 좋은 기술 값의 평균
            cov = sum(max(single[m][i] for m in dmg) for i in range(len(opp_mons)))
            types = {dex.moves[m]["type"] for m in dmg}
            stab = any(t in sp["types"] for t in types)
            if not stab:
                continue
            dup = len(dmg) - len(types)
            for st in itertools.combinations(status_pool, n_status) if n_status else [()]:
                score = cov - 0.15 * dup + sum(_status_bonus(m) for m in st)
                combos.append({"moves": list(dmg) + list(st), "score": round(score, 3)})
        if len(combos) > 3000:
            break
    combos.sort(key=lambda c: -c["score"])
    out = []
    seen = set()
    for c in combos:
        key = tuple(sorted(c["moves"]))
        if key in seen:
            continue
        seen.add(key)
        c["names"] = [dex.moves[m]["ko"] for m in c["moves"]]
        out.append(c)
        if len(out) >= top:
            break
    singles = [{"move": dex.moves[m]["ko"], "id": m, "type": dex.moves[m]["type"],
                "value": round(sum(single[m]) / max(1, sum(w for _, w in opp_mons)), 3)} for m in cand]
    return {"combos": out, "singles": singles}


def _status_bonus(m):
    return {"protect": 0.35, "swordsdance": 0.3, "nastyplot": 0.3, "dragondance": 0.35, "calmmind": 0.25,
            "willowisp": 0.25, "thunderwave": 0.2, "recover": 0.25, "roost": 0.25, "slackoff": 0.25,
            "tailwind": 0.3, "trickroom": 0.25, "fakeout": 0.3, "spore": 0.4, "uturn": 0.2, "voltswitch": 0.2,
            "quiverdance": 0.4, "shellsmash": 0.35, "taunt": 0.15, "followme": 0.25, "ragepowder": 0.25,
            "helpinghand": 0.15, "substitute": 0.15, "encore": 0.15, "toxic": 0.15}.get(m, 0.1)


# ======================================================================== 도구
def recommend_items(dex: Dex, me: dict, used_items=(), threats=None, top=6) -> dict:
    threats = threats or meta_threats(dex, 25)
    field = FieldState()
    opp_mons = [(OppBelief(dex, s).assumed_mon(), w) for s, w in threats[:16]]
    pol = HeuristicPolicy(dex)
    base = mon_from_set(dex, me)
    sp = dex.species[base.species]
    mega_item = dex.mega_for(to_id(me["species"]), me.get("item", ""))
    cands = ["lifeorb", "expertbelt", "choicescarf", "focussash", "sitrusberry", "leftovers", "lumberry",
             "muscleband", "wiseglasses", "scopelens"]
    for t in sp["types"]:
        if TYPE_ITEM.get(t):
            cands.append(TYPE_ITEM[t])
    weak = [t for t in dex.typechart if t not in ("Stellar", "???") and
            sum(dex.effectiveness(t, d) for d in sp["types"]) >= 1 and not any(dex.immune(t, d) for d in sp["types"])]
    for t in weak:
        if RESIST_BERRY.get(t):
            cands.append(RESIST_BERRY[t])
    # 메가스톤
    for it in dex.items.values():
        if it.get("megaStone") and sp["name"].split("-Mega")[0] in it["megaStone"]:
            cands.append(it["id"])
    rows = []
    for it in dict.fromkeys(cands):
        if it not in dex.items:
            continue
        s2 = dict(me)
        s2["item"] = it
        m = mon_from_set(dex, s2)
        give = take = outsp = wsum = 0.0
        for o, w in opp_mons:
            g = pol.best_vs(m, o, field, 0)
            t = pol.best_vs(o, m, field, 1)
            survive_sash = it == "focussash" and t[1] > 0.5
            give += w * (g[0] + g[1])
            take += w * ((t[0] + t[1]) * (0.5 if survive_sash else 1.0))
            outsp += w * (pol.speed(m, field, 0) > pol.speed(o, field, 1))
            wsum += w
        heal = {"sitrusberry": 0.12, "leftovers": 0.1, "lumberry": 0.06}.get(it, 0)
        score = (give - take) / wsum + 0.5 * outsp / wsum + heal
        rows.append({"item": it, "ko": dex.items[it]["ko"], "score": round(score, 4), "give": round(give / wsum, 3),
                     "take": round(take / wsum, 3), "outspeed": round(outsp / wsum, 3),
                     "used": it in used_items, "mega": bool(dex.items[it].get("megaStone"))})
    rows.sort(key=lambda r: (r["used"], -r["score"]))
    return {"items": rows[:top + 4], "currentMega": mega_item}
