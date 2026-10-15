# -*- coding: utf-8 -*-
"""쇼다운 리플레이(관전자 로그) → 자기대전과 같은 형태의 판 기록 (사람 기보 모방학습용).

리플레이에는 '요청'(그 플레이어가 고를 수 있던 선택지)이 없다. 그래서 양쪽 플레이어 시점을 다시 만든다.

  1. 별명 → 종 이름으로 통일 (학습 변수가 이름으로 세트를 찾기 때문)
  2. 판 전체에서 드러난 기술·도구·특성을 모은다 (그 플레이어는 자기 세트를 다 알고 있으므로
     '나중에 드러난 것'도 그 시점에 아는 정보다). |showteam| 이 있으면 그것이 정답.
     끝까지 안 드러난 칸은 메타 통계로 가장 그럴듯한 값을 채운다.
  3. 결정 시점마다 그 플레이어의 요청을 만들어 선택지를 뽑고, 다음 줄들에서 실제로 한 행동을 찾는다.
     - 선출: 선봉 + 나온 포켓몬들과 맞는 선택지 전부 (안 나온 포켓몬은 모르므로 고르게 나눔)
     - 턴: 처음 쓴 기술(대상·메가 포함) 또는 교체. 못 움직였거나(마비·풀죽음) 먼저 쓰러지면 모름
     - 교체(쓰러짐·유턴 등): 들어온 포켓몬
  4. 정답이 하나로 안 정해지면 맞는 선택지들에 고르게 나눈 분포를 정답으로 둔다 (정보가 없는 결정은 버림).

해석기의 정확도는 우리 시뮬레이터 대전(정답을 아는 판)의 관전자 로그로 검사한다 (check_with_selfplay).
"""
from __future__ import annotations

import copy
import json
import re

from .belief import ROLE_SPREADS, OppBelief
from .dex import Dex, to_id
from .mon import Mon
from .protocol import BattleView, legal_choices, parse_hp, species_from_details, split_ident
from .selfplay import compact_req
from .teamgen import normalize_set

ID_RE = re.compile(r"\b(p[12])([ab]?): ([^|\n]+)")
SKIP_CMDS = {"c", "c:", "chat", "j", "J", "l", "L", "n", "N", "raw", "html", "uhtml", "uhtmlchange", "inactive",
             "inactiveoff", "join", "leave", "name", "chatmsg", "chatmsg-raw", "bigerror", "message", "debug", "seed"}
LOCKED_FROM = ("lockedmove",)
TRICK_FROM = ("move: Trick", "move: Switcheroo", "move: Bestow", "ability: Magician", "ability: Pickpocket",
              "move: Thief", "move: Covet")


def _cmd(line):
    if not line.startswith("|"):
        return None, []
    p = line.split("|")[1:]
    return (p[0] if p else None), p[1:]


def _tags(args):
    out = {}
    for a in args:
        m = re.match(r"^\[(\w+)\]\s*(.*)$", a)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def unpack_team(dex: Dex, packed: str) -> list:
    """쇼다운 packed 팀 텍스트 → 세트 목록 (|showteam| 용)."""
    out = []
    for buf in packed.split("]"):
        f = buf.split("|")
        if len(f) < 6:
            continue
        name, species = f[0], f[1] or f[0]
        sid = dex.find("species", species) or to_id(species)
        if sid not in dex.species:
            continue
        evs = (f[6] if len(f) > 6 else "").split(",")
        sp = {}
        for k, v in zip(("hp", "atk", "def", "spa", "spd", "spe"), evs):
            if v.strip().isdigit() and int(v):
                sp[k] = int(v)
        moves = [dex.find("moves", m) or to_id(m) for m in f[4].split(",") if m]
        out.append(normalize_set(dex, {
            "species": sid, "ability": dex.find("abilities", f[3]) or to_id(f[3]),
            "item": dex.find("items", f[2]) or to_id(f[2]), "nature": dex.find("natures", f[5]) or "serious",
            "sp": sp, "moves": [m for m in moves if m in dex.moves][:4], "gender": f[7] if len(f) > 7 else "",
        }))
    return out


class _Hindsight(BattleView):
    """판 전체를 관전자로 읽으며 양쪽의 원래 도구를 기억하는 시점."""

    def __init__(self):
        super().__init__("p3")
        self.first_item = {}      # (side, name) → 처음 도구
        self.tricked = set()

    def _note_item(self, m, item, how=""):
        if not m or not item:
            return
        k = (m.side, m.name)
        if any(how.startswith(t) for t in TRICK_FROM):
            self.tricked.add(k)
            return
        if k not in self.first_item and k not in self.tricked:
            self.first_item[k] = to_id(item)

    def _c__item(self, a, t):
        m = self.by_ident(a[0])
        self._note_item(m, a[1] if len(a) > 1 else "", t.get("from", ""))
        super()._c__item(a, t)

    def _c__enditem(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1 and not (m.side, m.name) in self.tricked:
            self._note_item(m, a[1])
        super()._c__enditem(a, t)

    def _c__mega(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 2:
            self._note_item(m, a[2])
        super()._c__mega(a, t)

    def _reveal_from_tags(self, args, tagd):
        src = tagd.get("from", "")
        if src.startswith("item:"):
            holder = self.by_ident(tagd["of"]) if tagd.get("of") else (self.by_ident(args[0]) if args and args[0].startswith("p") else None)
            self._note_item(holder, src.split(":", 1)[1].strip())
        super()._reveal_from_tags(args, tagd)


class ReplayParser:
    def __init__(self, dex: Dex):
        self.dex = dex
        self.stats = {"decisions": 0, "dropped_unknown": 0, "dropped_forced": 0, "dropped_mismatch": 0}

    # ------------------------------------------------------------ 준비
    def _canon_lines(self, lines):
        """별명 → 종 이름 (p1a: 별명 → p1a: Garchomp)."""
        dex = self.dex
        nick = {}
        for ln in lines:
            c, a = _cmd(ln)
            if c in ("switch", "drag", "replace") and len(a) >= 2:
                side, _, name = split_ident(a[0])
                sid = dex.find("species", species_from_details(a[1])) or to_id(species_from_details(a[1]))
                if sid in dex.species:
                    nick[(side, name)] = dex.ident_name(sid)

        def sub(m):
            return f"{m.group(1)}{m.group(2)}: {nick.get((m.group(1), m.group(3).strip()), m.group(3).strip())}"
        out = []
        for ln in lines:
            c, _ = _cmd(ln)
            if c is None or c in SKIP_CMDS:
                continue
            out.append(ID_RE.sub(sub, ln) if (": " in ln) else ln)
        return out

    def _fill_set(self, sid, side, name, hs: _Hindsight, game_type, used_items):
        dex = self.dex
        mv = hs.mons[side].get(name) if name else None
        moves = [m for m in (mv.moves if mv else []) if m in dex.moves][:4]
        item = hs.first_item.get((side, name)) if name else None
        ability = (mv.ability if mv else None) or None
        mega_target = ""
        if mv and mv.mega and dex.species.get(mv.species, {}).get("isMega"):
            mega_target = mv.species
        base = dex.base_of_mega(sid)
        b = OppBelief(dex, base, game_type)
        b.observe(moves, item, ability)
        probs = b.move_probs()
        for m in probs:
            if len(moves) >= 4:
                break
            if m not in moves and m in dex.moves:
                moves.append(m)
        if item is None:
            ip = {i: p for i, p in b.item_probs().items() if i in dex.items and i not in used_items}
            if mega_target:
                item = dex.find("items", dex.species[mega_target].get("requiredItem") or "") or ""
            elif ip:
                item = max(ip, key=ip.get)
                # 메가스톤인데 끝까지 메가진화를 안 했다면 다른 도구일 가능성이 크다
                if dex.items[item].get("megaStone"):
                    rest = {i: p for i, p in ip.items() if not dex.items[i].get("megaStone")}
                    item = max(rest, key=rest.get) if rest else item
            else:
                item = ""
        sps = b.spread_probs()
        if sps:
            spread, nature = dict(sps[0]["sp"]), sps[0]["nature"] or "serious"
        else:
            bs = dex.species[base]["baseStats"]
            spread, nature = ROLE_SPREADS["fast" if bs["spe"] >= 90 else "bulky"][0 if bs["atk"] >= bs["spa"] else 1]
            spread = dict(spread)
        s = {"species": base, "ability": ability or b.likely_ability(), "item": item or "", "nature": nature,
             "sp": spread, "moves": moves, "gender": "", "role": "replay"}
        if mega_target and mega_target != base:
            s["megaTarget"] = mega_target
        return normalize_set(dex, s)

    # ------------------------------------------------------------ 본체
    def parse(self, text_or_lines, meta: dict | None = None) -> dict | None:
        dex = self.dex
        meta = meta or {}
        if isinstance(text_or_lines, str):
            t = text_or_lines.strip()
            if t.startswith("{"):
                d = json.loads(t)
                meta = {**d, **meta}
                t = d.get("log", "")
            lines = t.split("\n")
        else:
            lines = list(text_or_lines)
        lines = self._canon_lines(lines)

        players, winner_name, game_type, pick = {}, None, "singles", None
        preview = {"p1": [], "p2": []}
        showteam = {}
        for ln in lines:
            c, a = _cmd(ln)
            if c == "player" and len(a) >= 2 and a[1]:
                players[a[0]] = a[1]
            elif c == "win":
                winner_name = a[0] if a else None
            elif c == "gametype" and a:
                game_type = a[0]
            elif c == "teampreview" and a and a[0].isdigit():
                pick = int(a[0])
            elif c == "poke" and len(a) >= 2 and a[0] in preview:
                preview[a[0]].append(species_from_details(a[1]))
            elif c == "showteam" and len(a) >= 2:
                showteam[a[0]] = unpack_team(dex, "|".join(a[1:]))
        if game_type not in ("singles", "doubles") or not winner_name:
            return None
        winner = next((s for s, n in players.items() if n == winner_name), None)
        if winner not in ("p1", "p2"):
            return None
        n_active = 2 if game_type == "doubles" else 1
        pick = pick or (4 if game_type == "doubles" else 3)

        # ---- 판 전체에서 드러난 정보
        hs = _Hindsight()
        hs.feed(lines)
        appeared, leads = {"p1": [], "p2": []}, {"p1": [], "p2": []}
        started = False
        for ln in lines:
            c, a = _cmd(ln)
            if c == "start":
                started = True
            elif c == "turn":
                started = None if started else started
            if c in ("switch", "drag", "replace") and a:
                side, slot, name = split_ident(a[0])
                if side in appeared and name not in appeared[side]:
                    appeared[side].append(name)
                    if started is True:
                        leads[side].append(name)
        # 프리뷰 종 → 이름·세트
        teams, names = {}, {}
        for side in ("p1", "p2"):
            if not preview[side]:   # 프리뷰가 없는 형식이면 나온 포켓몬만
                preview[side] = [hs.mons[side][n].species for n in appeared[side] if n in hs.mons[side]]
            st_sets = {dex.ident_name(s["species"]): s for s in showteam.get(side, [])}
            sets, nm, used = [], [], set()
            for sp_txt in preview[side]:
                sid = dex.find("species", sp_txt.replace("-*", "")) or to_id(sp_txt.replace("-*", ""))
                if sid not in dex.species:
                    continue
                name = dex.ident_name(sid)
                hv = hs.mons[side].get(name)
                if hv and hv.species in dex.species:
                    sid = dex.base_of_mega(hv.species)
                s = st_sets.get(name) or self._fill_set(sid, side, name, hs, game_type, used)
                if s.get("item"):
                    used.add(s["item"])
                sets.append(s)
                nm.append(name)
            teams[side], names[side] = sets, nm
        for side in ("p1", "p2"):
            if any(n not in names[side] for n in appeared[side]):
                return None   # 프리뷰와 실제 등장이 안 맞는 기록

        # ---- 데려온 포켓몬과 처음 순서 (안 나온 칸은 사용률 높은 순으로 추정)
        brought = {}
        for side in ("p1", "p2"):
            order = list(leads[side]) + [n for n in appeared[side] if n not in leads[side]]
            rest = [n for n in names[side] if n not in order]
            rest.sort(key=lambda n: -((dex.usage_of(teams[side][names[side].index(n)]["species"], game_type) or {}).get("usage", 0)))
            brought[side] = (order + rest)[:max(pick, len(order))]
        return self._replay(lines, meta, players, winner, game_type, pick, n_active, teams, names, brought, leads, appeared)

    def _replay(self, lines, meta, players, winner, game_type, pick, n_active, teams, names, brought, leads, appeared):
        dex = self.dex
        views = {s: BattleView(s) for s in ("p1", "p2")}
        sets_by = {s: {names[s][i]: teams[s][i] for i in range(len(names[s]))} for s in ("p1", "p2")}
        order = {s: list(brought[s]) for s in ("p1", "p2")}
        maxhp = {}
        for s in ("p1", "p2"):
            for n, st in sets_by[s].items():
                m = Mon(species=st["species"], ability=st.get("ability", ""), item=st.get("item", ""),
                        nature=st.get("nature", "serious"), sp=dict(st.get("sp") or {}), moves=list(st["moves"]))
                m.setup(dex)
                mm = copy.copy(m)
                mega_stats = None
                if mm.mega_evolve():
                    mega_stats = dict(mm.stats)
                maxhp[(s, n)] = (m.maxhp, dict(m.stats), mega_stats)
        cur_item = {(s, n): st.get("item", "") for s in ("p1", "p2") for n, st in sets_by[s].items()}
        pp_used = {}
        lock = {}
        charging = {}       # (side, name) → 2턴 기술 (다음 턴은 그 기술만)
        decisions = []
        turn_open = {}      # side → dict(snapshot, req, options, actions{slot: …}, mega{slot})
        force_open = {}     # side → dict(..., parts{slot: name})
        started = False

        def slot_idx(slot):
            return {"a": 0, "b": 1}.get(slot or "a", 0)

        def build_req(side, kind, force_slots=None, strict=True):
            pokes = []
            v = views[side]
            for i, n in enumerate(order[side]):
                st = sets_by[side][n]
                mv = v.mons[side].get(n)
                mh, stats, mstats = maxhp[(side, n)]
                fainted = bool(mv and mv.fainted)
                pct = mv.hp_pct if mv else 100.0
                species = mv.species if (mv and mv.species) else st["species"]
                if fainted:
                    cond = "0 fnt"
                else:
                    hp = max(1, round(pct * mh / 100.0)) if pct > 0 else 0
                    cond = f"{hp}/{mh}" + (f" {mv.status}" if mv and mv.status else "")
                active = i < n_active and kind != "preview"
                pokes.append({"ident": f"{side}: {n}", "details": f"{dex.species[species]['name'] if species in dex.species else n}, L50",
                              "condition": cond, "active": active, "item": cur_item.get((side, n), ""),
                              "ability": st.get("ability", ""), "baseAbility": st.get("ability", ""),
                              "moves": list(st["moves"]), "stats": {k: x for k, x in ((mstats if mv and mv.mega and mstats else stats)).items() if k != "hp"}})
            req = {"side": {"name": side, "id": side, "pokemon": pokes}, "rqid": None}
            if kind == "force":
                req["forceSwitch"] = [i in (force_slots or ()) for i in range(n_active)]
                return req
            act = []
            for i in range(n_active):
                if i >= len(order[side]):
                    break
                n = order[side][i]
                st = sets_by[side][n]
                mv = views[side].mons[side].get(n)
                vol = set(mv.volatiles) if mv else set()
                lk = lock.get((side, n)) if (cur_item.get((side, n), "") or "").startswith("choice") else None
                moves = []
                chg = charging.get((side, n))
                for mid in ([chg] if chg else st["moves"]):
                    mo = dex.moves.get(mid, {})
                    mx = mo.get("maxpp") or mo.get("pp") or 10
                    dis = False
                    if strict:
                        if lk and lk in st["moves"] and mid != lk:
                            dis = True
                        if "taunt" in vol and mo.get("category") == "Status":
                            dis = True
                        if "encore" in vol and mv and mv.last_move and mid != mv.last_move:
                            dis = True
                    moves.append({"move": mo.get("name", mid), "id": mid, "pp": max(0, mx - pp_used.get((side, n, mid), 0)),
                                  "maxpp": mx, "target": "self" if chg else mo.get("target", "normal"), "disabled": dis})
                can_mega = bool(st.get("megaTarget")) and not views[side].mega_used[side] and not (mv and mv.mega)
                act.append({"moves": moves, "canMegaEvo": can_mega, "trapped": False})
            req["active"] = act
            return req

        def open_decision(side, kind, force_slots=None):
            req = build_req(side, kind, force_slots)
            views[side].set_request(req)
            return {"side": side, "kind": kind, "turn": views[side].turn, "view": views[side].snapshot(),
                    "req": req, "order": list(order[side]), "actions": {}, "mega": set(), "parts": {},
                    "foe_fainted": False, "foes": len(views[side].active("p2" if side == "p1" else "p1"))}

        def finish(d):
            side = d["side"]
            req = d["req"]
            if d["kind"] == "force":
                slots = sorted(d["parts"])
                req["forceSwitch"] = [i in slots for i in range(n_active)]
            for attempt in (0, 1):
                opts = legal_choices(req)
                if len(opts) < 2:
                    self.stats["dropped_forced"] += 1
                    return
                ok = [k for k, o in enumerate(opts) if self._consistent(d, o["parts"], req)]
                if ok or attempt:
                    break
                if d["kind"] != "turn":
                    break
                # 내가 추정한 '못 쓰는 기술'(구애 고정·도발 등)이 틀렸을 수 있다 — 풀고 다시
                req = build_req(side, "turn", strict=False)
                req["side"] = d["req"]["side"]
            if not ok:
                self.stats["dropped_mismatch"] += 1
                return
            if len(ok) == len(opts):
                self.stats["dropped_unknown"] += 1
                return
            p = [0.0] * len(opts)
            for k in ok:
                p[k] = round(1.0 / len(ok), 4)
            self.stats["decisions"] += 1
            decisions.append({"side": side, "turn": d["turn"], "kind": d["kind"], "view": d["view"],
                              "req": compact_req(req), "options": [o["choice"] for o in opts], "chosen": ok[0],
                              "probs": p, "src": "replay"})

        for ln in lines:
            c, a = _cmd(ln)
            if c is None:
                continue
            tg = _tags(a)
            args = [x for x in a if not x.startswith("[")]
            if c == "teampreview":
                for v in views.values():
                    v.feed([ln])
                for side in ("p1", "p2"):
                    self._preview(side, views, teams, names, brought, leads, appeared, pick, decisions)
                continue
            if c == "start":
                started = True
            # 줄을 먹이기 전에 처리해야 하는 것: 강제 교체 시점의 스냅샷
            if c == "switch" and started and args:
                side, slot, name = split_ident(args[0])
                td = turn_open.get(side)
                si = slot_idx(slot)
                pending = td is not None and si not in td["actions"]
                if leads_phase(views, side):
                    pass
                elif pending:
                    td["actions"][si] = ("switch", name)
                else:
                    fd = force_open.get(side)
                    if fd is None:
                        fd = open_decision(side, "force", force_slots=[si])
                        force_open[side] = fd
                    fd["parts"][si] = name
            if c in ("turn", "win", "tie", "move"):
                for side in list(force_open):
                    finish(force_open.pop(side))
            if c in ("turn", "win", "tie"):
                for side in list(turn_open):
                    finish(turn_open.pop(side))
            # 줄 먹이기
            for v in views.values():
                v.feed([ln])
            # 먹인 뒤 처리
            if c in ("switch", "drag", "replace") and args:
                side, slot, name = split_ident(args[0])
                si = slot_idx(slot)
                if name in order[side]:
                    p = order[side].index(name)
                    if p != si and si < len(order[side]):
                        order[side][si], order[side][p] = order[side][p], order[side][si]
                lock.pop((side, name), None)
                charging.pop((side, name), None)
            elif c == "move" and args:
                side, slot, name = split_ident(args[0])
                mid = to_id(args[1]) if len(args) > 1 else ""
                frm = tg.get("from", "")
                td = turn_open.get(side)
                si = slot_idx(slot)
                chg = charging.pop((side, name), None)
                if td is not None and si not in td["actions"] and chg:
                    td["actions"][si] = ("move", mid, None, True)
                elif td is not None and si not in td["actions"]:
                    if frm.startswith(LOCKED_FROM) or mid == "struggle" or mid == "recharge":
                        td["actions"][si] = ("forced",)
                    elif not frm:
                        tgt = args[2] if len(args) > 2 and args[2].startswith("p") else None
                        # 노린 상대가 이미 쓰러졌거나 처음부터 한 마리뿐이면 쇼다운이 대상을 바꾼다 → 대상은 모름
                        loose = "spread" in tg or "notarget" in tg or td["foe_fainted"] or td["foes"] < 2
                        td["actions"][si] = ("move", mid, tgt, loose)
                if not frm or frm.startswith(LOCKED_FROM):
                    pp_used[(side, name, mid)] = pp_used.get((side, name, mid), 0) + 1
                    if (cur_item.get((side, name)) or "").startswith("choice") and not frm:
                        lock.setdefault((side, name), mid)
            elif c == "-prepare" and len(args) > 1:
                side, _, name = split_ident(args[0])
                charging[(side, name)] = to_id(args[1])
            elif c == "-mega" and args:
                side, slot, name = split_ident(args[0])
                td = turn_open.get(side)
                if td is not None:
                    td["mega"].add(slot_idx(slot))
            elif c == "cant" and args:
                side, slot, name = split_ident(args[0])
                td = turn_open.get(side)
                si = slot_idx(slot)
                if td is not None and si not in td["actions"]:
                    reason = to_id(args[1]) if len(args) > 1 else ""
                    if reason in ("recharge", "mustrecharge"):
                        td["actions"][si] = ("forced",)
                    elif len(args) > 2 and to_id(args[2]) in dex.moves:
                        # '|cant|X|Disable|기술' 처럼 고른 기술이 같이 나오는 경우
                        td["actions"][si] = ("move", to_id(args[2]), None, True)
                    else:
                        td["actions"][si] = ("unknown",)
            elif c == "faint" and args:
                side, slot, name = split_ident(args[0])
                other = turn_open.get("p2" if side == "p1" else "p1")
                if other is not None:
                    other["foe_fainted"] = True
                td = turn_open.get(side)
                si = slot_idx(slot)
                if td is not None and si not in td["actions"]:
                    td["actions"][si] = ("unknown",)
            elif c == "-enditem" and args:
                side, _, name = split_ident(args[0])
                cur_item[(side, name)] = ""
                lock.pop((side, name), None)
            elif c == "-item" and len(args) > 1:
                side, _, name = split_ident(args[0])
                cur_item[(side, name)] = to_id(args[1])
            elif c == "turn":
                for side in ("p1", "p2"):
                    if any(not views[side].mons[side].get(n) or not views[side].mons[side][n].fainted
                           for n in order[side][:n_active]):
                        turn_open[side] = open_decision(side, "turn")
        return {
            "v": 1, "id": meta.get("id") or "replay", "format": meta.get("formatid") or meta.get("format") or "",
            "gameType": game_type, "dataVersion": dex.meta.get("builtAt"), "seed": None,
            "teams": teams, "policies": {s: "human" for s in ("p1", "p2")}, "players": players,
            "rating": meta.get("rating"), "decisions": decisions, "winner": winner,
            "turns": views["p1"].turn, "source": "replay",
        }

    # ------------------------------------------------------------ 선출
    def _preview(self, side, views, teams, names, brought, leads, appeared, pick, decisions):
        n = len(names[side])
        req = {"teamPreview": True, "maxChosenTeamSize": pick,
               "side": {"name": side, "id": side, "pokemon": [
                   {"ident": f"{side}: {names[side][i]}", "details": f"{self.dex.species[teams[side][i]['species']]['name']}, L50",
                    "condition": "100/100", "active": False, "item": teams[side][i].get("item", ""),
                    "ability": teams[side][i].get("ability", ""), "moves": list(teams[side][i]["moves"]), "stats": None}
                   for i in range(n)]}}
        opts = legal_choices(req)
        if len(opts) < 2:
            return
        idx = {nm: i + 1 for i, nm in enumerate(names[side])}
        lead_set = {idx[x] for x in leads[side] if x in idx}
        seen = {idx[x] for x in appeared[side] if x in idx}
        lead_n = 1 if pick <= 3 else 2
        ok = []
        for k, o in enumerate(opts):
            order = o["parts"]
            if set(order[:lead_n]) != lead_set or not seen <= set(order):
                continue
            ok.append(k)
        if not ok or len(ok) == len(opts):
            self.stats["dropped_unknown"] += 1
            return
        views[side].set_request(req)
        p = [0.0] * len(opts)
        for k in ok:
            p[k] = round(1.0 / len(ok), 4)
        self.stats["decisions"] += 1
        decisions.append({"side": side, "turn": 0, "kind": "preview", "view": views[side].snapshot(),
                          "req": compact_req(req), "options": [o["choice"] for o in opts], "chosen": ok[0],
                          "probs": p, "src": "replay"})

    # ------------------------------------------------------------ 행동이 선택지와 맞는가
    def _consistent(self, d, parts, req):
        if d["kind"] == "force":
            for si, part in enumerate(parts):
                want = d["parts"].get(si)
                if want is None:
                    if part != "pass":
                        return False
                    continue
                if not part.startswith("switch "):
                    return False
                k = int(part.split()[1])
                if d["order"][k - 1] != want:
                    return False
            return True
        act = req.get("active") or []
        for si, part in enumerate(parts):
            a = d["actions"].get(si, ("unknown",))
            if a[0] == "forced":
                return False
            if a[0] == "unknown":
                continue
            toks = part.split()
            if a[0] == "switch":
                if toks[0] != "switch" or d["order"][int(toks[1]) - 1] != a[1]:
                    return False
                continue
            # 기술
            if toks[0] != "move":
                return False
            j = int(toks[1]) - 1
            moves = act[si]["moves"] if si < len(act) else []
            if j >= len(moves) or moves[j]["id"] != a[1]:
                return False
            mega = toks[-1] == "mega"
            can = act[si].get("canMegaEvo") if si < len(act) else False
            if can and mega != (si in d["mega"]):
                return False
            nums = [t for t in toks[2:] if re.match(r"^-?\d$", t)]
            if nums and a[2] and not a[3]:
                tside, tslot, _ = split_ident(a[2])
                t = int(nums[0])
                pos = {"a": 1, "b": 2}.get(tslot or "a", 1)
                if tside == d["side"]:
                    if t != -pos:
                        return False
                elif t != pos:
                    return False
        return True


def leads_phase(views, side):
    """|start| 직후 선봉이 나오는 중인가 (턴 1 전)."""
    return views[side].turn == 0


# ======================================================================== 검사
def describe(req: dict, choice: str, order_names=None):
    """선택 문자열 → 뜻 (요청 순서와 무관하게 비교하기 위함)."""
    side = req.get("side") or []
    nm = [split_ident(p["ident"])[2] for p in side]
    if choice.startswith("team "):
        k = 1 if (req.get("maxChosenTeamSize") or 3) <= 3 else 2
        o = [nm[int(ch) - 1] for ch in choice[5:]]
        return ("team", frozenset(o[:k]), frozenset(o))
    out = []
    act = req.get("active") or []
    for i, part in enumerate(choice.split(", ")):
        t = part.split()
        if t[0] == "switch":
            out.append(("switch", nm[int(t[1]) - 1]))
        elif t[0] == "move":
            j = int(t[1]) - 1
            mid = act[i]["moves"][j]["id"] if i < len(act) and j < len(act[i]["moves"]) else "?"
            nums = [x for x in t[2:] if re.match(r"^-?\d$", x)]
            out.append(("move", mid, t[-1] == "mega", nums[0] if nums else None))
        else:
            out.append((t[0],))
    return tuple(out)


def check_with_selfplay(n_games=6, game_type="singles", seed=11, ps_dir=None, log=print, policies=None):
    """정답을 아는 우리 시뮬레이터 대전의 관전자 로그를 해석기에 넣어 결정을 얼마나 맞게 복원하는지 잰다."""
    from .psbridge import Bridge
    from .selfplay import _default_policies, pick_format, play_game
    from .teamgen import TeamGen
    dex = Dex()
    tg = TeamGen(dex, seed=seed, game_type=game_type)
    P = ReplayParser(dex)
    tot = {"true": 0, "found": 0, "correct": 0, "exact": 0, "sharp": 0.0, "games": 0, "parsed": 0}
    with Bridge(ps_dir) as br:
        fmt = pick_format(br, game_type)
        for g in range(n_games):
            pol = policies(g) if policies else _default_policies(dex, seed + g)
            rec = play_game(br, dex, fmt, {"p1": tg.team(6), "p2": tg.team(6)}, pol, seed * 1000 + g, f"rc{g}", keep_log=True)
            tot["games"] += 1
            if not rec.get("winner"):
                continue
            # 실제 리플레이처럼 플레이어 이름을 붙인다
            lines = [ln.replace("|player|p1|p1|", "|player|p1|Alice|").replace("|player|p2|p2|", "|player|p2|Bob|")
                     .replace("|win|p1", "|win|Alice").replace("|win|p2", "|win|Bob") for ln in rec["log"]]
            out = P.parse(lines)
            if not out:
                continue
            tot["parsed"] += 1
            got = {}
            for d in out["decisions"]:
                got.setdefault((d["side"], d["kind"], d["turn"]), []).append(d)
            seen_key = {}
            for d in rec["decisions"]:
                if len(d["options"]) < 2:
                    continue
                key = (d["side"], d["kind"], d["turn"])
                i = seen_key.get(key, 0)
                seen_key[key] = i + 1
                tot["true"] += 1
                cand = got.get(key) or []
                if i >= len(cand):
                    continue
                pd = cand[i]
                tot["found"] += 1
                truth = describe(_fullreq(d["req"]), d["options"][d["chosen"]])
                ok = [describe(_fullreq(pd["req"]), pd["options"][k]) for k, p in enumerate(pd["probs"]) if p > 0]
                if truth in ok:
                    tot["correct"] += 1
                    if len(ok) == 1:
                        tot["exact"] += 1
                tot["sharp"] += len(ok) / len(pd["options"])
    f = max(1, tot["found"])
    res = {"games": tot["games"], "parsed": tot["parsed"], "trueDecisions": tot["true"],
           "recovered": round(tot["found"] / max(1, tot["true"]), 3), "correct": round(tot["correct"] / f, 3),
           "exact": round(tot["exact"] / f, 3), "avgCandidateShare": round(tot["sharp"] / f, 3), "parser": P.stats}
    log(json.dumps(res, ensure_ascii=False))
    return res


def _fullreq(c):
    """compact_req → describe 가 읽는 모양."""
    return {"side": c.get("side") or [], "active": c.get("active") or [], "maxChosenTeamSize": c.get("maxChosenTeamSize")}


def _nicknamed(lines, seed=0):
    """검사용: 실제 리플레이처럼 별명·채팅·관전 줄을 섞는다."""
    import random as _r
    rng = _r.Random(seed)
    nick = {}
    out = ["|j|☆Alice", "|j|☆Bob", "|c|☆Alice|gl hf"]
    for ln in lines:
        c, a = _cmd(ln)
        if c in ("switch", "drag", "replace") and a:
            side, _, name = split_ident(a[0])
            nick.setdefault((side, name), f"{name[:3]}#{rng.randrange(100)}x")
        out.append(ln)
        if c == "turn" and rng.random() < 0.2:
            out.append("|c|☆Bob|hmm")
            out.append("|inactive|Bob has 120 seconds left.")

    def sub(m):
        return f"{m.group(1)}{m.group(2)}: {nick.get((m.group(1), m.group(3).strip()), m.group(3).strip())}"
    return [ID_RE.sub(sub, ln) for ln in out]


def selfcheck(ps_dir=None) -> dict:
    """셀프테스트: 우리 대전의 관전자 로그로 해석 정확도 + 별명·채팅이 섞여도 같은 결과인지 + 메가 규칙."""
    from .psbridge import find_ps_dir
    if find_ps_dir(ps_dir) is None:
        return {"ok": True, "skipped": "쇼다운 폴더 없음 — 리플레이 해석 검사 생략"}
    errs = []
    rs = check_with_selfplay(4, "singles", seed=41, ps_dir=ps_dir, log=lambda *_: None)
    rd = check_with_selfplay(3, "doubles", seed=43, ps_dir=ps_dir, log=lambda *_: None)
    if rs["correct"] < 0.97 or rs["recovered"] < 0.8:
        errs.append(f"싱글 해석 정확도 낮음: {rs}")
    if rd["correct"] < 0.85 or rd["recovered"] < 0.75:
        errs.append(f"더블 해석 정확도 낮음: {rd}")
    # 별명·채팅 + 메가
    from .psbridge import Bridge
    from .selfplay import _default_policies, pick_format, play_game
    from .teamgen import TeamGen
    dex = Dex()
    P = ReplayParser(dex)
    tg = TeamGen(dex, seed=47, game_type="singles")
    mega_bad, same = 0, 0
    with Bridge(ps_dir) as br:
        fmt = pick_format(br, "singles")
        for g in range(2):
            rec = play_game(br, dex, fmt, {"p1": tg.team(6), "p2": tg.team(6)}, _default_policies(dex, g), 4700 + g,
                            f"sc{g}", keep_log=True)
            mega_bad += sum(1 for ln in rec["log"] if ln.startswith(("|switch|", "|drag|")) and "-Mega" in ln.split("|")[3])
            if not rec.get("winner"):
                same += 1
                continue
            lines = [ln.replace("|player|p1|p1|", "|player|p1|Alice|").replace("|player|p2|p2|", "|player|p2|Bob|")
                     .replace("|win|p1", "|win|Alice").replace("|win|p2", "|win|Bob") for ln in rec["log"]]
            a, b = P.parse(lines), P.parse(_nicknamed(lines, g))
            if a and b and [(d["kind"], d["turn"], d["probs"]) for d in a["decisions"]] == \
                    [(d["kind"], d["turn"], d["probs"]) for d in b["decisions"]]:
                same += 1
    if same < 2:
        errs.append("별명·채팅을 섞으면 해석 결과가 달라짐")
    if mega_bad:
        errs.append(f"메가 모습으로 시작한 포켓몬 {mega_bad}마리 (메가진화 규칙 위반)")
    return {"ok": not errs, "errors": errs,
            "summary": f"싱글 결정 복원 {rs['recovered']:.0%}·정답 포함 {rs['correct']:.0%} · "
                       f"더블 {rd['recovered']:.0%}·{rd['correct']:.0%} · 별명·채팅 무관 · 메가 규칙 정상"}
