# -*- coding: utf-8 -*-
"""쇼다운 프로토콜 해석기 — 한 플레이어 시점의 '보이는 정보'를 추적한다.

자기대전 기록, 실제 쇼다운 리플레이, 앱에서 손으로 입력한 대전을 모두 같은 형태(BattleView)로 만든다.
학습 변수(features)는 이 BattleView 스냅샷에서만 뽑으므로, 상대의 숨은 정보(아직 안 보인 기술·도구·
특성·SP)는 학습에도 쓰이지 않는다 — 실제 대전과 같은 조건.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dfield

from .dex import to_id

SIDE_CONDS = {"reflect", "lightscreen", "auroraveil", "tailwind", "safeguard", "mist",
              "stealthrock", "spikes", "toxicspikes", "stickyweb"}
WEATHER_IDS = {"sunnyday", "raindance", "sandstorm", "snowscape", "snow", "hail",
               "desolateland", "primordialsea", "deltastream"}


def parse_hp(s: str):
    """'58/100 brn' → (hp, maxhp, status, fainted)."""
    s = (s or "").strip()
    if not s:
        return None
    if s.endswith("fnt") or s.startswith("0 "):
        return (0, 100, "", True)
    parts = s.split()
    status = parts[1] if len(parts) > 1 else ""
    # 상대 HP 는 게임처럼 48칸 체력바로만 보인다: '25/48y' (y/r/g = 색)
    hp_txt = re.sub(r"[a-z]+$", "", parts[0])
    if "/" in hp_txt:
        a, b = hp_txt.split("/")
        return (float(a), float(b), status, float(a) <= 0)
    return (float(hp_txt), 100.0, status, float(hp_txt) <= 0)


def split_ident(ident: str):
    """'p2a: Garchomp' → ('p2', 'a', 'Garchomp')."""
    m = re.match(r"^(p\d)([a-c]?)\s*:?\s*(.*)$", ident.strip())
    if not m:
        return None, None, ident
    return m.group(1), m.group(2) or None, m.group(3).strip()


def species_from_details(details: str) -> str:
    return details.split(",")[0].strip()


@dataclass
class MonView:
    side: str
    name: str
    species: str = ""
    hp_pct: float = 100.0
    hp: int | None = None
    maxhp: int | None = None
    status: str = ""
    fainted: bool = False
    boosts: dict = dfield(default_factory=dict)
    volatiles: set = dfield(default_factory=set)
    slot: str | None = None            # 'a' / 'b' (나와 있으면)
    moves: list = dfield(default_factory=list)     # 드러난(상대) 또는 아는(나) 기술
    move_pp: dict = dfield(default_factory=dict)
    item: str | None = None            # None = 모름, '' = 없음(소모·탈취)
    ability: str | None = None
    stats: dict | None = None
    mega: bool = False
    revealed: bool = False             # 대전에 나왔는가
    last_move: str = ""
    turns_out: int = 0
    times_attacked: int = 0
    first_seen_turn: int | None = None

    def snapshot(self) -> dict:
        d = {
            "name": self.name, "species": self.species, "hpPct": round(self.hp_pct, 1),
            "status": self.status, "fainted": self.fainted,
            "boosts": {k: v for k, v in self.boosts.items() if v},
            "volatiles": sorted(self.volatiles), "slot": self.slot, "moves": list(self.moves),
            "item": self.item, "ability": self.ability, "mega": self.mega, "revealed": self.revealed,
            "lastMove": self.last_move, "turnsOut": self.turns_out, "timesAttacked": self.times_attacked,
        }
        if self.hp is not None:
            d["hp"], d["maxhp"] = self.hp, self.maxhp
        if self.stats:
            d["stats"] = self.stats
        if self.move_pp:
            d["pp"] = dict(self.move_pp)
        return d


class BattleView:
    def __init__(self, me: str = "p1"):
        self.me = me
        self.opp = "p2" if me == "p1" else "p1"
        self.game_type = "singles"
        self.turn = 0
        self.mons = {"p1": {}, "p2": {}}      # name → MonView
        self.preview = {"p1": [], "p2": []}   # 팀 프리뷰 종족
        self.pick_size = None
        self.weather, self.weather_start = "", 0
        self.terrain, self.terrain_start = "", 0
        self.pseudo = {}                       # trickroom/gravity → 시작 턴
        self.side_conds = {"p1": {}, "p2": {}}  # cond → {start, layers}
        self.mega_used = {"p1": False, "p2": False}
        self.fainted_count = {"p1": 0, "p2": 0}
        self.events = []                       # 이번 판의 사건 목록 (턴, 종류, 주체, 대상, 값)
        self.winner = None
        self.ended = False
        self.request = None

    # ------------------------------------------------------------ 조회
    def mon(self, side: str, name: str, create=True) -> MonView | None:
        m = self.mons[side].get(name)
        if m is None and create:
            m = MonView(side=side, name=name)
            self.mons[side][name] = m
        return m

    def active(self, side: str) -> list:
        out = [m for m in self.mons[side].values() if m.slot and not m.fainted]
        return sorted(out, key=lambda m: m.slot)

    def by_ident(self, ident: str):
        side, slot, name = split_ident(ident)
        if side not in self.mons:
            return None
        return self.mon(side, name)

    # ------------------------------------------------------------ 입력
    def feed(self, lines):
        for line in lines:
            self._line(line)

    def set_request(self, req: dict | None):
        """내 쪽 요청(JSON) — 내 팀의 정확한 HP·기술·도구·특성·능력치."""
        self.request = req
        if not req or "side" not in req:
            return
        if not req.get("teamPreview"):
            # 선출이 끝나면 데려오지 않은 포켓몬은 내 시점에서 뺀다
            names = {split_ident(p["ident"])[2] for p in req["side"]["pokemon"]}
            for n in [n for n in self.mons[self.me] if n not in names]:
                del self.mons[self.me][n]
        for i, p in enumerate(req["side"]["pokemon"]):
            side, _, name = split_ident(p["ident"])
            m = self.mon(self.me, name)
            m.species = to_id(species_from_details(p["details"]))
            hp = parse_hp(p["condition"])
            if hp and hp[3]:
                m.hp, m.hp_pct, m.fainted = 0, 0.0, True
                m.slot = None
            elif hp:
                m.hp, m.maxhp = int(hp[0]), int(hp[1]) if hp[1] else None
                m.hp_pct = 100.0 * hp[0] / hp[1] if hp[1] else 0
                m.status = hp[2] if hp[2] != "fnt" else ""
                m.fainted = hp[3]
            m.moves = [to_id(x) for x in p.get("moves", [])]
            m.item = to_id(p.get("item", ""))
            m.ability = to_id(p.get("ability") or p.get("baseAbility") or "")
            m.stats = p.get("stats")
            m.team_index = i + 1
        if req.get("active"):
            for a, p in zip(req["active"], [x for x in req["side"]["pokemon"] if x.get("active")]):
                _, _, name = split_ident(p["ident"])
                m = self.mon(self.me, name)
                m.move_pp = {to_id(x["id"]): x.get("pp", 0) for x in a.get("moves", []) if "id" in x}

    # ------------------------------------------------------------ 한 줄
    def _line(self, line: str):
        if not line.startswith("|"):
            return
        parts = line.split("|")[1:]
        if not parts:
            return
        cmd, args = parts[0], parts[1:]
        tags = [a for a in args if a.startswith("[")]
        args = [a for a in args if not a.startswith("[")]
        tagd = {}
        for t in tags:
            m = re.match(r"^\[(\w+)\]\s*(.*)$", t)
            if m:
                tagd[m.group(1)] = m.group(2)
        h = getattr(self, "_c_" + cmd.replace("-", "_"), None)
        if h:
            h(args, tagd)
        self._reveal_from_tags(args, tagd)

    def _reveal_from_tags(self, args, tagd):
        src = tagd.get("from", "")
        if not src:
            return
        who = self.by_ident(tagd["of"]) if tagd.get("of") else (self.by_ident(args[0]) if args and args[0].startswith("p") else None)
        if not who:
            return
        if src.startswith("ability:"):
            # [of] 가 있으면 그쪽의 특성 (예: 위협 → [of] 사용자)
            who.ability = to_id(src.split(":", 1)[1])
        elif src.startswith("item:"):
            holder = self.by_ident(args[0]) if args and args[0].startswith("p") and not tagd.get("of") else who
            if holder:
                holder.item = to_id(src.split(":", 1)[1])

    # ---- 일반
    def _c_gametype(self, a, t):
        self.game_type = a[0] if a else "singles"

    def _c_poke(self, a, t):
        if len(a) >= 2 and a[0] in self.preview:
            self.preview[a[0]].append(to_id(species_from_details(a[1]).replace("-*", "")))

    def _c_teampreview(self, a, t):
        if a and a[0].isdigit():
            self.pick_size = int(a[0])

    def _c_turn(self, a, t):
        self.turn = int(a[0])
        for s in ("p1", "p2"):
            for m in self.active(s):
                m.turns_out += 1

    def _c_win(self, a, t):
        self.ended = True
        self.winner = a[0] if a else None

    def _c_tie(self, a, t):
        self.ended = True

    # ---- 등장
    def _c_switch(self, a, t):
        side, slot, name = split_ident(a[0])
        if side not in self.mons:
            return
        for m in self.mons[side].values():   # 같은 자리에 있던 포켓몬은 들어간다
            if m.slot == slot:
                m.slot = None
                m.boosts.clear()
                m.volatiles.clear()
                m.turns_out = 0
        m = self.mon(side, name)
        m.species = to_id(species_from_details(a[1]))
        if "-Mega" in a[1]:
            m.mega = True
        m.slot = slot
        m.revealed = True
        m.turns_out = 0
        if m.first_seen_turn is None:
            m.first_seen_turn = self.turn
        if len(a) > 2:
            self._set_hp(m, a[2])
        self._event("switch", side, m.species)

    _c_drag = _c_switch
    _c_replace = _c_switch

    def _c_detailschange(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.species = to_id(species_from_details(a[1]))
            if "-Mega" in a[1]:
                m.mega = True

    def _c__mega(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.mega = True
            self.mega_used[m.side] = True
            if len(a) > 2:
                m.item = to_id(a[2])
            self._event("mega", m.side, m.species)

    _c__formechange = _c_detailschange

    # ---- 기술
    def _c_move(self, a, t):
        m = self.by_ident(a[0])
        if not m:
            return
        mid = to_id(a[1])
        called = t.get("from", "")
        if not called or called.startswith("lockedmove"):
            if mid not in m.moves and mid != "struggle" and m.side != self.me:
                m.moves.append(mid)
            m.last_move = mid
        tgt = self.by_ident(a[2]) if len(a) > 2 and a[2].startswith("p") else None
        self._event("move", m.side, m.species, mid, tgt.species if tgt else "")
        if tgt and tgt is not m:
            tgt.times_attacked += 1

    def _c_cant(self, a, t):
        m = self.by_ident(a[0])
        if m:
            self._event("cant", m.side, m.species, to_id(a[1]) if len(a) > 1 else "")

    # ---- HP · 상태
    def _set_hp(self, m: MonView, s: str):
        hp = parse_hp(s)
        if not hp:
            return
        cur, mx, status, fnt = hp
        if m.side == self.me and mx and mx != 100 or (m.maxhp and mx == m.maxhp):
            m.hp, m.maxhp = int(cur), int(mx)
        m.hp_pct = 100.0 * cur / mx if mx else 0.0
        if status in ("brn", "par", "psn", "tox", "slp", "frz"):
            m.status = status
        elif not status and not fnt:
            m.status = "" if m.status not in ("brn", "par", "psn", "tox", "slp", "frz") else m.status
        if fnt:
            self._faint(m)

    def _c__damage(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1:
            before = m.hp_pct
            self._set_hp(m, a[1])
            self._event("damage", m.side, m.species, t.get("from", ""), round(before - m.hp_pct, 1))

    def _c__heal(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1:
            self._set_hp(m, a[1])

    _c__sethp = _c__heal

    def _c__status(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1:
            m.status = a[1]
            self._event("status", m.side, m.species, a[1])

    def _c__curestatus(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.status = ""

    def _c__cureteam(self, a, t):
        m = self.by_ident(a[0])
        if m:
            for x in self.mons[m.side].values():
                x.status = ""

    def _faint(self, m: MonView):
        if not m.fainted:
            m.fainted = True
            m.hp_pct = 0.0
            if m.hp is not None:
                m.hp = 0
            self.fainted_count[m.side] += 1
            self._event("faint", m.side, m.species)
        m.slot = None

    def _c_faint(self, a, t):
        m = self.by_ident(a[0])
        if m:
            self._faint(m)

    # ---- 랭크
    def _c__boost(self, a, t, sign=1):
        m = self.by_ident(a[0])
        if m and len(a) > 2:
            st = a[1]
            m.boosts[st] = max(-6, min(6, m.boosts.get(st, 0) + sign * int(a[2])))

    def _c__unboost(self, a, t):
        self._c__boost(a, t, -1)

    def _c__setboost(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 2:
            m.boosts[a[1]] = int(a[2])

    def _c__clearboost(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.boosts.clear()

    def _c__clearallboost(self, a, t):
        for s in ("p1", "p2"):
            for m in self.mons[s].values():
                m.boosts.clear()

    def _c__clearnegativeboost(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.boosts = {k: v for k, v in m.boosts.items() if v > 0}

    def _c__clearpositiveboost(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.boosts = {k: v for k, v in m.boosts.items() if v < 0}

    def _c__invertboost(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.boosts = {k: -v for k, v in m.boosts.items()}

    def _c__copyboost(self, a, t):
        src, dst = self.by_ident(a[0]), self.by_ident(a[1]) if len(a) > 1 else None
        if src and dst:
            dst.boosts = dict(src.boosts)

    # ---- 도구 · 특성
    def _c__item(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1:
            m.item = to_id(a[1])

    def _c__enditem(self, a, t):
        m = self.by_ident(a[0])
        if m:
            m.item = ""
            self._event("enditem", m.side, m.species, to_id(a[1]) if len(a) > 1 else "")

    def _c__ability(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1:
            m.ability = to_id(a[1])

    # ---- 휘발 상태
    def _c__start(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1:
            v = to_id(re.sub(r"^(move|ability|item):\s*", "", a[1]))
            m.volatiles.add(v)

    def _c__end(self, a, t):
        m = self.by_ident(a[0])
        if m and len(a) > 1:
            v = to_id(re.sub(r"^(move|ability|item):\s*", "", a[1]))
            m.volatiles.discard(v)

    # ---- 필드
    def _c__weather(self, a, t):
        w = to_id(a[0]) if a else ""
        if w == "none" or not w:
            self.weather = ""
        elif "upkeep" not in t:
            if w != self.weather:
                self.weather, self.weather_start = w, self.turn
        if w == "snow":
            self.weather = "snowscape"

    def _c__fieldstart(self, a, t):
        eff = to_id(re.sub(r"^move:\s*", "", a[0]))
        if eff.endswith("terrain"):
            self.terrain, self.terrain_start = eff, self.turn
        else:
            self.pseudo[eff] = self.turn

    def _c__fieldend(self, a, t):
        eff = to_id(re.sub(r"^move:\s*", "", a[0]))
        if eff.endswith("terrain"):
            if self.terrain == eff:
                self.terrain = ""
        else:
            self.pseudo.pop(eff, None)

    def _c__sidestart(self, a, t):
        side = a[0][:2]
        eff = to_id(re.sub(r"^move:\s*", "", a[1])) if len(a) > 1 else ""
        if side in self.side_conds and eff:
            cur = self.side_conds[side].get(eff)
            if cur:
                cur["layers"] += 1
            else:
                self.side_conds[side][eff] = {"start": self.turn, "layers": 1}

    def _c__sideend(self, a, t):
        side = a[0][:2]
        eff = to_id(re.sub(r"^move:\s*", "", a[1])) if len(a) > 1 else ""
        if side in self.side_conds:
            self.side_conds[side].pop(eff, None)

    # ---- 결과 표시
    def _c__crit(self, a, t):
        m = self.by_ident(a[0])
        if m:
            self._event("crit", m.side, m.species)

    def _c__supereffective(self, a, t):
        m = self.by_ident(a[0])
        if m:
            self._event("se", m.side, m.species)

    def _c__resisted(self, a, t):
        m = self.by_ident(a[0])
        if m:
            self._event("resisted", m.side, m.species)

    def _c__immune(self, a, t):
        m = self.by_ident(a[0])
        if m:
            self._event("immune", m.side, m.species)

    def _c__miss(self, a, t):
        m = self.by_ident(a[0])
        if m:
            self._event("miss", m.side, m.species)

    def _event(self, kind, side, who, a="", b=""):
        self.events.append([self.turn, kind, side, who, a, b])

    # ------------------------------------------------------------ 스냅샷
    def snapshot(self) -> dict:
        def side_snap(s):
            return [m.snapshot() for m in self.mons[s].values()]
        return {
            "me": self.me, "gameType": self.game_type, "turn": self.turn,
            "weather": self.weather, "weatherTurns": (self.turn - self.weather_start) if self.weather else 0,
            "terrain": self.terrain, "terrainTurns": (self.turn - self.terrain_start) if self.terrain else 0,
            "pseudo": {k: self.turn - v for k, v in self.pseudo.items()},
            "sideConds": {s: {k: {"turns": self.turn - v["start"], "layers": v["layers"]}
                              for k, v in self.side_conds[s].items()} for s in ("p1", "p2")},
            "megaUsed": dict(self.mega_used), "fainted": dict(self.fainted_count),
            "preview": {s: list(v) for s, v in self.preview.items()}, "pickSize": self.pick_size,
            "mons": {s: side_snap(s) for s in ("p1", "p2")},
            "recent": self.events[-12:],
        }


# ======================================================================== 선택지
def legal_choices(req: dict) -> list:
    """요청에서 고를 수 있는 선택 문자열 목록. 각 항목: {"choice": str, "parts": [...]}"""
    if not req or req.get("wait"):
        return []
    if req.get("teamPreview"):
        n = len(req["side"]["pokemon"])
        k = req.get("maxChosenTeamSize") or n
        return _team_orders(n, k)
    pokemon = req["side"]["pokemon"]
    bench = [i + 1 for i, p in enumerate(pokemon)
             if not p.get("active") and not p["condition"].endswith("fnt") and not p.get("reviving")]
    if req.get("forceSwitch"):
        needs = [bool(x) for x in req["forceSwitch"]]
        short = len(bench) < sum(needs)   # 남은 포켓몬이 모자라면 남는 자리는 넘긴다
        slots = []
        for need in needs:
            slots.append(([f"switch {b}" for b in bench] + (["pass"] if short else [])) if need else ["pass"])
        out = _joint(slots, bench)
        want = min(len(bench), sum(needs))
        out = [o for o in out if sum(p.startswith("switch") for p in o["parts"]) == want]
        return out or [{"choice": ", ".join(["pass"] * len(needs)), "parts": ["pass"] * len(needs)}]
    slots = []
    actives = [p for p in pokemon if p.get("active")]
    doubles = len(req.get("active", [])) > 1
    for i, a in enumerate(req.get("active", [])):
        mon = actives[i] if i < len(actives) else None
        if mon is None or mon["condition"].endswith("fnt") or a.get("commanding"):
            slots.append(["pass"])
            continue
        opts = []
        mega = " mega" if a.get("canMegaEvo") else ""
        for j, mv in enumerate(a.get("moves", [])):
            if mv.get("disabled") or mv.get("pp") == 0:
                continue
            for tgt in _targets(mv.get("target"), i, doubles, len(req.get("active", []))):
                base = f"move {j + 1}" + (f" {tgt}" if tgt is not None else "")
                opts.append(base)
                if mega:
                    opts.append(base + mega)
        if not opts:
            opts.append("move 1")   # 발버둥
        if not a.get("trapped") and not a.get("maybeTrapped"):
            opts += [f"switch {b}" for b in bench]
        slots.append(opts)
    return _joint(slots, bench)


def _targets(target, slot_i, doubles, n_active):
    if not doubles:
        return [None]
    if target in ("normal", "any", "adjacentFoe"):
        return [1, 2] if target != "any" else [1, 2, -(2 - slot_i)]
    if target == "adjacentAlly":
        return [-(2 - slot_i)]
    if target == "adjacentAllyOrSelf":
        return [-1, -2]
    return [None]


def _joint(slots, bench):
    out = [[]]
    for opts in slots:
        nxt = []
        for prev in out:
            for o in opts:
                if o.startswith("switch") and o in prev:
                    continue   # 같은 포켓몬으로 두 자리 교체 불가
                if o.endswith(" mega") and any(p.endswith(" mega") for p in prev):
                    continue   # 메가진화는 한 번만
                nxt.append(prev + [o])
        out = nxt or out
    return [{"choice": ", ".join(p), "parts": p} for p in out]


def _team_orders(n, k):
    """팀 선출: k 마리를 고르고 앞쪽이 선봉. 순서까지 전부는 많으므로 선봉 조합 × 나머지 조합."""
    from itertools import combinations
    out = []
    lead_n = 1 if k <= 3 else 2
    for chosen in combinations(range(1, n + 1), k):
        for leads in combinations(chosen, lead_n):
            rest = [c for c in chosen if c not in leads]
            order = list(leads) + rest
            out.append({"choice": "team " + "".join(str(x) for x in order), "parts": order})
    return out


def load_replay_log(text: str) -> list:
    """쇼다운 리플레이(.log 또는 JSON 의 log 필드)에서 줄 목록."""
    text = text.strip()
    if text.startswith("{"):
        text = json.loads(text).get("log", "")
    return text.split("\n")
