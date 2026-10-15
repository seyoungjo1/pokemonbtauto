# -*- coding: utf-8 -*-
"""사람 vs AI 대전 (앱의 'AI와 대전' 탭).

쇼다운 시뮬레이터로 실제 대전을 진행한다. 두 플레이어는 동시에 고르므로, AI 는 요청을 받는 즉시 고르고
(내 선택을 보지 않음) 사람의 선택을 기다린다. 판이 끝나면 자기대전과 같은 형식으로 기록해
학습 데이터(userdata/selfplay/…/games_<형식>_vsuser_*.jsonl.gz)에 더한다 — 사람의 결정은 'human' 정답으로,
AI 의 결정은 그 AI 의 정답으로 쓰인다.
"""
from __future__ import annotations

import gzip
import json
import random
import re
import time

from .dex import Dex, to_id
from .protocol import BattleView, legal_choices, split_ident
from .psbridge import Bridge
from .selfplay import GAMES_DIR, compact_req, pick_format
from .teamgen import TeamGen, normalize_set, to_showdown_set

AI_LEVELS = {
    "search": "수읽기 AI (가장 강함 · 한 수에 몇 초)",
    "net": "신경망 AI (빠름)",
    "heuristic": "선생님 AI (데미지 계산 규칙)",
    "max_damage": "최대 데미지 AI",
    "showdown_random": "쇼다운 기본 AI",
}
STATUS_KO = {"brn": "화상", "par": "마비", "psn": "독", "tox": "맹독", "slp": "잠듦", "frz": "얼음"}
STAT_KO = {"atk": "공격", "def": "방어", "spa": "특공", "spd": "특방", "spe": "스피드", "accuracy": "명중", "evasion": "회피"}
WEATHER_KO = {"sunnyday": "쾌청", "raindance": "비", "sandstorm": "모래바람", "snowscape": "설경", "snow": "설경",
              "desolateland": "아주 강한 햇살", "primordialsea": "강한 비", "none": "날씨 없음"}


def make_ai(name, dex: Dex, game_type, seed=None, bridge=None):
    from . import nets
    from .policies import make_baseline
    if name == "search":
        from .search import SearchPolicy
        return SearchPolicy(dex, worlds=6, rollouts=3, depth=3, seed=seed, bridge=bridge, net=nets.load_net(game_type))
    if name == "net":
        net = nets.load_net(game_type)
        if net is not None:
            return nets.NetPolicy(dex, net, temperature=0.15, seed=seed)
        name = "heuristic"
    return make_baseline(name, dex, seed)


class PlaySession:
    def __init__(self, dex: Dex, game_type: str, my_team: list, ai: str = "search", opp_team=None,
                 seed=None, ps_dir=None):
        self.dex, self.gt = dex, game_type
        self.id = f"play{int(time.time())}"
        self.seed = seed if seed is not None else random.randrange(1, 10 ** 6)
        self.br = Bridge(ps_dir)
        self.fmt = pick_format(self.br, game_type)
        tg = TeamGen(dex, seed=self.seed, game_type=game_type)
        my_team = [normalize_set(dex, s) for s in my_team if s.get("species") in dex.species and s.get("moves")]
        if len(my_team) < (4 if game_type == "doubles" else 3):
            raise ValueError("팀에 포켓몬이 부족합니다 (기술이 있는 포켓몬 " + ("4" if game_type == "doubles" else "3") + "마리 이상).")
        opp = opp_team or (tg.meta_team(6) if tg.usage else []) or tg.team(6)
        self.teams = {"p1": my_team, "p2": [normalize_set(dex, s) for s in opp]}
        self.me, self.ai_side = "p1", "p2"
        self.ai_name = ai
        self.ai = make_ai(ai, dex, game_type, seed=self.seed, bridge=self.br)
        self.ai.start(self.ai_side, self.teams[self.ai_side], dex)
        self.views = {s: BattleView(s) for s in ("p1", "p2")}
        self.pending = {"p1": None, "p2": None}
        self.handled = {"p1": -1, "p2": -1}
        self.decisions, self.spec, self.lines = [], [], []
        self.ended, self.winner, self.saved = False, None, None
        self.ai_info = []
        packed = {s: self.br.pack([to_showdown_set(dex, m) for m in self.teams[s]]) for s in ("p1", "p2")}
        res = self.br.call(op="start", id=self.id, format=self.fmt,
                           seed=[self.seed, self.seed + 1, self.seed + 2, self.seed + 3], p1=packed["p1"], p2=packed["p2"])
        self._absorb(res)
        self._ai_turns()

    # ------------------------------------------------------------ 진행
    def _absorb(self, r):
        self.spec.extend(r.get("log", {}).get("spec", []))
        self.lines.extend(r.get("log", {}).get(self.me, []))
        for s in ("p1", "p2"):
            self.views[s].feed(r.get("log", {}).get(s, []))
            rq = (r.get("requests") or {}).get(s)
            if rq and (rq.get("rqid") is None or rq.get("rqid") != self.handled[s]) and rq is not self.pending[s]:
                self.pending[s] = rq
        if r.get("ended"):
            self.ended = True
            w = r.get("winner")
            self.winner = w if w in ("p1", "p2") else None
            self._save()

    def _send(self, side, choice):
        res = self.br.call(op="choose", id=self.id, side=side, choice=choice)
        return res

    def _record(self, side, rq, opts, idx, probs):
        kind = "preview" if rq.get("teamPreview") else ("force" if rq.get("forceSwitch") else "turn")
        self.decisions.append({"side": side, "turn": self.views[side].turn, "kind": kind,
                               "view": self.views[side].snapshot(), "req": compact_req(rq),
                               "options": [o["choice"] for o in opts], "chosen": idx,
                               "probs": [round(p, 4) for p in probs] if probs else None})

    def _ai_turns(self):
        guard = 0
        while not self.ended and self.pending[self.ai_side] and guard < 8:
            guard += 1
            s = self.ai_side
            rq = self.pending[s]
            self.views[s].set_request(rq)
            opts = legal_choices(rq)
            if not opts:
                self.pending[s] = None
                break
            t0 = time.time()
            idx, probs = self.ai.choose(self.views[s], rq, opts)
            self.ai_info.append({"turn": self.views[s].turn, "seconds": round(time.time() - t0, 1),
                                 "value": getattr(self.ai, "last_value", None)})
            self._record(s, rq, opts, idx, probs)
            self.handled[s] = rq.get("rqid")
            self.pending[s] = None
            res = self._send(s, opts[idx]["choice"])
            if not res.get("accepted", True):
                for alt in opts:
                    res = self._send(s, alt["choice"])
                    if res.get("accepted", True):
                        break
            self._absorb(res)

    def choose(self, choice: str) -> dict:
        """사람의 선택 ('move 1', 'switch 3', 'move 1 2, move 2 1', 'team 213'…)."""
        if self.ended:
            raise ValueError("이미 끝난 대전입니다.")
        rq = self.pending[self.me]
        if not rq:
            raise ValueError("지금은 고를 차례가 아닙니다.")
        opts = legal_choices(rq)
        idx = self._match(rq, opts, choice)
        if idx is None:
            raise ValueError("고를 수 없는 선택입니다: " + choice)
        self.views[self.me].set_request(rq)
        self._record(self.me, rq, opts, idx, None)
        self.handled[self.me] = rq.get("rqid")
        self.pending[self.me] = None
        send = choice.strip() if choice.startswith("team ") else opts[idx]["choice"]
        res = self._send(self.me, send)
        if not res.get("accepted", True):
            self.pending[self.me] = rq
            self.decisions.pop()
            raise ValueError("시뮬레이터가 거절한 선택입니다: " + ((res.get("errors") or {}).get(self.me) or send))
        self._absorb(res)
        self._ai_turns()
        return self.state()

    @staticmethod
    def _match(rq, opts, choice):
        choice = re.sub(r"\s+", " ", choice.strip())
        for i, o in enumerate(opts):
            if o["choice"] == choice:
                return i
        if choice.startswith("team "):
            order = [int(c) for c in choice[5:].strip()]
            k = rq.get("maxChosenTeamSize") or len(order)
            if len(order) != k or len(set(order)) != k:
                return None
            lead_n = 1 if k <= 3 else 2
            for i, o in enumerate(opts):
                p = o["parts"]
                if set(p[:lead_n]) == set(order[:lead_n]) and set(p) == set(order):
                    return i
        return None

    def forfeit(self):
        if not self.ended:
            self.ended, self.winner = True, self.ai_side
            self.br.call(op="end", id=self.id)
            self._save()

    def close(self):
        try:
            if not self.ended:
                self.br.call(op="end", id=self.id)
        finally:
            self.br.close()

    def _save(self):
        if self.saved or not self.decisions:
            return
        rec = {"v": 1, "id": self.id, "format": self.fmt, "gameType": self.gt, "dataVersion": self.dex.meta.get("builtAt"),
               "seed": self.seed, "teams": self.teams, "policies": {self.me: "human:user", self.ai_side: self.ai_name},
               "decisions": self.decisions, "winner": self.winner, "turns": self.views["p1"].turn,
               "endFainted": dict(self.views["p1"].fainted_count), "log": self.spec, "source": "vsuser"}
        out_dir = GAMES_DIR / (self.dex.meta.get("builtAt", "data").replace(":", "")[:15])
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"games_{self.gt}_vsuser_{int(time.time())}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.saved = str(p.name)

    # ------------------------------------------------------------ 화면용
    def _ko(self, kind, i):
        return (getattr(self.dex, kind).get(i) or {}).get("ko") or i

    def _mon_ko(self, side, name):
        v = self.views[self.me]
        m = v.mons.get(side, {}).get(name)
        sid = m.species if m and m.species else to_id(name)
        return self._ko("species", sid) if sid in self.dex.species else name

    def _name(self, ident):
        side, _, name = split_ident(ident)
        who = "내" if side == self.me else "상대"
        return f"{who} {self._mon_ko(side, name)}"

    def log_text(self, last=200):
        out = []
        for ln in self.lines:
            t = self._line_text(ln)
            if t:
                out.append(t)
        return out[-last:]

    def _line_text(self, ln):
        if not ln.startswith("|"):
            return None
        p = ln.split("|")[1:]
        c, a = p[0], p[1:]
        tags = {m.group(1): m.group(2) for m in (re.match(r"^\[(\w+)\]\s*(.*)$", x) for x in a) if m}
        a = [x for x in a if not x.startswith("[")]
        frm = tags.get("from", "")
        why = ""
        if frm.startswith("item:"):
            why = f" ({self._ko('items', to_id(frm[5:]))})"
        elif frm.startswith("ability:"):
            why = f" ({self._ko('abilities', to_id(frm[8:]))})"
        elif frm:
            why = f" ({self._ko('moves', to_id(frm.replace('move:', '')))})" if to_id(frm.replace('move:', '')) in self.dex.moves else ""
        try:
            if c == "turn":
                return f"── {a[0]}턴 ──"
            if c in ("switch", "drag"):
                hp = a[2].split()[0] if len(a) > 2 else ""
                return f"{self._name(a[0])} 등장! {self._hp_txt(a[0], hp)}"
            if c == "move":
                tgt = f" → {self._name(a[2])}" if len(a) > 2 and a[2].startswith("p") and a[2][:3] != a[0][:3] else ""
                return f"{self._name(a[0])}의 {self._ko('moves', to_id(a[1]))}!{tgt}"
            if c == "-damage":
                return f"   {self._name(a[0])} {self._hp_txt(a[0], a[1].split()[0])}{why}"
            if c in ("-heal", "-sethp"):
                return f"   {self._name(a[0])} 회복 {self._hp_txt(a[0], a[1].split()[0])}{why}"
            if c == "faint":
                return f"   {self._name(a[0])} 쓰러짐!"
            if c == "-status":
                return f"   {self._name(a[0])} {STATUS_KO.get(a[1], a[1])}{why}"
            if c == "-curestatus":
                return f"   {self._name(a[0])} 상태이상 회복"
            if c in ("-boost", "-unboost"):
                sign = "+" if c == "-boost" else "-"
                return f"   {self._name(a[0])} {STAT_KO.get(a[1], a[1])} {sign}{a[2]}{why}"
            if c == "-supereffective":
                return "   효과가 굉장했다!"
            if c == "-resisted":
                return "   효과가 별로인 듯하다"
            if c == "-immune":
                return f"   {self._name(a[0])}에게는 효과가 없다{why}"
            if c == "-crit":
                return "   급소에 맞았다!"
            if c == "-miss":
                return "   빗나갔다!"
            if c == "-fail":
                return "   실패했다"
            if c == "cant":
                r = to_id(a[1]) if len(a) > 1 else ""
                return f"   {self._name(a[0])} 움직일 수 없다 ({STATUS_KO.get(r, self._ko('moves', r) if r in self.dex.moves else r)})"
            if c == "-mega":
                return f"{self._name(a[0])} 메가진화!"
            if c == "-weather":
                w = to_id(a[0])
                if tags.get("upkeep") is not None and "upkeep" in tags:
                    return None
                return f"날씨: {WEATHER_KO.get(w, a[0])}{why}"
            if c == "-fieldstart":
                return f"필드: {self._ko('moves', to_id(a[0].replace('move:', '')))}"
            if c == "-fieldend":
                return f"필드 끝: {self._ko('moves', to_id(a[0].replace('move:', '')))}"
            if c == "-sidestart":
                who = "내 진영" if a[0].startswith(self.me) else "상대 진영"
                return f"{who}: {self._ko('moves', to_id(a[1].replace('move:', '')))}"
            if c == "-sideend":
                who = "내 진영" if a[0].startswith(self.me) else "상대 진영"
                return f"{who}: {self._ko('moves', to_id(a[1].replace('move:', '')))} 끝"
            if c == "-enditem":
                return f"   {self._name(a[0])}의 {self._ko('items', to_id(a[1]))} 사라짐"
            if c == "-item":
                return f"   {self._name(a[0])}: {self._ko('items', to_id(a[1]))}{why}"
            if c == "-ability":
                return f"   {self._name(a[0])}의 {self._ko('abilities', to_id(a[1]))}"
            if c == "win":
                return "★ 승리했습니다!" if a and a[0] == self.me else "★ 패배했습니다."
        except (IndexError, KeyError):
            return None
        return None

    def _hp_txt(self, ident, hp):
        if hp.strip() == "0":
            return "HP 0%"
        if "/" not in hp:
            return ""
        x, y = hp.split("/")
        y = re.sub(r"[a-z]+$", "", y)
        try:
            x, y = float(x), float(y)
        except ValueError:
            return ""
        pct = round(100 * x / y) if y else 0
        if ident.startswith(self.me) and y not in (100, 48):
            return f"HP {int(x)}/{int(y)} ({pct}%)"
        return f"HP {pct}%"

    def _mon_card(self, side, m):
        d = self.dex
        return {"name": m.name, "species": m.species, "ko": self._ko("species", m.species) if m.species in d.species else m.name,
                "hpPct": round(m.hp_pct, 1), "hp": m.hp, "maxhp": m.maxhp, "status": m.status,
                "statusKo": STATUS_KO.get(m.status, ""), "fainted": m.fainted, "slot": m.slot,
                "boosts": {STAT_KO.get(k, k): v for k, v in m.boosts.items() if v}, "mega": m.mega,
                "item": self._ko("items", m.item) if m.item else ("없음" if m.item == "" else ""),
                "ability": self._ko("abilities", m.ability) if m.ability else "",
                "moves": [self._ko("moves", x) for x in m.moves]}

    def _options(self, rq):
        """사람이 고를 선택지를 자리별로 (기술·대상·메가 / 교체)."""
        if not rq:
            return None
        d = self.dex
        if rq.get("teamPreview"):
            return {"kind": "preview", "pick": rq.get("maxChosenTeamSize") or len(rq["side"]["pokemon"]),
                    "mons": [{"n": i + 1, "species": to_id(p["details"].split(",")[0]),
                              "ko": self._ko("species", to_id(p["details"].split(",")[0])),
                              "item": self._ko("items", to_id(p.get("item") or "")) if p.get("item") else "",
                              "moves": [self._ko("moves", to_id(m)) for m in p.get("moves", [])]}
                             for i, p in enumerate(rq["side"]["pokemon"])]}
        joint = legal_choices(rq)
        n_slots = len(joint[0]["parts"]) if joint else 0
        pokes = rq["side"]["pokemon"]
        opp_act = {m.slot: m for m in self.views[self.me].active(self.ai_side)}
        my_act = {m.slot: m for m in self.views[self.me].active(self.me)}
        slots = []
        for si in range(n_slots):
            parts = []
            for o in joint:
                p = o["parts"][si]
                base = p[:-5] if p.endswith(" mega") else p
                if base not in parts:
                    parts.append(base)
            entries = []
            act = (rq.get("active") or [{}] * n_slots)[si] if not rq.get("forceSwitch") else {}
            for p in parts:
                t = p.split()
                if t[0] == "move":
                    j = int(t[1]) - 1
                    mv = (act.get("moves") or [])[j] if j < len(act.get("moves") or []) else {}
                    mid = to_id(mv.get("id", ""))
                    md = d.moves.get(mid, {})
                    label = md.get("ko") or mid or "발버둥"
                    tgt = ""
                    if len(t) > 2:
                        n = int(t[2])
                        if n > 0:
                            m = opp_act.get("a" if n == 1 else "b")
                            tgt = f"상대 {'왼쪽' if n == 1 else '오른쪽'}" + (f" {self._ko('species', m.species)}" if m else "")
                        else:
                            m = my_act.get("a" if n == -1 else "b")
                            tgt = "같은 편" + (f" {self._ko('species', m.species)}" if m else "")
                    entries.append({"part": p, "kind": "move", "label": label, "target": tgt, "type": md.get("type", ""),
                                    "category": md.get("category", ""), "power": md.get("basePower", 0),
                                    "pp": mv.get("pp"), "maxpp": mv.get("maxpp"),
                                    "mega": any(o["parts"][si] == p + " mega" for o in joint)})
                elif t[0] == "switch":
                    k = int(t[1])
                    pk = pokes[k - 1]
                    sid = to_id(pk["details"].split(",")[0])
                    entries.append({"part": p, "kind": "switch", "label": self._ko("species", sid), "species": sid,
                                    "condition": pk["condition"]})
                else:
                    entries.append({"part": p, "kind": p, "label": "넘기기" if p == "pass" else p})
            who = pokes[si] if si < len(pokes) else None
            slots.append({"slot": si, "who": self._ko("species", to_id(who["details"].split(",")[0])) if who else "",
                          "entries": entries})
        return {"kind": "force" if rq.get("forceSwitch") else "turn", "slots": slots,
                "joint": [o["choice"] for o in joint]}

    def state(self) -> dict:
        v = self.views[self.me]
        mine = [self._mon_card(self.me, m) for m in v.mons[self.me].values()]
        opp = [self._mon_card(self.ai_side, m) for m in v.mons[self.ai_side].values()]
        return {
            "id": self.id, "gameType": self.gt, "format": self.fmt, "turn": v.turn, "ended": self.ended,
            "winner": ("me" if self.winner == self.me else "ai" if self.winner else None),
            "ai": self.ai_name, "aiLabel": AI_LEVELS.get(self.ai_name, self.ai_name),
            "mine": mine, "opp": opp,
            "oppPreview": [{"species": s, "ko": self._ko("species", s)} for s in v.preview.get(self.ai_side, [])],
            "weather": WEATHER_KO.get(v.weather, v.weather), "terrain": self._ko("moves", v.terrain) if v.terrain else "",
            "options": None if self.ended else self._options(self.pending[self.me]),
            "log": self.log_text(), "saved": self.saved, "aiSeconds": [x["seconds"] for x in self.ai_info[-3:]],
        }
