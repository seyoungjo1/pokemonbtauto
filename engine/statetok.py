# -*- coding: utf-8 -*-
"""배틀 상태 토큰화 — 한 결정(내 시점 스냅샷 + 요청 + 선택지) → 토큰 묶음 (ARCHITECTURE.md 2.2 '배틀 상태 토큰').

  포켓몬 토큰 12개 : 세트 하위 토큰(종/능력치·특성·도구·기술×4, engine/tokens.py) ⊕ 동적 상태(HP·상태이상·랭크·휘발·자리·공개·PP·…)
                      내 6마리는 요청 순서(1~6), 상대 6마리는 프리뷰 순서. 빈 자리는 마스크.
                      상대의 숨은 세트는 '추정 세트'(드러난 기술 + 메타 사용률) — 나중에 믿음망 출력으로 교체.
  전역 토큰 1개   : 날씨·필드·트릭룸·중력·양쪽 장판/벽/순풍·턴·메가·남은 마리·선출 규모·형식
  역사 토큰 8개   : 최근 사건(턴, 종류, 편, 기술 타입/분류, 데미지%)
  선택지 토큰     : 합법 행동마다 (종류, 기술 하위 토큰 또는 교체 대상 포켓몬 번호, 대상, 메가) — 정책 헤드가 쓴다

여기서는 '숫자 배열'만 만든다. 신경망(engine/battlenet.py)은 이 배열을 어텐션으로 묶는다.
"""
from __future__ import annotations

import numpy as np

from .belief import OppBelief
from .dex import Dex, to_id
from .montoken import TYPES, STATUSES
from .protocol import split_ident
from .teamgen import normalize_set
from .tokens import Vocab, set_tokens

VOLATILES = ["substitute", "confusion", "leechseed", "taunt", "encore", "disable", "yawn", "perishsong", "curse",
             "attract", "protect", "saltcure", "partiallytrapped", "nightmare", "torment", "healblock"]
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
SIDE_CONDS = ["stealthrock", "spikes", "toxicspikes", "stickyweb", "reflect", "lightscreen", "auroraveil", "tailwind"]
EVENT_KINDS = ["switch", "move", "damage", "faint", "status", "mega", "cant", "crit", "se", "resisted", "immune", "miss", "enditem"]
WEATHERS = ["sunnyday", "raindance", "sandstorm", "snowscape"]
TERRAINS = ["electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"]
N_DYN = 1 + len(STATUSES) + len(BOOSTS) + len(VOLATILES) + 4 + 4 + 3 + 2 + 2   # 동적 상태 칸 수
N_GLOBAL = (len(WEATHERS) + 1) + (len(TERRAINS) + 1) + 4 + 2 * (len(SIDE_CONDS) + 2) + 6 + 2
N_EVENT = 1 + len(EVENT_KINDS) + 2 + 18 + 3 + 1
N_OPTION = 5 + 4 + 2


class StateTokenizer:
    def __init__(self, dex: Dex, vocab: Vocab | None = None, game_type="singles"):
        self.dex, self.vocab, self.gt = dex, vocab or Vocab(dex), game_type
        self._set_cache = {}
        self._belief_cache = {}

    # ------------------------------------------------------------ 세트 토큰 (캐시)
    def _set_key(self, s: dict):
        return (s.get("species"), s.get("ability"), s.get("item"), s.get("nature"),
                tuple(sorted((s.get("sp") or {}).items())), tuple(s.get("moves") or []))

    def set_tok(self, s: dict):
        k = self._set_key(s)
        if k not in self._set_cache:
            try:
                self._set_cache[k] = set_tokens(self.dex, s, self.vocab)
            except Exception:
                self._set_cache[k] = None
        return self._set_cache[k]

    def guess_opp_set(self, mv: dict) -> dict:
        """상대 포켓몬의 추정 세트: 드러난 기술·도구·특성 + 메타 사용률 (믿음망 전까지의 자리 메우기)."""
        dex = self.dex
        sid = mv.get("species") or ""
        base = dex.base_of_mega(sid) if sid in dex.species else sid
        key = (base, tuple(mv.get("moves") or []), mv.get("item"), mv.get("ability"))
        if key in self._belief_cache:
            return self._belief_cache[key]
        b = OppBelief(dex, base, self.gt)
        b.observe([m for m in (mv.get("moves") or []) if m in dex.moves], mv.get("item"), mv.get("ability"))
        probs = b.move_probs()
        moves = [m for m in (mv.get("moves") or []) if m in dex.moves][:4]
        for m in probs:
            if len(moves) >= 4:
                break
            if m not in moves and m in dex.moves:
                moves.append(m)
        ip = b.item_probs()
        sps = b.spread_probs() or [{"nature": "serious", "sp": {"hp": 32, "atk": 16, "spa": 16}}]
        item = mv.get("item") if mv.get("item") is not None else (max(ip, key=ip.get) if ip else "")
        if mv.get("mega") and sid in dex.species and dex.species[sid].get("isMega"):
            item = dex.find("items", dex.species[sid].get("requiredItem") or "") or item
        s = {"species": base, "ability": mv.get("ability") or b.likely_ability(), "item": item or "",
             "nature": sps[0].get("nature") or "serious", "sp": dict(sps[0].get("sp") or {}), "moves": moves}
        self._belief_cache[key] = normalize_set(dex, s)
        return self._belief_cache[key]

    # ------------------------------------------------------------ 동적 상태
    def dyn(self, mv: dict | None, mine: bool, active_req: dict | None, position: int, out: bool) -> np.ndarray:
        v = np.zeros(N_DYN, np.float32)
        if mv is None:
            return v
        i = 0
        v[i] = mv.get("hpPct", 100) / 100.0; i += 1
        st = mv.get("status") or ""
        for k, s in enumerate(STATUSES):
            v[i + k] = 1.0 if st == s else 0.0
        i += len(STATUSES)
        bs = mv.get("boosts") or {}
        for k, b in enumerate(BOOSTS):
            v[i + k] = bs.get(b, 0) / 6.0
        i += len(BOOSTS)
        vol = set(mv.get("volatiles") or [])
        for k, name in enumerate(VOLATILES):
            v[i + k] = 1.0 if name in vol else 0.0
        i += len(VOLATILES)
        # 자리·공개·쓰러짐·메가
        v[i] = 1.0 if mv.get("slot") and not mv.get("fainted") else 0.0
        v[i + 1] = 1.0 if mv.get("revealed") else 0.0
        v[i + 2] = 1.0 if mv.get("fainted") else 0.0
        v[i + 3] = 1.0 if mv.get("mega") else 0.0
        i += 4
        # PP 비율 4칸 (내 포켓몬만 정확, 상대는 모름=1)
        pp = mv.get("pp") or {}
        if active_req and active_req.get("moves"):
            for k, m in enumerate(active_req["moves"][:4]):
                mo = self.dex.moves.get(to_id(m.get("id") or ""), {})
                mx = mo.get("maxpp") or mo.get("pp") or 10
                v[i + k] = (m.get("pp") if m.get("pp") is not None else mx) / mx
        else:
            for k in range(4):
                v[i + k] = 1.0
        i += 4
        v[i] = min(1.0, mv.get("turnsOut", 0) / 10.0)
        v[i + 1] = min(1.0, mv.get("timesAttacked", 0) / 6.0)
        v[i + 2] = 1.0 if mv.get("item") == "" else 0.0          # 도구 소모됨
        i += 3
        v[i] = 1.0 if mine else 0.0
        v[i + 1] = position / 6.0
        i += 2
        v[i] = 1.0 if out else 0.0                                  # 선출에서 빠진(또는 미등장) 포켓몬
        v[i + 1] = 1.0 if (mv.get("moves") and not mine) else 0.0  # 상대 기술이 하나라도 드러남
        return v

    # ------------------------------------------------------------ 전역
    def global_tok(self, snap: dict) -> np.ndarray:
        v = []
        w = snap.get("weather") or ""
        v += [1.0 if w == x else 0.0 for x in WEATHERS] + [min(1.0, snap.get("weatherTurns", 0) / 8.0)]
        t = snap.get("terrain") or ""
        v += [1.0 if t == x else 0.0 for x in TERRAINS] + [min(1.0, snap.get("terrainTurns", 0) / 5.0)]
        ps = snap.get("pseudo") or {}
        v += [1.0 if "trickroom" in ps else 0.0, min(1.0, ps.get("trickroom", 0) / 5.0) if "trickroom" in ps else 0.0,
              1.0 if "gravity" in ps else 0.0, min(1.0, ps.get("gravity", 0) / 5.0) if "gravity" in ps else 0.0]
        me, opp = snap["me"], ("p2" if snap["me"] == "p1" else "p1")
        for side in (me, opp):
            sc = (snap.get("sideConds") or {}).get(side, {})
            v += [(sc[k]["layers"] / 3.0 if k in ("spikes", "toxicspikes") else 1.0) if k in sc else 0.0 for k in SIDE_CONDS]
            v += [min(1.0, sc["tailwind"]["turns"] / 4.0) if "tailwind" in sc else 0.0,
                  min(1.0, max((sc[k]["turns"] for k in ("reflect", "lightscreen", "auroraveil") if k in sc), default=0) / 8.0)]
        f = snap.get("fainted") or {}
        pick = snap.get("pickSize") or (4 if snap.get("gameType") == "doubles" else 3)
        v += [min(1.0, snap.get("turn", 0) / 50.0), 1.0 if (snap.get("megaUsed") or {}).get(me) else 0.0,
              1.0 if (snap.get("megaUsed") or {}).get(opp) else 0.0, (pick - f.get(me, 0)) / 6.0, (pick - f.get(opp, 0)) / 6.0, pick / 6.0]
        v += [1.0 if snap.get("gameType") == "doubles" else 0.0, 0.0]
        return np.array(v, np.float32)

    # ------------------------------------------------------------ 역사
    def history(self, snap: dict, n=8) -> tuple:
        ev = (snap.get("recent") or [])[-n:]
        out = np.zeros((n, N_EVENT), np.float32)
        mask = np.zeros(n, bool)
        me = snap["me"]
        for k, e in enumerate(ev):
            turn, kind, side, who, a, b = (list(e) + ["", ""])[:6]
            i = 0
            out[k, i] = min(1.0, (snap.get("turn", 0) - (turn or 0)) / 5.0); i += 1
            for j, kk in enumerate(EVENT_KINDS):
                out[k, i + j] = 1.0 if kind == kk else 0.0
            i += len(EVENT_KINDS)
            out[k, i] = 1.0 if side == me else 0.0
            out[k, i + 1] = 1.0 if side != me else 0.0
            i += 2
            mo = self.dex.moves.get(to_id(str(a))) if kind == "move" else None
            if mo:
                out[k, i + TYPES.index(mo["type"])] = 1.0 if mo["type"] in TYPES else 0.0
                out[k, i + 18 + ["Physical", "Special", "Status"].index(mo["category"])] = 1.0
            i += 21
            out[k, i] = (float(b) / 100.0) if kind == "damage" and isinstance(b, (int, float)) else 0.0
            mask[k] = True
        return out, mask

    # ------------------------------------------------------------ 결정 하나
    def encode(self, snap: dict, req: dict, my_sets: list, options: list) -> dict | None:
        """돌려줌: {'sets': [12개 세트 토큰 or None], 'dyn': [12, N_DYN], 'mon_mask': [12], 'global': [N_GLOBAL],
                   'hist': [8, N_EVENT], 'hist_mask': [8], 'options': [n, N_OPTION], 'opt_move': [n] 기술 하위 토큰 or None,
                   'opt_target': [n] 포켓몬 토큰 번호(0~11) or -1, 'actor': 내 행위자 토큰 번호}"""
        dex = self.dex
        me, opp = snap["me"], ("p2" if snap["me"] == "p1" else "p1")
        by_name = {dex.ident_name(s["species"]): normalize_set(dex, s) for s in my_sets if s.get("species") in dex.species}
        mine_views = {m["name"]: m for m in snap["mons"][me]}
        side = req.get("side") or []
        sets, dyn, mask = [None] * 12, np.zeros((12, N_DYN), np.float32), np.zeros(12, bool)
        actives = req.get("active") or []
        act_i = 0
        actor = -1
        # 내 6마리 (요청 순서)
        for pos, p in enumerate(side[:6]):
            name = split_ident(p["ident"])[2]
            s = by_name.get(name)
            if s is None:
                continue
            mv = mine_views.get(name)
            s2 = dict(s)
            if mv and mv.get("item") is not None and mv.get("item") != s.get("item"):
                s2["item"] = mv.get("item") or ""
            areq = None
            if p.get("active") and act_i < len(actives):
                areq = actives[act_i]
                act_i += 1
                if actor < 0 and not (mv and mv.get("fainted")):
                    actor = pos
            sets[pos] = self.set_tok(s2)
            dyn[pos] = self.dyn(mv, True, areq, pos, out=mv is None)
            mask[pos] = sets[pos] is not None
        # 상대 6마리 (프리뷰 순서, 드러난 정보 반영)
        opp_views = {m["species"]: m for m in snap["mons"][opp]}
        opp_views_base = {dex.base_of_mega(k): v for k, v in opp_views.items() if k in dex.species}
        for pos, sid in enumerate((snap.get("preview") or {}).get(opp, [])[:6]):
            if sid not in dex.species:
                continue
            mv = opp_views.get(sid) or opp_views_base.get(dex.base_of_mega(sid))
            gs = self.guess_opp_set(mv or {"species": sid})
            sets[6 + pos] = self.set_tok(gs)
            dyn[6 + pos] = self.dyn(mv, False, None, pos, out=mv is None)
            mask[6 + pos] = sets[6 + pos] is not None
        if actor < 0:
            actor = next((i for i in range(6) if mask[i]), 0)
        # 선택지
        n = len(options)
        opt = np.zeros((n, N_OPTION), np.float32)
        opt_move, opt_target = [None] * n, np.full(n, -1, np.int64)
        for k, o in enumerate(options):
            parts = [x.strip() for x in o.split(",")]
            p0 = parts[0]
            t = p0.split()
            if t[0] == "team":
                opt[k, 0] = 1.0
                order = [int(c) - 1 for c in t[1]]
                opt_target[k] = order[0]                       # 선봉
                opt[k, 5 + min(3, len(order) - 1)] = 1.0
                opt[k, 9] = sum(order) / 15.0
            elif t[0] == "move":
                opt[k, 1] = 1.0
                j = int(t[1]) - 1
                areq = actives[0] if actives else {}
                mid = to_id((areq.get("moves") or [{}] * 4)[j].get("id", "")) if j < len(areq.get("moves") or []) else ""
                st = sets[actor]
                if st and mid:
                    for vec, i in st["moves"]:
                        if self.vocab.move.get(mid, 0) == i:
                            opt_move[k] = vec
                            break
                    if opt_move[k] is None:
                        from .tokens import move_token
                        from .mon import Mon
                        try:
                            s_ = by_name.get(split_ident(side[actor]["ident"])[2]) or {}
                            m_ = Mon(species=s_["species"], ability=s_.get("ability", ""), item=s_.get("item", ""),
                                     nature=s_.get("nature", "serious"), sp=dict(s_.get("sp") or {}), moves=[mid]).setup(dex)
                            if dex.mega_for(m_.species, m_.item):
                                m_.mega_evolve()
                            opt_move[k] = move_token(dex, self.vocab, mid, m_, (1.0, 1.0))
                        except Exception:
                            opt_move[k] = None
                nums = [x for x in t[2:] if x.lstrip("-").isdigit()]
                if nums:
                    tg = int(nums[0])
                    opt_target[k] = (6 + (tg - 1)) if tg > 0 else (-tg - 1)
                    opt[k, 5 if tg > 0 else 6] = 1.0
                else:
                    opt_target[k] = 6 + next((i for i in range(6) if mask[6 + i] and (dyn[6 + i][1 + len(STATUSES) + len(BOOSTS) + len(VOLATILES)] > 0)), 0)
                if t[-1] == "mega":
                    opt[k, 7] = 1.0
                opt[k, 9] = j / 3.0
            elif t[0] == "switch":
                opt[k, 2] = 1.0
                opt_target[k] = int(t[1]) - 1
                opt[k, 8] = 1.0 if req.get("forceSwitch") else 0.0
            elif t[0] == "pass":
                opt[k, 3] = 1.0
            else:
                opt[k, 4] = 1.0
            opt[k, 10] = len(parts) / 2.0
        hist, hmask = self.history(snap)
        return {"sets": sets, "dyn": dyn, "mon_mask": mask, "global": self.global_tok(snap), "hist": hist, "hist_mask": hmask,
                "options": opt, "opt_move": opt_move, "opt_target": opt_target, "actor": int(actor)}


def selfcheck(dex: Dex | None = None) -> dict:
    """기존 자기대전 기록 몇 판을 토큰화해 모양·마스크·선택지 연결이 맞는지."""
    import glob
    from .selfplay import iter_games, GAMES_DIR
    dex = dex or Dex()
    st = StateTokenizer(dex)
    files = sorted(glob.glob(str(GAMES_DIR / "**" / "games_singles_*.jsonl.gz"), recursive=True))[:1]
    n = bad = 0
    for g in iter_games(files):
        for d in g["decisions"][:40]:
            if len(d.get("options") or []) < 2:
                continue
            e = st.encode(d["view"], d["req"], g["teams"][d["side"]], d["options"])
            n += 1
            if e is None or e["dyn"].shape != (12, N_DYN) or e["global"].shape != (N_GLOBAL,) or e["options"].shape[0] != len(d["options"]):
                bad += 1
                continue
            if d["kind"] == "turn" and any(o.startswith("move") for o in d["options"]):
                if all(m is None for m, o in zip(e["opt_move"], d["options"]) if o.startswith("move")):
                    bad += 1
        break
    return {"ok": n > 0 and bad == 0, "decisions": n, "bad": bad, "dims": {"dyn": N_DYN, "global": N_GLOBAL, "event": N_EVENT, "option": N_OPTION},
            "summary": f"상태 토큰 결정 {n}개 검사 (동적 {N_DYN}·전역 {N_GLOBAL}·사건 {N_EVENT}·선택지 {N_OPTION}칸)"}
