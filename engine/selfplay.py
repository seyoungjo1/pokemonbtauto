# -*- coding: utf-8 -*-
"""자기대전 — 쇼다운 시뮬레이터에서 정책끼리 대전시키고, 결정마다 원본 상태를 기록한다.

기록 (한 판 = JSON 한 줄, games_*.jsonl.gz)
  header : 형식·데이터 버전·두 팀(전체 세트)·정책 이름·시드
  decisions[] : side, turn, kind(preview/turn/force), view(그 플레이어 시점 스냅샷), req(요약),
                options(선택지 문자열), chosen(번호), probs(정책 확률)
  winner, turns

변수(features)는 기록 후에 뽑는다. 변수를 새로 만들어도 다시 대전할 필요가 없다.
"""
from __future__ import annotations

import gzip
import json
import random
import time
from pathlib import Path

from .dex import Dex
from .protocol import BattleView, legal_choices
from .psbridge import Bridge
from .teamgen import TeamGen, to_showdown_set

ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = ROOT / "userdata" / "selfplay"

FORMAT_PREF = {
    "singles": ["gen9championsbssregmc", "gen9championsbssregmb", "gen9championsbssregma"],
    "doubles": ["gen9championsvgc2026regmc", "gen9championsvgc2026regmb", "gen9championsvgc2026regma"],
}


def pick_format(bridge: Bridge, game_type: str) -> str:
    have = {f["id"] for f in bridge.formats()}
    for f in FORMAT_PREF[game_type]:
        if f in have:
            return f
    return "gen9championscustomgame" if game_type == "singles" else "gen9championsdoublescustomgame"


def compact_req(req: dict) -> dict:
    if not req:
        return {}
    out = {"rqid": req.get("rqid")}
    if req.get("teamPreview"):
        out["teamPreview"] = True
        out["maxChosenTeamSize"] = req.get("maxChosenTeamSize")
    if req.get("forceSwitch"):
        out["forceSwitch"] = req["forceSwitch"]
    out["side"] = [{"ident": p["ident"], "details": p["details"], "condition": p["condition"],
                    "active": p.get("active", False), "item": p.get("item"), "ability": p.get("ability"),
                    "moves": p.get("moves"), "stats": p.get("stats")} for p in req["side"]["pokemon"]]
    if req.get("active"):
        out["active"] = [{"moves": [{"id": m.get("id"), "pp": m.get("pp"), "disabled": m.get("disabled", False),
                                     "target": m.get("target")} for m in a.get("moves", [])],
                          "canMegaEvo": a.get("canMegaEvo", False), "trapped": a.get("trapped", False)}
                         for a in req["active"]]
    return out


def play_game(bridge: Bridge, dex: Dex, fmt: str, teams: dict, policies: dict, seed: int,
              game_id: str, max_turns: int = 200, record: bool = True) -> dict:
    packed = {s: bridge.pack([to_showdown_set(dex, m) for m in teams[s]]) for s in ("p1", "p2")}
    views = {s: BattleView(s) for s in ("p1", "p2")}
    for s in ("p1", "p2"):
        policies[s].start(s, teams[s], dex)
    rng = random.Random(seed)
    res = bridge.call(op="start", id=game_id, format=fmt, seed=[seed, seed + 1, seed + 2, seed + 3],
                      p1=packed["p1"], p2=packed["p2"])
    pending, handled = {"p1": None, "p2": None}, {"p1": -1, "p2": -1}
    decisions = []
    steps = 0

    def absorb(r):
        for s in ("p1", "p2"):
            views[s].feed(r.get("log", {}).get(s, []))
            rq = (r.get("requests") or {}).get(s)
            if rq and (rq.get("rqid") is None or rq.get("rqid") != handled[s]) and rq is not pending[s]:
                pending[s] = rq
    absorb(res)
    while not res.get("ended") and steps < max_turns * 4:
        steps += 1
        progressed = False
        for s in ("p1", "p2"):
            rq = pending[s]
            if not rq:
                continue
            views[s].set_request(rq)
            opts = legal_choices(rq)
            if not opts:
                pending[s] = None
                continue
            idx, probs = policies[s].choose(views[s], rq, opts)
            kind = "preview" if rq.get("teamPreview") else ("force" if rq.get("forceSwitch") else "turn")
            if record:
                decisions.append({
                    "side": s, "turn": views[s].turn, "kind": kind, "view": views[s].snapshot(),
                    "req": compact_req(rq), "options": [o["choice"] for o in opts], "chosen": idx,
                    "probs": [round(p, 4) for p in probs] if probs else None,
                })
            handled[s] = rq.get("rqid")
            pending[s] = None
            res = bridge.call(op="choose", id=game_id, side=s, choice=opts[idx]["choice"])
            if not res.get("accepted", True):
                # 시뮬레이터가 거절한 선택 — 다른 것을 무작위로
                for alt in rng.sample(opts, len(opts)):
                    res = bridge.call(op="choose", id=game_id, side=s, choice=alt["choice"])
                    if res.get("accepted", True):
                        break
                else:
                    res = bridge.call(op="choose", id=game_id, side=s, choice="default")
            absorb(res)
            progressed = True
            if res.get("ended"):
                break
        if not progressed:
            break
        if views["p1"].turn > max_turns:
            break
    if not res.get("ended"):
        bridge.call(op="end", id=game_id)
    winner = res.get("winner")
    w = "p1" if winner == "p1" else "p2" if winner == "p2" else None
    return {
        "v": 1, "id": game_id, "format": fmt, "gameType": views["p1"].game_type,
        "dataVersion": dex.meta.get("builtAt"), "seed": seed,
        "teams": teams, "policies": {s: policies[s].name for s in ("p1", "p2")},
        "decisions": decisions, "winner": w, "turns": views["p1"].turn,
        "endFainted": dict(views["p1"].fainted_count),
    }


def run(n_games: int, game_type: str = "singles", out_dir: Path | None = None, seed: int = 1,
        policy_factory=None, ps_dir=None, log=print, explore=0.1) -> Path:
    dex = Dex()
    out_dir = Path(out_dir or GAMES_DIR) / (dex.meta.get("builtAt", "data").replace(":", "")[:15])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"games_{game_type}_{seed}_{int(time.time())}.jsonl.gz"
    tg = TeamGen(dex, seed=seed, game_type=game_type)
    wins = {"p1": 0, "p2": 0, None: 0}
    t0 = time.time()
    with Bridge(ps_dir) as br, gzip.open(out, "wt", encoding="utf-8") as fo:
        fmt = pick_format(br, game_type)
        log(f"형식 {fmt} · {n_games}판 시작")
        for g in range(n_games):
            pol = policy_factory(g) if policy_factory else _default_policies(dex, seed + g)
            teams = {"p1": tg.team(6, explore), "p2": tg.team(6, explore)}
            try:
                rec = play_game(br, dex, fmt, teams, pol, seed * 100000 + g, f"g{seed}_{g}")
            except Exception as e:  # 한 판이 깨져도 계속
                log(f"  {g}판 오류: {type(e).__name__}: {e}")
                continue
            wins[rec["winner"]] += 1
            fo.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            if (g + 1) % max(1, n_games // 10) == 0:
                log(f"  {g + 1}/{n_games}판 · {time.time() - t0:.0f}초 · 승 p1 {wins['p1']} / p2 {wins['p2']} / 무 {wins[None]}")
    return out


def _default_policies(dex, seed):
    from .policies import HeuristicPolicy
    return {"p1": HeuristicPolicy(dex, seed=seed), "p2": HeuristicPolicy(dex, seed=seed + 7)}


def iter_games(paths):
    """기록 파일들을 한 판씩. 아직 쓰는 중이거나 깨진 파일은 읽을 수 있는 데까지만."""
    for p in paths:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            break
        except (EOFError, OSError):
            continue
