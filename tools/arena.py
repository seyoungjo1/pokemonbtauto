#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""대결 평가 — 정책끼리 여러 판을 붙여 승률표를 만든다 (자리는 번갈아).

  python tools/arena.py --me search --vs heuristic,max_damage,showdown_random --type singles --games 40
  정책 이름: search, model, heuristic, max_damage, showdown_random, random
결과: reports/arena.json (누적), reports/arena.md
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import model as M  # noqa: E402
from engine import selfplay as SP  # noqa: E402
from engine.dex import Dex  # noqa: E402
from engine.policies import BASELINES, make_baseline  # noqa: E402

NAMES = {**BASELINES, "search": "수읽기(탐색) AI", "model": "학습 모델 AI"}


def make(name, dex, seed, gt, opts):
    if name == "search":
        from engine.search import SearchPolicy
        return SearchPolicy(dex, seed=seed, worlds=opts.worlds, rollouts=opts.rollouts, depth=opts.depth, prior=opts.prior, leaf=opts.leaf)
    if name == "model":
        m = M.load(gt)
        if m is None:
            raise SystemExit(f"{gt} 학습 모델이 없습니다")
        return M.ModelPolicy(dex, m, seed=seed)
    return make_baseline(name, dex, seed)


def match(dex, gt, a, b, n, seed, opts):
    def fac(g):
        A, B = make(a, dex, seed + g, gt, opts), make(b, dex, seed + 1000 + g, gt, opts)
        return {"p1": A, "p2": B} if g % 2 == 0 else {"p1": B, "p2": A}
    t0 = time.time()
    p = SP.run(n, gt, out_dir=ROOT / "userdata" / "arena", seed=seed, policy_factory=fac, log=lambda *x: None)
    wins = played = turns = 0
    for g in SP.iter_games([p]):
        if g["winner"] is None:
            continue
        idx = int(g["id"].split("_")[-1])
        a_side = "p1" if idx % 2 == 0 else "p2"
        played += 1
        wins += g["winner"] == a_side
        turns += g["turns"]
    return {"me": a, "vs": b, "gameType": gt, "wins": wins, "played": played,
            "winrate": round(wins / played, 3) if played else 0, "avgTurns": round(turns / max(1, played), 1),
            "seconds": round(time.time() - t0), "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "settings": {"worlds": opts.worlds, "rollouts": opts.rollouts, "depth": opts.depth, "prior": opts.prior, "leaf": opts.leaf}}


def save(results):
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    pj = out / "arena.json"
    allr = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else []
    allr += results
    pj.write_text(json.dumps(allr[-300:], ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 대결 평가", "", "| 시각 (UTC) | 형식 | 나 | 상대 | 전적 | 승률 | 평균 턴 | 탐색 설정 |", "|---|---|---|---|---|---|---|---|"]
    for r in allr[-80:]:
        st = r.get("settings") or {}
        lines.append(f"| {r['when'][:16]} | {'싱글' if r['gameType'] == 'singles' else '더블'} | {NAMES.get(r['me'], r['me'])} | "
                     f"{NAMES.get(r['vs'], r['vs'])} | {r['wins']}/{r['played']} | {r['winrate']:.0%} | {r['avgTurns']} | "
                     + (f"세계 {st.get('worlds')}·굴림 {st.get('rollouts')}·{st.get('depth')}턴" if r['me'] == 'search' else "-") + " |")
    (out / "arena.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default="search")
    ap.add_argument("--vs", default="heuristic,max_damage,showdown_random")
    ap.add_argument("--type", default="singles", choices=["singles", "doubles", "both"])
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--prior", type=float, default=0.08)
    ap.add_argument("--leaf", default="mix", choices=["mix", "race", "material"])
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    dex = Dex()
    res = []
    for gt in (["singles", "doubles"] if a.type == "both" else [a.type]):
        for vs in a.vs.split(","):
            r = match(dex, gt, a.me, vs, a.games, a.seed, a)
            print(f"[{gt}] {NAMES.get(a.me, a.me)} vs {NAMES.get(vs, vs)}: {r['wins']}/{r['played']} ({r['winrate']:.0%}) · {r['seconds']}초", flush=True)
            res.append(r)
            save([r])
    return res


if __name__ == "__main__":
    main()
