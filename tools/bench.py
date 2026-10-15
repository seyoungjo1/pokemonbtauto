#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""벤치마크 평가 — 아키타입 고정 팀으로 거울전/교차전, 윌슨 95% 신뢰구간.

  python tools/bench.py --me search --vs heuristic --games 100 --mode mirror
  python tools/bench.py --me net --vs heuristic --games 400 --keys rain,stall,trickroom
결과: reports/bench.json 에 누적, 표는 reports/bench.md
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make(name, dex, seed, a):
    from engine import nets
    from engine.policies import make_baseline
    if name == "search":
        from engine.search import SearchPolicy
        return SearchPolicy(dex, worlds=a.worlds, rollouts=a.rollouts, depth=a.depth, seed=seed,
                            net=nets.load_net(a.type) if not a.no_net else None, leaf=a.leaf)
    if name == "net":
        n = nets.load_net(a.type)
        if n is None:
            raise SystemExit("신경망 없음")
        return nets.NetPolicy(dex, n, temperature=0.15, seed=seed)
    return make_baseline(name, dex, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default="search")
    ap.add_argument("--vs", default="heuristic")
    ap.add_argument("--type", default="singles")
    ap.add_argument("--games", type=int, default=100, help="팀마다 판 수 (양쪽 자리 반씩)")
    ap.add_argument("--mode", default="mirror", choices=["mirror", "cross"])
    ap.add_argument("--keys", default="")
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--leaf", default="mix")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from engine import benchmarks as BM
    from engine.dex import Dex
    from engine.psbridge import Bridge
    from engine.selfplay import pick_format
    dex = Dex()
    keys = [k for k in a.keys.split(",") if k] or None
    t0 = time.time()
    with Bridge() as br:
        fmt = pick_format(br, a.type)
        print(f"{a.me} vs {a.vs} · {a.mode} · 팀마다 {a.games}판 · {fmt}")
        res = BM.evaluate(dex, br, fmt, lambda s: make(a.me, dex, s, a), lambda s: make(a.vs, dex, s + 7, a),
                          a.type, games=a.games, mode=a.mode, keys=keys, seed=a.seed)
    entry = {"when": datetime.now(timezone.utc).isoformat(timespec="minutes"), "type": a.type, "me": a.me, "vs": a.vs,
             "mode": a.mode, "games": a.games, "search": {"worlds": a.worlds, "rollouts": a.rollouts, "depth": a.depth, "leaf": a.leaf},
             "result": res, "seconds": round(time.time() - t0)}
    rp = ROOT / "reports" / "bench.json"
    hist = json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else []
    hist.append(entry)
    rp.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 벤치마크 (아키타입 고정 팀 · 윌슨 95% 신뢰구간)", "",
             "| 시각 | 형식 | 나 | 상대 | 방식 | 팀당 판 | " + " | ".join(BM.SINGLES[k][0] for k in BM.SINGLES) + " | 전체 |",
             "|---|---|---|---|---|---|" + "---|" * (len(BM.SINGLES) + 1)]
    for e in hist[-40:]:
        r = e["result"]
        cell = lambda k: f"{r[k]['winrate']:.0%}" if k in r else "-"
        al = r["_all"]
        lines.append(f"| {e['when'][5:16]} | {e['type']} | {e['me']} | {e['vs']} | {e['mode']} | {e['games']} | "
                     + " | ".join(cell(k) for k in BM.SINGLES) + f" | **{al['winrate']:.0%}** [{al['ci'][0]:.0%}–{al['ci'][1]:.0%}] |")
    (ROOT / "reports" / "bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
