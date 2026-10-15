#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""밤샘 학습 — 알파고 제로식 반복 (파티 짜기 → 선출 → 배틀 전체).

  python tools/overnight.py --hours 8 --type singles --workers 6

한 회차
  1. 대전 생성 (CPU 여러 코어): 수읽기 AI(정책망을 사전확률로) vs 리그 상대
     [수읽기·신경망·선생님·최대 데미지·쇼다운 기본 AI·과거 신경망], 팀은 메타 팀 + 최적화된 팀
     → 결정마다 '수읽기가 매긴 분포'를 정답으로 기록
  2. 학습 (GPU · CUDA, 없으면 CPU): 최근 대전 기록으로 정책망·가치망 학습
  3. 평가: 새 신경망 vs 지금 최고 신경망 · 기본 AI 들 — 이길 때만 교체 (models/net_<형식>.json)
  4. 파티 최적화 몇 걸음 (포켓몬·도구·SP·기술·선출까지 대전으로 평가)
  5. 리포트: reports/insights.md (승률 변화, 잘 이기는 포켓몬·선봉·교체·메가 시점, 파티 개선)

끄고 다시 켜면 이어서 한다. 프로그램 폴더에 '학습중지.txt' 를 만들면 이번 회차를 마치고 멈춘다.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATE = ROOT / "userdata" / "overnight" / "state.json"
STOP = ROOT / "학습중지.txt"
GAMES = ROOT / "userdata" / "selfplay"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state():
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"iter": 0, "progress": [], "teamNotes": [], "started": now()}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


# ======================================================================== 대전 생성 (작업자)
def worker(args):
    gt, n, seed, settings = args
    from engine import nets, selfplay as SP
    from engine.dex import Dex
    from engine.policies import make_baseline
    from engine.search import SearchPolicy
    dex = Dex()
    net = nets.load_net(gt)
    past = sorted(glob.glob(str(ROOT / "models" / "history" / f"net_{gt}_*.json")))[-4:]
    past_nets = []
    for p in past:
        try:
            past_nets.append(nets.PolicyValueNet.load(p))
        except Exception:
            pass
    rng = random.Random(seed)

    def me(s):
        return SearchPolicy(dex, seed=s, worlds=settings["worlds"], rollouts=settings["rollouts"],
                            depth=settings["depth"], net=net)

    def opp(s):
        pool = [("search", 0.3), ("net", 0.15 if net else 0), ("heuristic", 0.2), ("max_damage", 0.1),
                ("showdown_random", 0.05), ("past", 0.2 if past_nets else 0)]
        names, w = zip(*[x for x in pool if x[1] > 0])
        k = rng.choices(names, w)[0]
        if k == "search":
            return me(s)
        if k == "net":
            return nets.NetPolicy(dex, net, temperature=0.3, seed=s)
        if k == "past":
            p = nets.NetPolicy(dex, rng.choice(past_nets), temperature=0.3, seed=s)
            p.name = "past_net"
            return p
        return make_baseline(k, dex, s)

    def fac(g):
        a, b = me(seed + g), opp(seed + 5000 + g)
        return {"p1": a, "p2": b} if g % 2 == 0 else {"p1": b, "p2": a}
    out = SP.run(n, gt, seed=seed, policy_factory=fac, log=lambda *x: None, explore=0.15)
    return str(out)


def generate(gt, total, workers, settings, seed, log):
    per = max(1, total // workers)
    jobs = [(gt, per, seed * 1000 + i, settings) for i in range(workers)]
    t0 = time.time()
    if workers == 1:
        files = [worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            files = pool.map(worker, jobs)
    log(f"  대전 {per * workers}판 생성 · {time.time() - t0:.0f}초")
    return files


# ======================================================================== 학습
def train(gt, log, max_games, epochs):
    import numpy as np
    from engine import nets, nets_torch, selfplay as SP
    from engine.dex import Dex
    dex = Dex()
    files = sorted(glob.glob(str(GAMES / "**" / f"games_{gt}_*.jsonl.gz"), recursive=True),
                   key=os.path.getmtime)[::-1]
    data = nets.build_listwise(SP.iter_games(files), dex, gt, max_games=max_games, log=log)
    if len(data["S"]) < 200:
        log("  학습할 결정이 너무 적어 건너뜀")
        return None
    tr, va = nets.split(data)
    net = nets.PolicyValueNet(len(data["S"][0]), data["O"][0].shape[1])
    net.set_norm([data["S"][i] for i in tr[:20000]], [data["O"][i] for i in tr[:20000]])
    if nets_torch.available():
        r = nets_torch.fit_torch(net, data, tr, va, epochs=epochs, log=log)
        log(f"  학습 장치: {r['device']}")
    else:
        log("  학습 장치: CPU (numpy) — PyTorch 를 설치하면 GPU 로 학습")
        net.fit_numpy(data, epochs=min(epochs, 10), tr_idx=tr, val_idx=va, log=log)
    m = net.evaluate(data, va[:3000])
    net.info = {"metrics": m, "games": data["meta"]["games"], "decisions": len(data["S"])}
    tmp = ROOT / "models" / f"net_{gt}_candidate.json"
    net.save(tmp, data["meta"])
    log(f"  검증: 정책 일치 {m['top1']:.1%} · 승패 예측 {m['valueAcc']:.1%}")
    return tmp, data["meta"]["games"]


def evaluate(gt, cand_path, n, log):
    """새 신경망 vs 지금 최고 · 기본 AI 들."""
    import tools.arena as A  # noqa
    from engine import nets
    from engine import selfplay as SP
    from engine.dex import Dex
    from engine.policies import make_baseline
    dex = Dex()
    cand = nets.PolicyValueNet.load(cand_path)
    best = nets.load_net(gt)

    def match(make_b, label, games):
        def fac(g):
            a = nets.NetPolicy(dex, cand, temperature=0.15, seed=g)
            b = make_b(g)
            return {"p1": a, "p2": b} if g % 2 == 0 else {"p1": b, "p2": a}
        p = SP.run(games, gt, out_dir=ROOT / "userdata" / "arena", seed=4321, policy_factory=fac, log=lambda *x: None)
        w = pl = 0
        for g in SP.iter_games([p]):
            if g["winner"] is None:
                continue
            idx = int(g["id"].split("_")[-1])
            pl += 1
            w += g["winner"] == ("p1" if idx % 2 == 0 else "p2")
        p.unlink(missing_ok=True)
        r = {"wins": w, "played": pl, "winrate": w / pl if pl else 0, "ko": label}
        log(f"  새 신경망 vs {label}: {w}/{pl} ({r['winrate']:.0%})")
        return r
    league = {}
    for k, label in (("showdown_random", "쇼다운 기본 AI"), ("max_damage", "최대 데미지 AI"), ("heuristic", "선생님 AI")):
        league[k] = match(lambda g, k=k: make_baseline(k, dex, g + 77), label, max(10, n // 2))
    if best is not None:
        league["best"] = match(lambda g: nets.NetPolicy(dex, best, temperature=0.15, seed=g + 9), "지금 최고 신경망", n)
    return league


# ======================================================================== 파티 최적화 (몇 걸음)
def team_steps(gt, steps, log, st):
    from engine.dex import Dex
    from engine.psbridge import Bridge
    from engine.selfplay import pick_format
    from engine.teamopt import TeamOptimizer
    dex = Dex()
    with Bridge() as br:
        fmt = pick_format(br, gt)
        opt = TeamOptimizer(dex, br, fmt, gt, n_opp=8, games=2, k_orders=4, seed=st["iter"] + 7, log=log)
        cur = st.get("bestTeam", {}).get(gt)
        res = opt.optimize(cur, steps=steps)
    st.setdefault("bestTeam", {})[gt] = res["team"]
    for h in res["history"][1:]:
        st["teamNotes"].append(f"[{gt}] {h['change']} → 승률 {h['score']:.0%}")
    log(f"  파티 최적화: 승률 {res['score']:.1%} · 평가 {res['evals']}번")


# ======================================================================== 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8)
    ap.add_argument("--type", default="singles", choices=["singles", "doubles", "both"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--games", type=int, default=240, help="회차마다 만들 대전 수")
    ap.add_argument("--train-games", type=int, default=6000, help="학습에 쓸 최근 판 수")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--team-steps", type=int, default=6)
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=0, help="0 = 시간이 다 될 때까지")
    a = ap.parse_args()
    from engine import insights, selfplay as SP
    from engine.dex import Dex
    t_end = time.time() + a.hours * 3600
    st = load_state()
    logf = ROOT / "userdata" / "overnight" / "log.txt"
    logf.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    types = ["singles", "doubles"] if a.type == "both" else [a.type]
    settings = {"worlds": a.worlds, "rollouts": a.rollouts, "depth": a.depth}
    log(f"밤샘 학습 시작 · {a.hours}시간 · 작업자 {a.workers} · 형식 {types} · 회차 {st['iter'] + 1}부터")
    done = 0
    while time.time() < t_end:
        if STOP.is_file():
            log("학습중지.txt 확인 — 멈춥니다.")
            break
        st["iter"] += 1
        it = st["iter"]
        for gt in types:
            log(f"===== 회차 {it} · {'싱글' if gt == 'singles' else '더블'} =====")
            generate(gt, a.games, a.workers, settings, seed=it * 37 + (0 if gt == "singles" else 11), log=log)
            r = train(gt, log, a.train_games, a.epochs)
            if not r:
                continue
            cand, n_games = r
            league = evaluate(gt, cand, a.eval_games, log)
            best_wr = league.get("best", {}).get("winrate")
            keep = best_wr is None or best_wr >= 0.55
            if keep:
                from engine import nets
                dst = nets.path_for(gt)
                hist = ROOT / "models" / "history"
                hist.mkdir(parents=True, exist_ok=True)
                if dst.is_file():
                    (hist / f"net_{gt}_{int(time.time())}.json").write_bytes(dst.read_bytes())
                cand.replace(dst)
                log(f"  → 새 신경망 채택 ({'첫 신경망' if best_wr is None else f'이전 최고 상대 {best_wr:.0%}'})")
            else:
                cand.unlink(missing_ok=True)
                log(f"  → 유지 (이전 최고 상대 {best_wr:.0%} < 55%)")
            st["progress"].append({"iter": it, "when": now(), "gameType": gt, "games": n_games,
                                   "league": {k: {"winrate": v["winrate"], "played": v["played"]} for k, v in league.items()},
                                   "kept": keep})
            if a.team_steps:
                try:
                    team_steps(gt, a.team_steps, log, st)
                except Exception as e:
                    log(f"  파티 최적화 오류: {e}")
            save_state(st)
        # 리포트
        dex = Dex()
        files = sorted(glob.glob(str(GAMES / "**" / "games_*.jsonl.gz"), recursive=True), key=os.path.getmtime)[-40:]
        an = insights.analyze_games(dex, SP.iter_games(files))
        p = insights.write_report(dex, an, st["progress"], st["teamNotes"])
        log(f"리포트: {p.relative_to(ROOT)}")
        save_state(st)
        done += 1
        if a.iterations and done >= a.iterations:
            break
    log("밤샘 학습 종료")


if __name__ == "__main__":
    main()
