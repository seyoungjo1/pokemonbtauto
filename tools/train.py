#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""학습 도구 (알파고식 반복: 자기대전 → 학습 → 대결 평가 → 변수 장부).

  python tools/train.py selfplay --type singles --games 500 [--policy heuristic|model|mix]
  python tools/train.py train    --type singles [--arena 60]
  python tools/train.py ablation --type singles            # 변수 묶음 제거 실험 → features/ledger.json, FEATURES.md
  python tools/train.py analyze                            # 대전 후 분석 → 다음에 넣을 변수 후보
  python tools/train.py loop     --type both --rounds 3 --games 400
  python tools/train.py report                             # reports/training.md

자기대전에는 Node.js 와 쇼다운 폴더(PS_DIR 또는 build/pokemon-showdown)가 필요하다.
"""
import argparse
import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine import features as F  # noqa: E402
from engine import model as M  # noqa: E402
from engine import selfplay as SP  # noqa: E402
from engine.dex import Dex  # noqa: E402
from engine.policies import BASELINES, HeuristicPolicy, make_baseline  # noqa: E402

REPORTS = ROOT / "reports"
FEATURES_MD = ROOT / "FEATURES.md"
HISTORY = ROOT / "reports" / "training_history.json"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def game_files(game_type=None):
    files = sorted(glob.glob(str(SP.GAMES_DIR / "**" / "games_*.jsonl.gz"), recursive=True))
    files += sorted(glob.glob(str(ROOT / "userdata" / "battles" / "*.jsonl.gz")))
    if game_type:
        files = [f for f in files if f"_{game_type}_" in Path(f).name or "battles" in f]
    return files


# ======================================================================== 자기대전
def cmd_selfplay(a, dex=None):
    """자기대전 + 리그전. --policy
       heuristic : 선생님끼리 (모델이 없을 때의 시작점)
       league    : 현재 모델(또는 선생님) vs 상대 풀 [선생님·최대데미지·쇼다운 기본 AI·무작위·과거 모델들]
       model     : 현재 모델끼리
    """
    dex = dex or Dex()
    types = ["singles", "doubles"] if a.type == "both" else [a.type]
    outs = []
    for gt in types:
        model = M.load(gt)
        past = [M.Model(p) for p in sorted((M.MODELS / "history").glob(f"{gt}_*.json"))[-6:]] \
            if (M.MODELS / "history").is_dir() else []
        pool = [("heuristic", 0.35), ("max_damage", 0.2), ("showdown_random", 0.1), ("random", 0.05)]
        if past:
            pool.append(("past", 0.3))
        names, weights = zip(*pool)

        def me_policy(seed, model=model):
            if model and a.policy in ("league", "model"):
                return M.ModelPolicy(dex, model, temperature=0.05, seed=seed)
            return HeuristicPolicy(dex, temperature=0.12 + 0.1 * (seed % 3), seed=seed)

        def fac(g, gt=gt):
            import random as _r
            seed = a.seed * 7919 + g
            if a.policy == "heuristic":
                pols = [HeuristicPolicy(dex, temperature=0.12 + 0.1 * ((g + k) % 3), seed=seed + k) for k in range(2)]
            elif a.policy == "model":
                pols = [me_policy(seed), me_policy(seed + 1)]
            else:
                rng = _r.Random(seed)
                opp = rng.choices(names, weights)[0]
                if opp == "past":
                    o = M.ModelPolicy(dex, rng.choice(past), temperature=0.05, seed=seed + 1)
                    o.name = "past_model"
                else:
                    o = make_baseline(opp, dex, seed + 1)
                pols = [me_policy(seed), o]
                if g % 2:
                    pols.reverse()
            return {"p1": pols[0], "p2": pols[1]}
        outs.append(SP.run(a.games, gt, seed=a.seed, policy_factory=fac, explore=a.explore))
    return outs


# ======================================================================== 대결 평가
def arena(dex, game_type, pol_a, pol_b, n=40, seed=12345, log=print):
    """pol_a(번호 → 정책) 가 pol_b 를 상대로 몇 판 이기는가 (양쪽 자리 번갈아)."""
    wins = 0
    played = 0

    def fac(g):
        A, B = pol_a(g), pol_b(g)
        return {"p1": A, "p2": B} if g % 2 == 0 else {"p1": B, "p2": A}
    p = SP.run(n, game_type, out_dir=ROOT / "userdata" / "arena", seed=seed, policy_factory=fac,
               log=lambda *x: None)
    for g in SP.iter_games([p]):
        if g["winner"] is None:
            continue
        played += 1
        idx = int(g["id"].split("_")[-1])
        a_side = "p1" if idx % 2 == 0 else "p2"
        wins += g["winner"] == a_side
    p.unlink(missing_ok=True)
    return {"wins": wins, "played": played, "winrate": wins / played if played else 0.0}


def league_eval(dex, gt, cand, current, n):
    """새 모델을 기본 AI 들, 그리고 지금 쓰는 모델과 붙여 상대별 승률을 잰다."""
    out = {}
    plan = [("heuristic", n), ("max_damage", max(10, n // 2)), ("showdown_random", max(10, n // 3))]
    for k, cnt in plan:
        r = arena(dex, gt, lambda g: M.ModelPolicy(dex, cand, seed=g), lambda g, k=k: make_baseline(k, dex, g + 99),
                  n=cnt, seed=4242 + cnt)
        r["ko"] = BASELINES[k]
        out[k] = r
    if current is not None:
        r = arena(dex, gt, lambda g: M.ModelPolicy(dex, cand, seed=g), lambda g: M.ModelPolicy(dex, current, seed=g + 7),
                  n=n, seed=777)
        r["ko"] = "지금 쓰는 모델"
        out["current"] = r
    return out


# ======================================================================== 학습
def load_data(dex, gt, log=print, max_games=None):
    files = game_files(gt)
    if not files:
        raise SystemExit(f"학습할 대전 기록이 없습니다 ({gt}). 먼저 selfplay 를 돌리세요.")
    log(f"[{gt}] 기록 파일 {len(files)}개에서 변수 계산")
    return M.build_dataset(SP.iter_games(files), dex, game_type=gt, log=log, max_games=max_games)


def cmd_train(a, dex=None, data=None):
    dex = dex or Dex()
    types = ["singles", "doubles"] if a.type == "both" else [a.type]
    results = {}
    for gt in types:
        d = data or load_data(dex, gt)
        led = F.sync_ledger()
        exclude = tuple(k for k, v in led["groups"].items() if v.get("status") == "rejected")
        t0 = time.time()
        q, v, mets = M.train_q_v(d, exclude=exclude, seed=a.seed)
        print(f"[{gt}] 학습 {time.time() - t0:.0f}초 · 결정 {len(d[2])}개 · Q {mets['q']} · V {mets['v']}")
        # 저장 전 리그 평가: 새 모델 vs 기본 AI 들 (+ 지금 모델)
        old = M.load(gt)
        tmp = M.save(gt + "_candidate", q, v, d[6], exclude, mets, dex)
        cand = M.Model(tmp)
        # 보정 강도 beta 고르기: 선생님 AI 상대로 가장 잘 이기는 값 (0 = 선생님 그대로)
        beta, beta_tab = 1.0, {}
        if a.arena:
            for b_ in (0.0, 0.5, 1.5, 4.0):
                r = arena(dex, gt, lambda g, b_=b_: M.ModelPolicy(dex, cand, seed=g, beta=b_),
                          lambda g: HeuristicPolicy(dex, seed=g + 99), n=max(20, a.arena * 2 // 3), seed=9090)
                beta_tab[b_] = r["winrate"]
                print(f"[{gt}] 보정 강도 {b_}: 선생님 상대 {r['wins']}/{r['played']} ({r['winrate']:.0%})")
            beta = max(beta_tab, key=lambda b_: (round(beta_tab[b_], 3), -b_))
            cand.info["beta"] = beta
            print(f"[{gt}] 보정 강도 선택: {beta}")
        league = league_eval(dex, gt, cand, old, a.arena) if a.arena else {}
        for k, r in league.items():
            print(f"[{gt}] 새 모델 vs {r['ko']}: {r['wins']}/{r['played']} ({r['winrate']:.0%})")
        keep = True
        reason = "첫 모델"
        if old and league:
            prev = old.info.get("league") or {}
            drops = [k for k, r in league.items() if k in prev and k != "current"
                     and r["winrate"] < prev[k]["winrate"] - 0.12]
            vs_cur = league.get("current", {}).get("winrate", 0.5)
            keep = not drops and (vs_cur >= 0.45 or mets["q"]["logloss"] < old.info["metrics"]["q"]["logloss"])
            reason = (f"기존 모델 상대 {vs_cur:.0%}" + (f", 하락: {drops}" if drops else ""))
        ar = league.get("heuristic", {})
        if keep:
            p = M.save(gt, q, v, d[6], exclude, mets, dex,
                       extra={"arena": ar, "league": league, "beta": beta, "betaSearch": beta_tab})
            hist = M.MODELS / "history"
            hist.mkdir(exist_ok=True)
            (hist / f"{gt}_{int(time.time())}.json").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            for old_f in sorted(hist.glob(f"{gt}_*.json"))[:-8]:
                old_f.unlink()
            print(f"[{gt}] 저장: {p.relative_to(ROOT)} ({reason})")
        else:
            print(f"[{gt}] 교체하지 않음 ({reason})")
        tmp.unlink(missing_ok=True)
        results[gt] = {"metrics": mets, "arena": ar, "league": league, "kept": keep, "beta": beta,
                       "decisions": int(len(d[2])), "games": d[6]["games"]}
        _history({"when": now(), "type": "train", "gameType": gt, **results[gt]})
    return results


# ======================================================================== 변수 제거 실험
def cmd_ablation(a, dex=None, data=None):
    dex = dex or Dex()
    types = ["singles", "doubles"] if a.type == "both" else [a.type]
    led = F.sync_ledger()
    for gt in types:
        d = data or load_data(dex, gt)
        seeds = list(range(a.seeds))
        base = []
        for s in seeds:
            _, _, m = M.train_q_v(d, seed=s, epochs=a.epochs)
            base.append(m["q"]["logloss"])
        b = float(np.mean(base))
        print(f"[{gt}] 전체 변수 검증 손실 {b:.4f} (시드 {len(seeds)}개)")
        for g in F.REGISTRY + F.PART_GROUPS:
            ls = []
            for s in seeds:
                _, _, m = M.train_q_v(d, exclude=(g.key,), seed=s, epochs=a.epochs)
                ls.append(m["q"]["logloss"])
            w = float(np.mean(ls))
            delta = w - b   # +: 빼면 나빠짐 = 쓸모 있음
            e = led["groups"][g.key]
            test = {"when": now(), "gameType": gt, "games": d[6]["games"], "decisions": int(len(d[2])),
                    "loss_with": round(b, 5), "loss_without": round(w, 5), "delta": round(delta, 5)}
            e["tests"].append(test)
            e["tests"] = e["tests"][-20:]
            e["status"] = _verdict(e)
            print(f"  {g.key:11s} {g.ko:10s} 빼면 {delta:+.4f} → {e['status']}")
        led["history"].append({"when": now(), "gameType": gt, "baseline": round(b, 5), "games": d[6]["games"]})
    F.save_ledger(led)
    write_features_md(led)
    return led


def _verdict(e):
    """최근 실험들로 채택/후보/제외 판정. 노이즈를 감안해 일관될 때만 바꾼다."""
    ts = e["tests"][-4:]
    if not ts:
        return "candidate"
    ds = [t["delta"] for t in ts]
    if all(d > 0.0005 for d in ds[-2:]) and len(ds) >= 1 and np.mean(ds) > 0.0008:
        return "adopted"
    if len(ds) >= 2 and all(d < -0.0015 for d in ds[-2:]):
        return "rejected"
    return e.get("status") if e.get("status") in ("adopted",) and np.mean(ds) > 0 else "candidate"


STATUS_KO = {"adopted": "채택", "candidate": "후보 (효과 확인 중)", "rejected": "제외"}


def write_features_md(led):
    lines = ["# 변수 장부", "",
             "AI 가 판단에 쓰는 변수 묶음과, 셀프테스트(제거 실험)로 확인한 효과를 기록합니다.",
             "**빼면 손실 증가**가 클수록 그 변수가 승패 예측에 도움이 된다는 뜻입니다 (검증 로그손실, 판 단위 분할).",
             "", "자동 생성: `python tools/train.py ablation` — 직접 고치지 마세요.", "",
             "| 묶음 | 이름 | 종류 | 칸 | 추가일 | 상태 | 최근 실험 (빼면 손실 증가) | 실험 데이터 |",
             "|---|---|---|---|---|---|---|---|"]
    for k, e in led["groups"].items():
        t = e["tests"][-1] if e["tests"] else None
        lines.append(f"| `{k}` | {e['ko']} | {'상태' if e['kind'] == 'state' else '행동'} | {e['size']} | {e['added']} | "
                     f"{STATUS_KO.get(e['status'], e['status'])} | "
                     + (f"{t['delta']:+.4f} ({t['gameType']})" if t else "-") + " | "
                     + (f"{t['games']}판 / {t['decisions']}결정" if t else "-") + " |")
    lines += ["", "## 다음에 넣을 변수 후보 (대전 후 분석)", ""]
    sugg = led.get("suggestions") or []
    if not sugg:
        lines.append("_아직 없음 — `python tools/train.py analyze` 를 돌리면 채워집니다._")
    for s in sugg[-30:]:
        lines.append(f"- **{s['name']}** — {s['why']} (근거: {s['evidence']}, {s['when'][:10]})"
                     + (" ✅ 반영됨" if s.get("done") else ""))
    lines += ["", "## 변수 묶음 설명", ""]
    for g in F.REGISTRY + F.PART_GROUPS:
        lines.append(f"- `{g.key}` **{g.ko}** — {g.desc}")
    FEATURES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ======================================================================== 대전 후 분석
def cmd_analyze(a, dex=None):
    """모델이 크게 틀린 결정(이길 거라 했는데 졌다 / 반대)에 자주 끼는 상황을 찾아 변수 후보로 올린다."""
    from engine.analysis import analyze_surprises
    dex = dex or Dex()
    led = F.sync_ledger()
    found = []
    for gt in ("singles", "doubles"):
        m = M.load(gt)
        files = game_files(gt)[-a.files:] if a.files else game_files(gt)
        if not m or not files:
            continue
        found += analyze_surprises(dex, m, SP.iter_games(files), gt, max_games=a.max_games)
    have = {s["name"] for s in led.get("suggestions", [])}
    for s in found:
        if s["name"] not in have:
            s["when"] = now()
            led.setdefault("suggestions", []).append(s)
            print(f"  후보: {s['name']} — {s['why']} ({s['evidence']})")
    F.save_ledger(led)
    write_features_md(led)
    return found


# ======================================================================== 반복
def cmd_loop(a):
    dex = Dex()
    for r in range(a.rounds):
        print(f"===== 반복 {r + 1}/{a.rounds} =====")
        a.seed = int(time.time()) % 100000 + r
        a.policy = "league" if any(M.load(t) for t in ("singles", "doubles")) else "heuristic"
        cmd_selfplay(a, dex)
        cmd_train(a, dex)
        if (r + 1) % max(1, a.ablation_every) == 0:
            cmd_ablation(a, dex)
            cmd_analyze(a, dex)
    cmd_report(a)


def _history(entry):
    REPORTS.mkdir(exist_ok=True)
    h = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.is_file() else []
    h.append(entry)
    HISTORY.write_text(json.dumps(h[-500:], ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_report(a=None):
    REPORTS.mkdir(exist_ok=True)
    h = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.is_file() else []
    lines = ["# 학습 기록", "", "상대별 승률은 새 모델이 학습하지 않은 기본 AI·지금 모델과 붙은 결과입니다.", "",
             "| 시각 (UTC) | 형식 | 학습 판수 | 결정 수 | Q 검증 손실 (기준) | 정확도 | 선생님 AI | 최대 데미지 AI | 쇼다운 기본 AI | 지금 모델 | 교체 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in h[-60:]:
        if e.get("type") != "train":
            continue
        q = e["metrics"]["q"]
        lg = e.get("league") or {}
        cell = lambda k: (f"{lg[k]['wins']}/{lg[k]['played']} ({lg[k]['winrate']:.0%})" if k in lg else "-")
        lines.append(f"| {e['when'][:16]} | {e['gameType']} | {e['games']} | {e['decisions']} | "
                     f"{q['logloss']} ({q['baseline']}) | {q['acc']:.1%} | {cell('heuristic')} | {cell('max_damage')} | "
                     f"{cell('showdown_random')} | {cell('current')} | {'예' if e.get('kept') else '아니오'} |")
    (REPORTS / "training.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("reports/training.md 갱신")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selfplay", "train", "ablation", "analyze", "loop", "report"):
        p = sub.add_parser(name)
        p.add_argument("--type", default="singles", choices=["singles", "doubles", "both"])
        p.add_argument("--games", type=int, default=300)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--policy", default="league", choices=["heuristic", "model", "league"])
        p.add_argument("--explore", type=float, default=0.1)
        p.add_argument("--arena", type=int, default=40)
        p.add_argument("--seeds", type=int, default=2)
        p.add_argument("--epochs", type=int, default=25)
        p.add_argument("--rounds", type=int, default=2)
        p.add_argument("--ablation-every", type=int, default=1)
        p.add_argument("--files", type=int, default=0)
        p.add_argument("--max-games", type=int, default=400)
    a = ap.parse_args()
    {"selfplay": cmd_selfplay, "train": cmd_train, "ablation": cmd_ablation, "analyze": cmd_analyze,
     "loop": cmd_loop, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
