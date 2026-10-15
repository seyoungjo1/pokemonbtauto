#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""토큰 배틀망 학습 명령 (PC 용 · 1단계) — 모든 명령은 JSON 결과를 돌려준다 (나중에 MCP 도구로 그대로 노출).

  python tools/train2.py bc       [--type singles] [--max-games 20000] [--epochs 20]   기보+자기대전으로 정책·가치 학습 → 후보
  python tools/train2.py eval     [--type singles] [--games 400]                        후보 vs 선생님 AI · 지금 토큰망 (SPRT)
  python tools/train2.py adopt    [--type singles]                                     게이트 통과한 후보를 models/battle_<형식>.json 으로
  python tools/train2.py all      (bc → eval → adopt)
  python tools/train2.py status                                                        지금 모델·데이터 현황
  --tiny : 샌드박스용 작은 설정 (판 수·망 크기·판 수 축소)

학습 장치: PyTorch(CUDA) 가 있으면 engine/battlenet_torch.fit_torch, 없으면 numpy(engine/battlenet.fit).
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GAMES = ROOT / "userdata" / "selfplay"
LOGF = ROOT / "userdata" / "train2.log"


def _log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGF.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def game_files(gt, replay_share=0.3, max_games=20000):
    """학습 판 = 사람 기보(최대 replay_share) + 최근 자기대전·나와 둔 대전."""
    from tools.fetch_replays import replay_files
    files = sorted(glob.glob(str(GAMES / "**" / f"games_{gt}_*.jsonl.gz"), recursive=True), key=os.path.getmtime)[::-1]
    return replay_files(gt), files, int(max_games * replay_share)


def iter_mixed(gt, replay_share, max_games):
    from engine import selfplay as SP
    rf, files, n_rep = game_files(gt, replay_share, max_games)
    k = 0
    for g in SP.iter_games(rf):
        if k >= n_rep:
            break
        if g.get("gameType") == gt:
            k += 1
            yield g
    yield from SP.iter_games(files)


def cmd_bc(a) -> dict:
    from engine import battlenet as BN
    from engine.dex import Dex
    from engine.tokens import Vocab
    dex = Dex()
    vb = Vocab(dex)
    t0 = time.time()
    max_games = 300 if a.tiny else a.max_games
    data = BN.build_data(iter_mixed(a.type, a.replay_share, max_games), dex, a.type, max_games=max_games, log=_log,
                         teachers=("search", "human"))
    if len(data["encs"]) < 2000:
        _log(f"  수읽기·사람 결정이 {len(data['encs'])}개뿐이라 선생님 AI 결정도 씁니다")
        data = BN.build_data(iter_mixed(a.type, a.replay_share, max_games), dex, a.type, max_games=max_games, log=_log)
    if len(data["encs"]) < 200:
        return {"ok": False, "error": f"학습할 결정이 {len(data['encs'])}개뿐입니다"}
    tr, va = BN.split(data)
    cur = BN.load(a.type)
    if cur is not None and cur.cfg.get("vocab_dims") == vb.dims and not a.fresh:
        net = cur
        lr, epochs = a.lr * 0.3, a.epochs
        _log("  지금 토큰망에서 이어서 학습")
    else:
        size = {"d_set": 48, "d_battle": 64, "set_layers": 1, "battle_layers": 1} if a.tiny else \
            {"d_set": a.d_set, "d_battle": a.d_battle, "set_layers": a.set_layers, "battle_layers": a.battle_layers}
        net = BN.BattleNet(vb.dims, heads=4, dropout=0.1, **size)
        lr, epochs = a.lr, a.epochs
    epochs = min(epochs, 3) if a.tiny else epochs
    _log(f"  결정 {len(data['encs'])}개 ({data['games']}판) · 파라미터 {sum(p.size for p in net.params()):,}")
    device = "cpu(numpy)"
    try:
        from engine import battlenet_torch as BT
        if BT.available() and not a.numpy:
            m = BT.fit_torch(net, data, tr, va, epochs=epochs, lr=lr, batch=a.batch, log=_log)
            device = m.get("device", "torch")
        else:
            m = BN.fit(net, data, tr, va, epochs=epochs, lr=lr, batch=min(a.batch, 32), log=_log)
    except ImportError:
        m = BN.fit(net, data, tr, va, epochs=epochs, lr=lr, batch=min(a.batch, 32), log=_log)
    net.info = {"metrics": {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}, "games": data["games"],
                "decisions": len(data["encs"]), "device": device, "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    cand = ROOT / "models" / f"battle_{a.type}_candidate.json"
    net.save(cand)
    _log(f"  검증: 정답 일치 {m['top1']:.1%} · 승패 예측 {m['valueAcc']:.1%} · {time.time() - t0:.0f}초 → {cand.name}")
    return {"ok": True, "candidate": str(cand), "metrics": net.info["metrics"], "games": data["games"], "decisions": len(data["encs"]),
            "device": device, "seconds": round(time.time() - t0)}


def _match(dex, gt, make_a, make_b, n, seed):
    """SP.run 으로 n 판 (자리 반씩) → (승, 판)."""
    from engine import selfplay as SP
    def fac(g):
        A, B = make_a(seed + g), make_b(seed + 1000 + g)
        return {"p1": A, "p2": B} if g % 2 == 0 else {"p1": B, "p2": A}
    p = SP.run(n, gt, out_dir=ROOT / "userdata" / "arena", seed=seed, policy_factory=fac, log=lambda *x: None)
    w = pl = 0
    for g in SP.iter_games([p]):
        if g["winner"] is None:
            continue
        idx = int(g["id"].split("_")[-1])
        pl += 1
        w += g["winner"] == ("p1" if idx % 2 == 0 else "p2")
    p.unlink(missing_ok=True)
    return w, pl


def cmd_eval(a) -> dict:
    from engine import battlenet as BN, benchmarks as BM
    from engine.dex import Dex
    from engine.policies import make_baseline
    dex = Dex()
    cand_p = ROOT / "models" / f"battle_{a.type}_candidate.json"
    if not cand_p.is_file():
        return {"ok": False, "error": "후보가 없습니다 (먼저 bc)"}
    cand = BN.BattleNet.load(cand_p)
    cur = BN.load(a.type)
    out = {"ok": True, "vs": {}}
    cap = 60 if a.tiny else a.games
    me = lambda s: BN.TokenNetPolicy(dex, cand, a.type, temperature=0.15, seed=s)
    opponents = [("heuristic", "선생님 AI", lambda s: make_baseline("heuristic", dex, s))]
    if cur is not None:
        opponents.append(("current", "지금 토큰망", lambda s: BN.TokenNetPolicy(dex, cur, a.type, temperature=0.15, seed=s)))
    for key, ko, mk in opponents:
        w = n = 0
        r = {"decision": "continue"}
        while n < cap and r["decision"] == "continue":
            w_, n_ = _match(dex, a.type, me, mk, 20, 5000 + n)
            w += w_
            n += n_
            r = BM.sprt(w, n - w)
        p, lo, hi = BM.wilson(w, n)
        out["vs"][key] = {"ko": ko, "wins": w, "played": n, "winrate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "sprt": r["decision"]}
        _log(f"  후보 vs {ko}: {w}/{n} ({p:.1%}, [{lo:.0%}–{hi:.0%}]) 순차검정 {r['decision']}")
    gate = out["vs"]["current"]["sprt"] == "accept" if cur is not None else out["vs"]["heuristic"]["winrate"] >= 0.5
    out["gate"] = bool(gate)
    (ROOT / "models" / f"battle_{a.type}_candidate.eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def cmd_adopt(a) -> dict:
    cand_p = ROOT / "models" / f"battle_{a.type}_candidate.json"
    ev_p = ROOT / "models" / f"battle_{a.type}_candidate.eval.json"
    if not cand_p.is_file():
        return {"ok": False, "error": "후보가 없습니다"}
    ev = json.loads(ev_p.read_text(encoding="utf-8")) if ev_p.is_file() else {}
    if not a.force and not ev.get("gate"):
        return {"ok": False, "error": "게이트를 통과하지 못했습니다 (eval 결과 gate=false). --force 로 강제 가능", "eval": ev}
    dst = ROOT / "models" / f"battle_{a.type}.json"
    hist = ROOT / "models" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        (hist / f"battle_{a.type}_{int(time.time())}.json").write_bytes(dst.read_bytes())
    cand_p.replace(dst)
    _log(f"  채택: {dst.name}")
    return {"ok": True, "model": str(dst), "eval": ev}


def cmd_status(a) -> dict:
    from engine import battlenet as BN
    out = {"ok": True}
    for gt in ("singles", "doubles"):
        p = BN.path_for(gt)
        out[gt] = {"model": p.is_file(), "info": (BN.load(gt).info if p.is_file() else None)}
    files = glob.glob(str(GAMES / "**" / "games_*.jsonl.gz"), recursive=True)
    out["selfplayFiles"] = len(files)
    try:
        from engine import battlenet_torch as BT
        out["torch"] = BT.available()
        out["device"] = BT.device_name() if BT.available() else "numpy"
    except Exception:
        out["torch"] = False
    return out


def run(cmd: str, **kw) -> dict:
    """파이썬·MCP 에서 부르는 진입점."""
    a = _args([cmd] + [f"--{k.replace('_', '-')}={v}" if not isinstance(v, bool) else f"--{k.replace('_', '-')}" for k, v in kw.items() if v is not False])
    return _dispatch(a)


def _args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bc", "eval", "adopt", "all", "status"])
    ap.add_argument("--type", default="singles", choices=["singles", "doubles"])
    ap.add_argument("--max-games", type=int, default=20000)
    ap.add_argument("--replay-share", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--d-set", type=int, default=128)
    ap.add_argument("--d-battle", type=int, default=192)
    ap.add_argument("--set-layers", type=int, default=2)
    ap.add_argument("--battle-layers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="지금 모델에서 이어서 하지 않고 새로")
    ap.add_argument("--numpy", action="store_true", help="토치가 있어도 numpy 로")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def _dispatch(a):
    if a.cmd == "all":
        r1 = cmd_bc(a)
        if not r1.get("ok"):
            return r1
        r2 = cmd_eval(a)
        r3 = cmd_adopt(a) if r2.get("gate") else {"ok": False, "error": "게이트 미통과 — 유지"}
        return {"ok": True, "bc": r1, "eval": r2, "adopt": r3}
    return {"bc": cmd_bc, "eval": cmd_eval, "adopt": cmd_adopt, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    a = _args(sys.argv[1:])
    res = _dispatch(a)
    print(json.dumps(res, ensure_ascii=False, indent=None if a.json else 1))
