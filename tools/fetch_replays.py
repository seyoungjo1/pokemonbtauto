#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""쇼다운 기보(리플레이) 받기 → 해석 → 모방학습.

  python tools/fetch_replays.py fetch  [--pages 20] [--formats bss,vgc]   리플레이 받기 (새것만)
  python tools/fetch_replays.py parse                                     받은 원본 → 학습용 판 기록
  python tools/fetch_replays.py train  [--type singles] [--epochs 20]     기보만으로 신경망 학습(모방) + 평가
  python tools/fetch_replays.py all                                       위 셋을 차례로

원본 로그(userdata/replays/raw/)를 그대로 보관한다. 해석기나 학습 변수가 바뀌어도(패치·새 기믹)
인터넷에서 다시 받을 필요 없이 원본에서 다시 만든다.

쇼다운 리플레이 서버 (공개 API)
  https://replay.pokemonshowdown.com/search.json?format=<형식>&before=<업로드시각>   최근 51개씩
  https://replay.pokemonshowdown.com/<id>.json                                        한 판 (log 포함)
"""
import argparse
import glob
import gzip
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "https://replay.pokemonshowdown.com"
RDIR = ROOT / "userdata" / "replays"
RAW = RDIR / "raw"
GAMES = RDIR / "games"
INDEX = RDIR / "index.json"
UA = "pokemonbtauto-replay-fetcher (personal research; github.com/seyoungjo1/pokemonbtauto)"


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        return json.loads(r.read().decode("utf-8"))


def champion_formats(kinds=("bss", "vgc")):
    from engine.dex import Dex
    dex = Dex()
    out = []
    for f in dex.formats:
        fid = f["id"]
        if fid.startswith("gen9champions") and any(k in fid for k in kinds):
            out.append(fid)
    return out


def load_index():
    if INDEX.is_file():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    return {"seen": {}, "newest": {}}


def save_index(ix):
    RDIR.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(ix, ensure_ascii=False), encoding="utf-8")


def fetch(formats, pages=20, delay=0.4, min_rating=0, log=print):
    """형식마다 최신 리플레이부터 거슬러 받는다. 이미 받은 것은 건너뛴다."""
    ix = load_index()
    RAW.mkdir(parents=True, exist_ok=True)
    total = 0
    for fid in formats:
        seen = set(ix["seen"].get(fid, []))
        newest = ix["newest"].get(fid, 0)
        before = None
        got = 0
        out = RAW / f"{fid}_{int(time.time())}.jsonl.gz"
        fo = None
        try:
            for _ in range(pages):
                q = {"format": fid}
                if before:
                    q["before"] = before
                try:
                    lst = _get(f"{BASE}/search.json?{urllib.parse.urlencode(q)}")
                except Exception as e:
                    log(f"  {fid}: 목록 실패 ({type(e).__name__}: {e})")
                    break
                if not lst:
                    break
                stop = False
                for r in lst[:50]:
                    if r.get("private") or r.get("password"):
                        continue
                    if r["id"] in seen:
                        continue
                    if r.get("uploadtime", 0) <= newest:
                        stop = True
                    if (r.get("rating") or 0) < min_rating:
                        continue
                    try:
                        d = _get(f"{BASE}/{r['id']}.json")
                    except Exception as e:
                        log(f"  {r['id']}: 실패 ({type(e).__name__})")
                        continue
                    if fo is None:
                        fo = gzip.open(out, "wt", encoding="utf-8")
                    keep = {k: d.get(k) for k in ("id", "formatid", "format", "players", "rating", "uploadtime", "log")}
                    keep["rating"] = keep.get("rating") or r.get("rating")
                    fo.write(json.dumps(keep, ensure_ascii=False) + "\n")
                    seen.add(r["id"])
                    ix["newest"][fid] = max(ix["newest"].get(fid, 0), d.get("uploadtime") or r.get("uploadtime") or 0)
                    got += 1
                    time.sleep(delay)
                if stop or len(lst) <= 50:
                    break
                before = lst[min(49, len(lst) - 1)]["uploadtime"]
        finally:
            if fo:
                fo.close()
            ix["seen"][fid] = sorted(seen)
            save_index(ix)
        log(f"  {fid}: 새 리플레이 {got}개 (누적 {len(seen)}개)")
        total += got
    return total


def iter_raw(paths=None):
    for p in paths or sorted(glob.glob(str(RAW / "*.jsonl.gz"))):
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


def parse_all(min_rating=0, log=print):
    """원본 전부 → 형식별 판 기록 (다시 하면 덮어쓴다: 해석기·변수가 바뀌어도 원본에서 새로)."""
    from engine.dex import Dex
    from engine.replay_parse import ReplayParser
    dex = Dex()
    P = ReplayParser(dex)
    GAMES.mkdir(parents=True, exist_ok=True)
    outs, n_ok, n_bad = {}, 0, 0
    t0 = time.time()
    try:
        for r in iter_raw():
            if (r.get("rating") or 0) < min_rating:
                continue
            try:
                g = P.parse(r.get("log") or "", meta=r)
            except Exception:
                g = None
            if not g:
                n_bad += 1
                continue
            gt = g["gameType"]
            if gt not in outs:
                outs[gt] = gzip.open(GAMES / f"games_{gt}_replay.jsonl.gz.tmp", "wt", encoding="utf-8")
            outs[gt].write(json.dumps(g, ensure_ascii=False, separators=(",", ":")) + "\n")
            n_ok += 1
            if n_ok % 200 == 0:
                log(f"  해석 {n_ok}판 · {time.time() - t0:.0f}초")
    finally:
        for gt, fo in outs.items():
            fo.close()
            tmp = GAMES / f"games_{gt}_replay.jsonl.gz.tmp"
            tmp.replace(GAMES / f"games_{gt}_replay.jsonl.gz")
    log(f"  해석 완료: {n_ok}판 (못 읽음 {n_bad}) · 결정 {P.stats}")
    return n_ok


def replay_files(gt):
    return sorted(glob.glob(str(GAMES / f"games_{gt}_replay*.jsonl.gz")))


def train_imitation(gt, epochs=20, max_games=20000, log=print, eval_games=40):
    """기보만으로 정책망·가치망 학습 (사람의 선택을 따라 하기) → 기본 AI 들과 평가 → 더 나으면 채택."""
    from engine import nets, nets_torch, selfplay as SP
    from engine.dex import Dex
    dex = Dex()
    files = replay_files(gt)
    if not files:
        log("기보가 없습니다. 먼저 fetch · parse 를 하세요.")
        return None
    data = nets.build_listwise(SP.iter_games(files), dex, gt, max_games=max_games, log=log)
    if len(data["S"]) < 200:
        log("학습할 결정이 너무 적습니다.")
        return None
    tr, va = nets.split(data)
    cur = nets.load_net(gt)
    if cur is not None:   # 지금 신경망에서 이어서 (변수가 바뀌었으면 수술)
        net = nets.surgery(cur, log=log)
        nets.fit_new_norms(net, data)
        lr = 3e-4
    else:
        net = nets.PolicyValueNet(len(data["S"][0]), data["O"][0].shape[1])
        net.set_norm([data["S"][i] for i in tr[:20000]], [data["O"][i] for i in tr[:20000]])
        lr = 1e-3
    if nets_torch.available():
        nets_torch.fit_torch(net, data, tr, va, epochs=epochs, lr=lr, log=log)
    else:
        net.fit_numpy(data, epochs=min(epochs, 12), lr=lr, tr_idx=tr, val_idx=va, log=log)
    m = net.evaluate(data, va[:3000])
    net.info = {"metrics": m, "games": data["meta"]["games"], "decisions": len(data["S"]), "source": "replay-imitation"}
    cand = ROOT / "models" / f"net_{gt}_candidate.json"
    net.save(cand, data["meta"], extra={"source": "replay-imitation"})
    log(f"검증: 사람 선택 일치 {m['top1']:.1%} · 승패 예측 {m['valueAcc']:.1%} ({data['meta']['games']}판)")
    if eval_games:
        import tools.overnight as O
        league = O.evaluate(gt, cand, eval_games, log)
        best = league.get("best", {}).get("winrate")
        if best is None or best >= 0.55:
            dst = nets.path_for(gt)
            hist = ROOT / "models" / "history"
            hist.mkdir(parents=True, exist_ok=True)
            if dst.is_file():
                (hist / f"net_{gt}_{int(time.time())}.json").write_bytes(dst.read_bytes())
            cand.replace(dst)
            log("→ 기보 신경망 채택")
        else:
            log(f"→ 지금 신경망 유지 (기보 신경망이 이전 최고 상대 {best:.0%})")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "parse", "train", "all", "formats"])
    ap.add_argument("--pages", type=int, default=20, help="형식마다 받을 목록 쪽수 (한 쪽 50판)")
    ap.add_argument("--formats", default="bss,vgc", help="bss,vgc 또는 형식 id 를 쉼표로")
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--type", default="singles", choices=["singles", "doubles"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=40)
    a = ap.parse_args()
    kinds = [k.strip() for k in a.formats.split(",") if k.strip()]
    fmts = [k for k in kinds if k.startswith("gen9")] or champion_formats(tuple(kinds))
    if a.cmd == "formats":
        print("\n".join(fmts))
        return
    if a.cmd in ("fetch", "all"):
        print(f"리플레이 받기: {', '.join(fmts)}")
        fetch(fmts, pages=a.pages, min_rating=a.min_rating)
    if a.cmd in ("parse", "all"):
        parse_all(min_rating=a.min_rating)
    if a.cmd in ("train", "all"):
        train_imitation(a.type, epochs=a.epochs, eval_games=a.eval_games)


if __name__ == "__main__":
    main()
