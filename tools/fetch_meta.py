#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""메타 데이터 수집 — 상대 세트 예측과 자기대전 팀에 쓰는 '실제로 많이 쓰이는 것'.

출처 (받을 수 있는 것만, 순서대로)
  1. 스모곤 월간 사용률 통계 (smogon.com/stats/<월>/chaos/<포맷>-<기준>.json)  — PC·Actions 에서
  2. pkmn/smogon 저장소의 사용률(data/stats) · 전문가 세트(data/sets) — GitHub 경유

결과: data/<규칙>/usage.json  (포켓몬별 사용률·기술·도구·특성·SP 배분·성격·같이 쓰는 포켓몬·선봉 비율)
      data/<규칙>/expert_sets.json (스모곤 전문가 세트, 싱글·더블별)

  python tools/fetch_meta.py [--out data/champions] [--cache build/meta]
"""
import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.dex import Dex, to_id  # noqa: E402

PKMN = "https://raw.githubusercontent.com/pkmn/smogon/main/data/"
SMOGON = "https://www.smogon.com/stats/"
# 싱글(6→3) · 더블(6→4) 에 쓸 통계 후보 — 앞에 있는 것이 우선
STAT_FORMATS = {
    "singles": ["gen9championsbssregmc", "gen9championsbssregmb", "gen9championsbssregma",
                "gen9championsbattlestadiumsingles", "gen9championsou"],
    "doubles": ["gen9championsvgc2026regmc", "gen9championsvgc2026regmb", "gen9championsvgc2026regma",
                "gen9championsvgc2026", "gen9championsou"],
}
SET_FILES = {"singles": ["championsbattlestadiumsingles", "championsou", "champions"],
             "doubles": ["championsvgc2026", "champions"]}
NATURE_STAT = {}


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "pokemonbtauto-meta"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def try_json(url):
    try:
        return json.loads(get(url).decode("utf-8"))
    except Exception:
        return None


def smogon_latest(fmt):
    """smogon.com/stats 의 최신 월 chaos 통계 (접속되면)."""
    try:
        idx = get(SMOGON, 20).decode("utf-8", "ignore")
    except Exception:
        return None, None
    months = sorted(set(re.findall(r'href="(\d{4}-\d{2}(?:-DLC\d)?)/"', idx)), reverse=True)
    for m in months[:3]:
        for cut in ("1630", "1500", "0"):
            d = try_json(f"{SMOGON}{m}/chaos/{fmt}-{cut}.json")
            if d:
                return d, f"smogon {m} {fmt}-{cut}"
    return None, None


def trim_stats(dex: Dex, raw: dict, source: str) -> dict:
    """chaos 형식 → 필요한 것만 (한 포켓몬당 상위 몇 개)."""
    mons = raw.get("data") or raw.get("pokemon") or {}
    out = {}
    for name, d in mons.items():
        sid = dex.find("species", name)
        if not sid:
            continue
        usage = d.get("usage")
        if isinstance(usage, dict):
            usage = usage.get("weighted", usage.get("real", 0))
        if usage is None:
            cnt = d.get("Raw count", 0)
            usage = cnt / max(1, raw.get("info", {}).get("number of battles", 1) * 6)

        def top(key, n, kind=None, norm=True):
            src = d.get(key) or d.get(key.capitalize()) or {}
            items = sorted(src.items(), key=lambda kv: -kv[1])
            tot = sum(v for _, v in items) or 1
            res = {}
            for k, v in items:
                if kind:
                    kid = dex.find(kind, k) if k not in ("Nothing", "") else ""
                    if kid is None:
                        continue
                    k = kid
                res[k] = round(v / tot if norm else v, 4)
                if len(res) >= n:
                    break
            return res
        spreads = []
        for k, v in sorted((d.get("spreads") or d.get("Spreads") or {}).items(), key=lambda kv: -kv[1])[:16]:
            m = re.match(r"^(\w+):([\d/]+)$", k)
            if not m:
                continue
            nums = [int(x) for x in m.group(2).split("/")]
            if len(nums) != 6:
                continue
            sp = {s: n for s, n in zip(("hp", "atk", "def", "spa", "spd", "spe"), nums) if n}
            if any(n > 32 for n in nums):   # 노력치(EV) 형식이면 SP 로 대략 환산
                sp = {s: min(32, round(n / 8)) for s, n in sp.items() if round(n / 8)}
            spreads.append({"nature": to_id(m.group(1)), "sp": sp, "p": round(v, 4)})
        tot_sp = sum(s["p"] for s in spreads) or 1
        for s in spreads:
            s["p"] = round(s["p"] / tot_sp, 4)
        moves_raw = d.get("moves") or d.get("Moves") or {}
        tot_m = sum(moves_raw.values()) or 1
        moves = {}
        for k, v in sorted(moves_raw.items(), key=lambda kv: -kv[1]):
            mid = dex.find("moves", k)
            if mid:
                moves[mid] = round(min(1.0, 4 * v / tot_m), 4)   # '이 기술을 가질 확률' (4칸 기준)
            if len(moves) >= 24:
                break
        lead = d.get("lead")
        if isinstance(lead, dict):
            lead = lead.get("weighted", lead.get("real"))
        out[sid] = {
            "usage": round(float(usage or 0), 5), "lead": round(float(lead or 0), 5),
            "moves": moves, "items": top("items", 14, "items"), "abilities": top("abilities", 4, "abilities"),
            "spreads": spreads, "teammates": top("teammates", 15, "species", norm=False),
        }
    return {"source": source, "pokemon": out}


def convert_sets(dex: Dex, raw: dict) -> dict:
    out = {}
    for name, sets in (raw or {}).items():
        sid = dex.find("species", name)
        if not sid:
            continue
        lst = []
        for label, s in sets.items():
            moves = []
            for m in s.get("moves", []):
                opts = m if isinstance(m, list) else [m]
                ids = [dex.find("moves", x) for x in opts]
                moves.append([i for i in ids if i])
            items = s.get("item") if isinstance(s.get("item"), list) else [s.get("item")]
            abil = s.get("ability") if isinstance(s.get("ability"), list) else [s.get("ability")]
            nat = s.get("nature") if isinstance(s.get("nature"), list) else [s.get("nature")]
            evs = s.get("evs") if isinstance(s.get("evs"), dict) else ((s.get("evs") or [{}])[0])
            lst.append({"label": label, "moves": moves,
                        "items": [dex.find("items", i) for i in items if i and dex.find("items", i)],
                        "abilities": [dex.find("abilities", a) for a in abil if a and dex.find("abilities", a)],
                        "natures": [to_id(n) for n in nat if n], "sp": {k: v for k, v in (evs or {}).items() if v}})
        if lst:
            out[sid] = lst
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "champions"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    dex = Dex(out)
    usage = {"fetchedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for gt, fmts in STAT_FORMATS.items():
        got = None
        for f in fmts:
            raw, src = smogon_latest(f)
            if raw:
                got = trim_stats(dex, raw, src)
                break
            raw = try_json(PKMN + f"stats/{f}.json")
            if raw:
                got = trim_stats(dex, raw, f"pkmn/smogon stats/{f}")
                break
        usage[gt] = got
        print(f"[{gt}] 사용률: {got['source'] + ' · ' + str(len(got['pokemon'])) + '종' if got else '못 받음'}")
    sets = {}
    for gt, files in SET_FILES.items():
        merged = {}
        for f in files:
            raw = try_json(PKMN + f"sets/{f}.json")
            if raw:
                for sid, lst in convert_sets(dex, raw).items():
                    merged.setdefault(sid, [])
                    merged[sid] += [dict(x, source=f) for x in lst]
        sets[gt] = merged
        print(f"[{gt}] 전문가 세트: {len(merged)}종")
    import hashlib
    hist = out / "history"
    if usage.get("singles") or usage.get("doubles"):
        body = {k: v for k, v in usage.items() if k != "fetchedAt"}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        usage["metaVersion"] = digest
        old = out / "usage.json"
        prev = json.loads(old.read_text(encoding="utf-8")) if old.is_file() else {}
        if prev.get("metaVersion") != digest:
            # 메타가 바뀌었다 → 지난 통계는 history 에 보관
            if prev:
                hist.mkdir(exist_ok=True)
                stamp = (prev.get("fetchedAt") or "old")[:10]
                (hist / f"usage_{stamp}_{prev.get('metaVersion', 'x')}.json").write_text(
                    json.dumps(prev, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            old.write_text(json.dumps(usage, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"메타 갱신: {prev.get('metaVersion')} → {digest}")
        else:
            print(f"메타 변화 없음 ({digest})")
    else:
        print("통계를 하나도 받지 못해 기존 파일을 그대로 둡니다.")
    if sets["singles"] or sets["doubles"]:
        (out / "expert_sets.json").write_text(json.dumps(sets, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
