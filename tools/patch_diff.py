#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""패치 감지 — 지난 학습 때의 데이터와 지금 데이터를 비교해 '무엇이 바뀌었는지'와 '어떤 포켓몬 위주로
다시 대전해 봐야 하는지'를 정한다.

  python tools/patch_diff.py                      지난 스냅숏(userdata/data_snapshot) ↔ 지금 데이터
  python tools/patch_diff.py --old A --new B      두 데이터 폴더 비교
  python tools/patch_diff.py --snapshot           지금 데이터를 스냅숏으로 저장 (새 신경망을 채택할 때 자동)

결과: reports/patch_diff.md (사람용) · reports/patch_diff.json (focus: 종 → 가중치, 밤샘 학습이 읽음)

바뀐 것의 종류에 따라 하는 일 (docs/STUDY.md 참고)
  - 수치(위력·명중·종족값·타입·특성) → 엔진 계산 변수가 자동으로 새 값을 쓴다. 바뀐 포켓몬이 들어간 팀을
    더 자주 대전시켜(focus) 미세조정한다.
  - 새 포켓몬·기술·도구·특성 → 마찬가지 + 새 효과가 데미지 엔진에 있는지 대조 셀프테스트로 확인.
  - 규칙(사용 가능 포켓몬) → 팀 생성 풀이 바뀐다. 메타 통계가 쌓일 때까지 learnset 기반 추정.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SNAP = ROOT / "userdata" / "data_snapshot"
REPORTS = ROOT / "reports"
FILES = ("species", "moves", "items", "abilities", "learnsets", "formats", "typechart", "usage")
MOVE_KEYS = ("type", "category", "basePower", "accuracy", "pp", "priority", "target", "secondary", "secondaries",
             "critRatio", "flags", "callbacks")
SPECIES_KEYS = ("types", "baseStats", "abilities", "requiredItem")
ITEM_KEYS = ("megaStone", "fling", "naturalGift", "callbacks")
ABILITY_KEYS = ("callbacks", "flags")


def load(d: Path):
    out = {}
    for f in FILES:
        p = d / f"{f}.json"
        out[f] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return out


def _cmp(a: dict, b: dict, keys):
    ch = {}
    for k in keys:
        if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True):
            ch[k] = [a.get(k), b.get(k)]
    return ch


def diff(old: dict, new: dict) -> dict:
    res = {}
    for kind, keys in (("species", SPECIES_KEYS), ("moves", MOVE_KEYS), ("items", ITEM_KEYS), ("abilities", ABILITY_KEYS)):
        o, n = old.get(kind, {}), new.get(kind, {})
        res[kind] = {
            "added": sorted(set(n) - set(o)), "removed": sorted(set(o) - set(n)),
            "changed": {k: c for k in sorted(set(o) & set(n)) if (c := _cmp(o[k], n[k], keys))},
        }
    lo, ln = old.get("learnsets", {}), new.get("learnsets", {})
    res["learnsets"] = {k: {"added": sorted(set(ln.get(k, [])) - set(lo.get(k, []))),
                            "removed": sorted(set(lo.get(k, [])) - set(ln.get(k, [])))}
                        for k in sorted(set(lo) | set(ln)) if set(lo.get(k, [])) != set(ln.get(k, []))}
    res["typechart"] = json.dumps(old.get("typechart"), sort_keys=True) != json.dumps(new.get("typechart"), sort_keys=True)
    return res


def focus(res: dict, new: dict) -> dict:
    """바뀐 것과 관련된 포켓몬 → 대전 생성 가중치 (1 = 보통)."""
    w = {}

    def bump(sid, x):
        if sid in new["species"]:
            w[sid] = max(w.get(sid, 1.0), x)
    for sid in res["species"]["added"]:
        bump(sid, 4.0)
    for sid in res["species"]["changed"]:
        bump(sid, 3.0)
    for sid, ch in res["learnsets"].items():
        bump(sid, 2.0 if ch["added"] else 1.5)
    changed_moves = set(res["moves"]["added"]) | set(res["moves"]["changed"])
    changed_abil = set(res["abilities"]["added"]) | set(res["abilities"]["changed"])
    changed_items = set(res["items"]["added"]) | set(res["items"]["changed"])
    learn = new.get("learnsets", {})
    usage = new.get("usage") or {}
    for sid, sp in new["species"].items():
        # 바뀐 기술을 실제로 많이 쓰는 포켓몬은 크게, 배우기만 하는 포켓몬은 조금
        used = set()
        for gt in ("singles", "doubles"):
            u = ((usage.get(gt) or {}).get("pokemon") or {}).get(sid) or {}
            used |= {m for m, p in (u.get("moves") or {}).items() if p >= 0.1}
        if changed_moves & used:
            bump(sid, 1.8)
        elif changed_moves and changed_moves & set(learn.get(sid, [])):
            bump(sid, 1.15)
        if changed_abil and changed_abil & {a.lower().replace(" ", "").replace("-", "") for a in sp.get("abilities", {}).values()}:
            bump(sid, 2.0)
        if sp.get("requiredItem") and sp["requiredItem"].lower().replace(" ", "").replace("-", "") in changed_items:
            bump(sid, 2.5)
    return dict(sorted(w.items(), key=lambda kv: -kv[1]))


def write_report(res, fw, old_v, new_v):
    REPORTS.mkdir(exist_ok=True)
    L = [f"# 패치 감지 ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC)", "",
         f"- 비교: {old_v} → {new_v}", ""]
    total = 0
    names = {"species": "포켓몬", "moves": "기술", "items": "도구", "abilities": "특성"}
    for kind, ko in names.items():
        r = res[kind]
        n = len(r["added"]) + len(r["removed"]) + len(r["changed"])
        total += n
        if not n:
            continue
        L.append(f"## {ko} — 추가 {len(r['added'])} · 삭제 {len(r['removed'])} · 변경 {len(r['changed'])}")
        if r["added"]:
            L.append("- 추가: " + ", ".join(r["added"][:60]))
        if r["removed"]:
            L.append("- 삭제: " + ", ".join(r["removed"][:60]))
        for k, ch in list(r["changed"].items())[:60]:
            L.append(f"- {k}: " + "; ".join(f"{f} {json.dumps(a, ensure_ascii=False)[:60]} → {json.dumps(b, ensure_ascii=False)[:60]}"
                                         for f, (a, b) in ch.items()))
        L.append("")
    if res["learnsets"]:
        total += len(res["learnsets"])
        L.append(f"## 배우는 기술 변경 — {len(res['learnsets'])}마리")
        for k, ch in list(res["learnsets"].items())[:40]:
            L.append(f"- {k}: +{', '.join(ch['added'][:8])} −{', '.join(ch['removed'][:8])}")
        L.append("")
    if res["typechart"]:
        total += 1
        L.append("## 타입 상성표가 바뀌었습니다 — 모든 포켓몬을 다시 봐야 합니다\n")
    L.append("## 다음 학습에서 더 자주 대전시킬 포켓몬")
    L.append(", ".join(f"{k}×{v:g}" for k, v in list(fw.items())[:40]) or "(없음)")
    if not total:
        L = L[:3] + ["바뀐 것이 없습니다."]
    (REPORTS / "patch_diff.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (REPORTS / "patch_diff.json").write_text(json.dumps({"from": old_v, "to": new_v, "changes": total, "focus": fw},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    return total


def snapshot(data_dir: Path):
    SNAP.mkdir(parents=True, exist_ok=True)
    for f in FILES + ("meta",):
        p = data_dir / f"{f}.json"
        if p.is_file():
            shutil.copy2(p, SNAP / p.name)


def run(old_dir=None, new_dir=None, log=print) -> dict:
    from engine.dex import DATA_DIR
    new_dir = Path(new_dir or DATA_DIR)
    old_dir = Path(old_dir or SNAP)
    if not (old_dir / "species.json").is_file():
        snapshot(new_dir)
        log("  패치 감지: 처음이라 지금 데이터를 기준으로 저장했습니다.")
        return {"changes": 0, "focus": {}}
    old, new = load(old_dir), load(new_dir)
    res = diff(old, new)
    fw = focus(res, new)
    ver = lambda d: json.loads((d / "meta.json").read_text(encoding="utf-8")).get("builtAt") if (d / "meta.json").is_file() else "?"
    n = write_report(res, fw, ver(old_dir), ver(new_dir))
    log(f"  패치 감지: 바뀐 항목 {n}개 · 집중 대전 포켓몬 {len(fw)}마리 (reports/patch_diff.md)")
    return {"changes": n, "focus": fw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old")
    ap.add_argument("--new")
    ap.add_argument("--snapshot", action="store_true")
    a = ap.parse_args()
    if a.snapshot:
        from engine.dex import DATA_DIR
        snapshot(Path(a.new or DATA_DIR))
        print("스냅숏 저장:", SNAP)
        return
    r = run(a.old, a.new)
    print(json.dumps({"changes": r["changes"], "focus": dict(list(r["focus"].items())[:20])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
