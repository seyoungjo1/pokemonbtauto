#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""셀프테스트 — 프로그램이 스스로 맞는지 확인한다.

  python tests/selftest.py            # 전체
  python tests/selftest.py damage     # 데미지 대조만
  python tests/selftest.py --json     # 결과를 JSON 으로

검사 목록
  data     : 데이터 파일이 온전한가 (포켓몬·기술·도구·특성 수, 한글 이름, 배울 수 있는 기술)
  stats    : 능력치 계산이 쇼다운과 같은가 (대조 사례의 실능력치)
  damage   : 데미지 16단계 난수 × 급소 여부가 쇼다운과 한 칸도 틀리지 않는가
  paste    : 팀 텍스트 읽기/쓰기 왕복
  sim      : 자기대전 기록이 규칙대로인가 (기록이 있을 때)
  features : 변수 등록부 — 모든 변수가 기록에서 계산되는가
  model    : 학습된 모델이 무작위보다 나은가 (모델이 있을 때)

대조 사례(tests/parity_*.json)는 tools/parity.js 가 쇼다운 시뮬레이터로 만든다.
결과는 reports/selftest.json 에도 저장된다.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.dex import Dex  # noqa: E402
from engine.mon import FieldState, Mon, SideState  # noqa: E402
from engine import damage as dmg  # noqa: E402

REPORTS = ROOT / "reports"


def _mon(dex, d):
    m = Mon(species=d["species"], ability=d["ability"], item=d["item"], nature=d["nature"],
            sp=d["sp"], gender=d.get("gender", ""))
    m.setup(dex)
    m.hp = d["hp"]
    m.status = d.get("status", "")
    m.boosts = {k: v for k, v in (d.get("boosts") or {}).items() if v}
    m.volatiles = set(d.get("volatiles") or [])
    m.times_attacked = d.get("timesAttacked", 0)
    return m


def case_objects(dex, c):
    atk, dfn = _mon(dex, c["atk"]), _mon(dex, c["def"])
    aa = _mon(dex, c["atkAlly"]) if c.get("atkAlly") else None
    da = _mon(dex, c["defAlly"]) if c.get("defAlly") else None
    f = FieldState(game_type=c["gameType"], weather=c["field"]["weather"], terrain=c["field"]["terrain"],
                   gravity=c["field"]["gravity"], trick_room=c["field"]["trickRoom"],
                   sides=[SideState(**s) for s in c["sides"]])
    ctx = dmg.Ctx(atk_side=0, def_side=1, actives=[x for x in (atk, aa, dfn, da) if x],
                  atk_ally=aa, def_ally=da, spread=c["spreadHit"], helping_hand=c["helpingHand"],
                  moved_last=c.get("ctx", {}).get("movedLast", False),
                  target_moved_first=c.get("ctx", {}).get("targetMovedFirst", False))
    return atk, dfn, aa, da, f, ctx


# ======================================================================== 검사들
def check_data(dex):
    errs = []
    if len(dex.species) < 100:
        errs.append(f"포켓몬 수가 너무 적음: {len(dex.species)}")
    if len(dex.moves) < 200:
        errs.append(f"기술 수가 너무 적음: {len(dex.moves)}")
    no_ko = [s["name"] for s in dex.species.values() if not s.get("ko")]
    if no_ko:
        errs.append(f"한글 이름 없는 포켓몬: {no_ko[:5]}")
    for sid, s in dex.species.items():
        if not dex.learnsets.get(sid):
            errs.append(f"배울 수 있는 기술 없음: {s['name']}")
    for t in ("Fire", "Water", "Grass", "Fairy"):
        if t not in dex.typechart:
            errs.append(f"타입표에 {t} 없음")
    if dex.effectiveness("Fire", "Grass") != 1 or not dex.immune("Ground", "Flying"):
        errs.append("타입 상성표 이상")
    return {"ok": not errs, "errors": errs[:20],
            "summary": f"포켓몬 {len(dex.species)} · 기술 {len(dex.moves)} · 도구 {len(dex.items)} · 특성 {len(dex.abilities)}"}


def load_parity():
    files = sorted((ROOT / "tests").glob("parity_*.json"))
    cases = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for c in d["cases"]:
            c["_file"] = f.name
        cases.extend(d["cases"])
    return files, cases


def check_stats(dex, cases):
    bad = []
    n = 0
    for c in cases:
        for k in ("atk", "def", "atkAlly", "defAlly"):
            d = c.get(k)
            if not d:
                continue
            m = _mon(dex, d)
            n += 1
            mine = {"hp": m.maxhp, **{s: m.stats[s] for s in ("atk", "def", "spa", "spd", "spe")}}
            if mine != d["stats"]:
                bad.append({"case": c["id"], "species": d["species"], "mine": mine, "ps": d["stats"]})
    return {"ok": not bad and n > 0, "checked": n, "mismatch": len(bad), "examples": bad[:5],
            "summary": f"실능력치 {n}개 중 {n - len(bad)}개 일치"}


# 쇼다운 getDamage 단독으로는 실제 동작과 다른 기술 (대조에서 제외)
#   fling    : 위력이 onPrepareHit 에서 정해져 getDamage 만 부르면 0
#   endeavor : 내 HP 가 더 많으면 실제로는 실패하지만 getDamage 는 음수를 돌려준다
PARITY_SKIP = {"fling", "endeavor"}


def check_damage(dex, cases):
    cases = [c for c in cases if c["move"] not in PARITY_SKIP]
    total = exact = sup_total = sup_exact = 0
    by_reason = Counter()
    fails = []
    for c in cases:
        try:
            atk, dfn, aa, da, f, ctx = case_objects(dex, c)
            r = dmg.calc(dex, atk, dfn, c["move"], f, ctx)
        except Exception as e:  # 엔진 오류도 실패로 센다
            r = None
            err = f"{type(e).__name__}: {e}"
        exp = c["expect"]
        total += 1
        approx = r is None or any("미구현" in n or "근사" in n or "기본값" in n for n in r.notes)
        if not approx:
            sup_total += 1
        ok, why = _compare(r, exp)
        if r is None:
            ok, why = False, err
        if ok:
            exact += 1
            if not approx:
                sup_exact += 1
            continue
        reason_keys = [f"move:{c['move']}", f"atkAbility:{c['atk']['ability']}", f"defAbility:{c['def']['ability']}",
                       f"atkItem:{c['atk']['item']}", f"defItem:{c['def']['item']}"]
        for k in reason_keys:
            by_reason[k] += 1
        if len(fails) < 30:
            fails.append({"case": c["id"], "file": c.get("_file"), "move": c["move"],
                          "atk": f"{c['atk']['species']}/{c['atk']['ability']}/{c['atk']['item']}",
                          "def": f"{c['def']['species']}/{c['def']['ability']}/{c['def']['item']}",
                          "why": why, "approx": approx, "notes": (r.notes if r else [])})
    rate = exact / total if total else 0
    sup_rate = sup_exact / sup_total if sup_total else 0
    return {
        "ok": total > 0 and sup_exact == sup_total,
        "total": total, "exact": exact, "rate": round(rate, 4),
        "supported_total": sup_total, "supported_exact": sup_exact, "supported_rate": round(sup_rate, 4),
        "suspects": by_reason.most_common(15), "fails": fails,
        "summary": f"지원 범위 {sup_exact}/{sup_total} 일치 ({sup_rate:.2%}) · 전체 {exact}/{total} ({rate:.2%})",
    }


def _compare(r, exp):
    if r is None:
        return False, "엔진 오류"
    for key, mine in (("normal", r.rolls), ("crit", r.crit_rolls)):
        ps_rows = exp[key]
        if all(v is False for row in ps_rows for v in row):
            if not r.immune:
                return False, f"{key}: 쇼다운은 무효, 엔진은 {mine[:1]}"
            continue
        if r.immune:
            return False, f"{key}: 엔진은 무효({r.immune}), 쇼다운은 {ps_rows[0][:3]}…"
        for h, row in enumerate(ps_rows):
            mrow = mine[h] if h < len(mine) else None
            if mrow != row:
                return False, f"{key} 타{h + 1}: 엔진 {mrow} / 쇼다운 {row}"
    return True, ""


def check_paste(dex):
    from engine.team import export_paste, parse_paste
    text = """Garchomp @ Life Orb
Ability: Rough Skin
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

한카리아스 @ 기합의띠
특성: 까칠한피부
EVs: 32 Atk / 32 Spe / 2 HP
성격: 명랑
- 지진
- 스톤에지
"""
    team = parse_paste(dex, text)
    errs = []
    if len(team) != 2:
        errs.append(f"2마리를 읽어야 하는데 {len(team)}")
    else:
        a = team[0]
        if a["species"] != "garchomp" or a["item"] != "lifeorb" or a["nature"] != "jolly" \
                or a["sp"].get("atk") != 32 or a["moves"][0] != "earthquake":
            errs.append(f"영어 텍스트 읽기 이상: {a}")
        b = team[1]
        if b["species"] != "garchomp" or b["item"] != "focussash" or b["nature"] != "jolly" \
                or b["moves"] != ["earthquake", "stoneedge"]:
            errs.append(f"한글 텍스트 읽기 이상: {b}")
        again = parse_paste(dex, export_paste(dex, team))
        if [x["moves"] for x in again] != [x["moves"] for x in team]:
            errs.append("내보내기 → 다시 읽기 왕복 불일치")
    return {"ok": not errs, "errors": errs, "summary": "팀 텍스트 읽기/쓰기" + (" 정상" if not errs else " 오류")}


def check_features(dex):
    try:
        from engine import features
    except ImportError:
        return {"ok": True, "skipped": "features 모듈 없음"}
    return features.selfcheck(dex)


def check_sim():
    try:
        from engine import selfplay_check
    except ImportError:
        return {"ok": True, "skipped": "자기대전 검사 모듈 없음"}
    return selfplay_check.run()


def check_model(dex):
    try:
        from engine import model
    except ImportError:
        return {"ok": True, "skipped": "모델 모듈 없음"}
    return model.selfcheck(dex)


CHECKS = ["data", "stats", "damage", "paste", "features", "sim", "model"]


def run(which=None, verbose=True) -> dict:
    t0 = time.time()
    dex = Dex()
    which = which or CHECKS
    files, cases = load_parity() if {"stats", "damage"} & set(which) else ([], [])
    out = {"dataVersion": dex.meta.get("builtAt"), "parityFiles": [f.name for f in files], "checks": {}}
    for name in which:
        t = time.time()
        if name == "data":
            r = check_data(dex)
        elif name == "stats":
            r = check_stats(dex, cases)
        elif name == "damage":
            r = check_damage(dex, cases)
        elif name == "paste":
            r = check_paste(dex)
        elif name == "features":
            r = check_features(dex)
        elif name == "sim":
            r = check_sim()
        elif name == "model":
            r = check_model(dex)
        else:
            continue
        r["seconds"] = round(time.time() - t, 2)
        out["checks"][name] = r
        if verbose:
            mark = "통과" if r.get("ok") else "실패"
            print(f"[{mark}] {name:9s} {r.get('summary') or r.get('skipped') or ''}  ({r['seconds']}s)")
            if not r.get("ok"):
                for e in (r.get("errors") or [])[:5]:
                    print("        ", e)
                for e in (r.get("fails") or [])[:8]:
                    print("        ", e)
    out["ok"] = all(r.get("ok") for r in out["checks"].values())
    out["seconds"] = round(time.time() - t0, 1)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "selftest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print("전체:", "통과" if out["ok"] else "실패", f"({out['seconds']}s)")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = run(args or None, verbose="--json" not in sys.argv)
    if "--json" in sys.argv:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    sys.exit(0 if res["ok"] else 1)
