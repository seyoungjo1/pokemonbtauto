# -*- coding: utf-8 -*-
"""인사이트 리포트 — 쌓인 대전 기록에서 사람이 읽을 수 있는 발견을 뽑는다.

  · 선출됐을 때 승률이 높은/낮은 포켓몬, 선봉 승률
  · 교체를 얼마나·언제 했고, 교체한 판의 승률
  · 메가진화 시점과 승률
  · 학습 진행 (상대별 승률 변화)
  · 파티 최적화에서 승률을 올린 변경들
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .dex import Dex, to_id

ROOT = Path(__file__).resolve().parent.parent


def species_of(details: str) -> str:
    return to_id(details.split(",")[0])


def analyze_games(dex: Dex, games, min_n=15) -> dict:
    brought = defaultdict(lambda: [0, 0])     # species → [판, 승]
    lead = defaultdict(lambda: [0, 0])
    switch_rate = []                           # (판 교체 비율, 승)
    mega_turn = defaultdict(lambda: [0, 0])
    n_games = 0
    turns = 0
    by_policy = defaultdict(lambda: [0, 0])
    for g in games:
        if g.get("winner") not in ("p1", "p2"):
            continue
        n_games += 1
        turns += g.get("turns", 0)
        for side in ("p1", "p2"):
            won = g["winner"] == side
            by_policy[g["policies"][side]][0] += 1
            by_policy[g["policies"][side]][1] += won
            decs = [d for d in g["decisions"] if d["side"] == side]
            seen = set()
            first = True
            sw = tot = 0
            for d in decs:
                req = d.get("req") or {}
                if d["kind"] != "preview":
                    for p in req.get("side") or []:
                        sid = species_of(p["details"])
                        base = to_id(dex.species.get(sid, {}).get("baseSpecies", sid))
                        if base not in seen:
                            seen.add(base)
                            brought[base][0] += 1
                            brought[base][1] += won
                    if first:
                        for p in req.get("side") or []:
                            if p.get("active"):
                                base = to_id(dex.species.get(species_of(p["details"]), {}).get("baseSpecies", ""))
                                lead[base][0] += 1
                                lead[base][1] += won
                        first = False
                if d["kind"] == "turn":
                    tot += 1
                    ch = d["options"][d["chosen"]]
                    if "switch" in ch:
                        sw += 1
                    if "mega" in ch:
                        b = min(5, d["turn"])
                        mega_turn[b][0] += 1
                        mega_turn[b][1] += won
            if tot:
                switch_rate.append((sw / tot, won))
    def table(d, k=12):
        rows = [(s, n, w / n) for s, (n, w) in d.items() if n >= min_n]
        rows.sort(key=lambda r: -r[2])
        name = lambda s: dex.species.get(s, {}).get("ko", s)
        return {"best": [(name(s), n, round(r, 3)) for s, n, r in rows[:k]],
                "worst": [(name(s), n, round(r, 3)) for s, n, r in rows[-k:][::-1]]}
    sw_bins = defaultdict(lambda: [0, 0])
    for r, w in switch_rate:
        b = "교체 없음" if r == 0 else ("조금 (~15%)" if r <= 0.15 else ("보통 (~35%)" if r <= 0.35 else "많이 (35%~)"))
        sw_bins[b][0] += 1
        sw_bins[b][1] += w
    return {
        "games": n_games, "avgTurns": round(turns / max(1, n_games), 1),
        "brought": table(brought), "lead": table(lead),
        "switch": {k: (n, round(w / n, 3)) for k, (n, w) in sw_bins.items() if n},
        "megaTurn": {f"{k}턴{'+' if k == 5 else ''}": (n, round(w / n, 3)) for k, (n, w) in sorted(mega_turn.items()) if n},
        "policies": {k: (n, round(w / n, 3)) for k, (n, w) in by_policy.items()},
    }


def write_report(dex: Dex, analysis: dict, progress: list, team_notes: list, path=None) -> Path:
    path = Path(path or ROOT / "reports" / "insights.md")
    path.parent.mkdir(exist_ok=True)
    a = analysis
    L = ["# 인사이트 리포트", "", f"대전 {a['games']}판 분석 · 평균 {a['avgTurns']}턴", ""]
    L += ["## 학습 진행 (상대별 승률)", "", "| 회차 | 시각 | 형식 | 학습 판수 | 쇼다운 기본 AI | 최대 데미지 AI | 선생님 AI | 이전 최고 신경망 | 교체 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in progress[-20:]:
        c = lambda k: f"{p['league'][k]['winrate']:.0%}" if k in p.get("league", {}) else "-"
        L.append(f"| {p['iter']} | {p['when'][5:16]} | {'싱글' if p['gameType'] == 'singles' else '더블'} | {p['games']} | "
                 f"{c('showdown_random')} | {c('max_damage')} | {c('heuristic')} | {c('best')} | {'예' if p.get('kept') else '아니오'} |")
    L += ["", "## 선출됐을 때 승률이 높은 포켓몬", "", "| 포켓몬 | 판수 | 승률 |", "|---|---|---|"]
    L += [f"| {s} | {n} | {r:.0%} |" for s, n, r in a["brought"]["best"]]
    L += ["", "## 선출됐을 때 승률이 낮은 포켓몬", "", "| 포켓몬 | 판수 | 승률 |", "|---|---|---|"]
    L += [f"| {s} | {n} | {r:.0%} |" for s, n, r in a["brought"]["worst"]]
    L += ["", "## 선봉 승률 (상위)", "", "| 포켓몬 | 판수 | 승률 |", "|---|---|---|"]
    L += [f"| {s} | {n} | {r:.0%} |" for s, n, r in a["lead"]["best"]]
    L += ["", "## 교체를 얼마나 했을 때 이겼나", "", "| 교체 빈도 | 판수 | 승률 |", "|---|---|---|"]
    L += [f"| {k} | {n} | {w:.0%} |" for k, (n, w) in a["switch"].items()]
    if a["megaTurn"]:
        L += ["", "## 메가진화 시점", "", "| 시점 | 판수 | 승률 |", "|---|---|---|"]
        L += [f"| {k} | {n} | {w:.0%} |" for k, (n, w) in a["megaTurn"].items()]
    L += ["", "## 정책별 승률 (리그 전체)", "", "| 정책 | 판수 | 승률 |", "|---|---|---|"]
    L += [f"| {k} | {n} | {w:.0%} |" for k, (n, w) in sorted(a["policies"].items(), key=lambda kv: -kv[1][1])]
    if team_notes:
        L += ["", "## 파티 최적화에서 승률을 올린 변경", ""]
        L += [f"- {t}" for t in team_notes[-30:]]
    L += ["", "> 주의: 승률은 이 프로그램의 AI끼리 둔 결과입니다. 사람 상대 실력은 실제 대전 기록(앱의 승/패 기록)으로 따로 확인합니다."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (path.with_suffix(".json")).write_text(json.dumps({"analysis": a, "progress": progress[-50:], "teamNotes": team_notes[-50:]},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    return path
