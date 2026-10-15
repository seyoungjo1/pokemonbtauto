# -*- coding: utf-8 -*-
"""존재 전체 — '쓸 수 있는 모든 포켓몬 × 그 포켓몬이 가질 수 있는 세트'를 변수 묶음으로 놓고 모두 대 모두 상성을 잰다.

엔트리(6마리)는 메타 팀 몇 개가 아니라 이 전체에서 고른다.

  존재 하나 = 세트(종·특성·도구·성격·SP·기술 4) + 토큰(속성 묶음) + 메타 비중(사용률 기반, 없으면 작게)
  세트 만들기: 메타 통계가 있으면 실제로 쓰이는 조합(가장 흔한 것 + 도구·기술 대안),
             없으면 배우는 기술·특성·능력치로 역할별(물리·특수·내구) 세트 → 처음 보는 포켓몬도 포함
  메가: 메가스톤을 든 세트를 따로 하나의 존재로 (메가 후 능력치·특성)
  G[i, j] = 존재 i 가 j 를 1:1 로 얼마나 유리하게 상대하는가 (데미지 엔진: 주는/받는 데미지·처치·스피드)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .bring import Pairwise
from .dex import Dex, to_id
from .teamgen import TeamGen, normalize_set

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "userdata" / "cache"


def _top(d: dict, n=1, skip=()):
    return [k for k, _ in sorted((d or {}).items(), key=lambda kv: -kv[1]) if k not in skip][:n]


def entities(dex: Dex, game_type="singles", per_species=3, seed=7) -> list:
    U = (dex.usage.get(game_type) or {}).get("pokemon") or {}
    tg = TeamGen(dex, seed=seed, game_type=game_type, meta=0.0)
    out, seen = [], set()

    def add(s, weight, src):
        s = normalize_set(dex, s)
        if s.get("species") not in dex.species or not s.get("moves"):
            return
        key = (s["species"], s.get("item"), s.get("ability"), tuple(sorted(s["moves"])), s.get("nature"))
        if key in seen:
            return
        seen.add(key)
        out.append({"set": s, "weight": float(weight), "src": src, "ko": dex.species[s["species"]]["ko"]
                    + (f" (메가)" if s.get("megaTarget") else "")})
    # 이번 시즌에 고를 수 있는 모든 종 + 메타 통계에 있는 메가 모습 (메가스톤 세트)
    species = dex.season_species() + [s for s in U if dex.species.get(s, {}).get("isMega")]
    for sid in species:
        sp = dex.species[sid]
        u = U.get(sid)
        if u:
            learn = set(dex.learnsets.get(dex.base_of_mega(sid)) or dex.learnsets.get(sid) or [])
            mv = {m: p for m, p in (u.get("moves") or {}).items() if m in dex.moves and (not learn or m in learn)}
            items = {i: p for i, p in (u.get("items") or {}).items() if i in dex.items}
            abil = _top(u.get("abilities"), 1) or [to_id(sp["abilities"].get("0", ""))]
            sps = u.get("spreads") or [{"sp": {"hp": 32, "atk": 16, "spa": 16}, "nature": "serious"}]
            variants = []
            top_moves = _top(mv, 4)
            for k in range(per_species):
                it = (_top(items, k + 1) or [""])[-1] if items else ""
                moves = list(top_moves)
                if k >= 1 and len(mv) > 4:   # 대안: 4번째 기술을 다음 후보로
                    alt = _top(mv, 4 + k)[-1]
                    moves = top_moves[:3] + [alt]
                spr = sps[min(k, len(sps) - 1)]
                variants.append({"species": sid, "ability": abil[0], "item": it, "nature": spr.get("nature") or "serious",
                                 "sp": dict(spr.get("sp") or {}), "moves": moves})
            for k, s in enumerate(variants):
                add(s, u.get("usage", 0.01) * (0.6 if k == 0 else 0.2), "meta")
        elif not sp.get("isMega"):
            for k in range(max(1, per_species - 1)):
                s = tg.make_set(sid)
                if s:
                    add(s, 0.002, "generated")
    return out


def universe(dex: Dex, game_type="singles", per_species=3, log=print, force=False):
    """존재 목록 + 모두 대 모두 1:1 상성표 (데이터 버전별로 캐시)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ver = (dex.meta.get("builtAt") or "x").replace(":", "")[:15]
    p = CACHE / f"universe_{game_type}_{ver}_{per_species}.npz"
    ents = entities(dex, game_type, per_species)
    if p.is_file() and not force:
        z = np.load(p, allow_pickle=False)
        if z["G"].shape[0] == len(ents):
            return ents, z["G"]
    pw = Pairwise(dex, game_type)
    sets = [e["set"] for e in ents]
    n = len(sets)
    G = np.zeros((n, n), np.float32)
    t0 = time.time()
    mons = [pw.mon(s) for s in sets]
    for i in range(n):
        a = mons[i]
        for j in range(i, n):
            b = mons[j]
            g = pw.pol.best_vs(a, b, pw.field, 0)
            t = pw.pol.best_vs(b, a, pw.field, 1)
            fast = pw.pol.speed(a, pw.field, 0) > pw.pol.speed(b, pw.field, 1)
            v = (g[0] + g[1]) - (t[0] + t[1]) + (0.15 if fast else -0.15)
            G[i, j] = v
            G[j, i] = -v
        if log and (i + 1) % 100 == 0:
            log(f"  상성표 {i + 1}/{n} · {time.time() - t0:.0f}초")
    np.savez_compressed(p, G=G)
    (CACHE / f"universe_{game_type}_{ver}_{per_species}.json").write_text(
        json.dumps([{"ko": e["ko"], "species": e["set"]["species"], "item": e["set"].get("item"), "weight": e["weight"],
                     "src": e["src"]} for e in ents], ensure_ascii=False), encoding="utf-8")
    return ents, G
