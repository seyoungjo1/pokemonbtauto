# -*- coding: utf-8 -*-
"""하위 토큰 — 포켓몬 한 세트를 '종/능력치 · 특성 · 도구 · 기술 ×4' 토큰으로 쪼갠다 (ARCHITECTURE.md 2.2).

각 토큰 = 엔진이 계산한 물리량 벡터 + id (학습 임베딩용). 신경망은 토큰들을 어텐션으로 묶어 세트 벡터를 만들고,
그 층에서 "테크니션 ↔ 불릿펀치", "잔비 ↔ 일렉트로빔", "구애스카프 ↔ 유턴" 같은 요소 사이의 관계를 배운다.
종 이름은 id 로만 들어가고(드롭아웃으로 자주 가려짐), 물리량만으로도 동작해야 하므로 처음 보는 종·기술·특성도 표현된다.
"""
from __future__ import annotations

import math

import numpy as np

from .dex import Dex, to_id
from .mon import Mon
from .montoken import (ABILITY_IMMUNE, MOVE_FLAGS, STATUSES, TARGETS, TYPES, WEATHER_SETTERS, SPEED_ABILITY,
                       _eff, move_multipliers, weather_profile)
from .teamgen import normalize_set

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
ROLE_ABILITIES = ["intimidate", "regenerator", "prankster", "magicguard", "sturdy", "multiscale", "shadowshield",
                  "unaware", "guts", "poisonheal", "speedboost", "moxie", "sheerforce", "technician", "adaptability",
                  "hugepower", "purepower", "infiltrator", "goodasgold", "magicbounce", "disguise", "supremeoverlord",
                  "roughskin", "ironbarbs", "flamebody", "static", "cursedbody", "pressure", "stamina", "armortail",
                  "electricsurge", "psychicsurge", "grassysurge", "mistysurge"]
ITEM_FLAGS = ["choicescarf", "focussash", "lifeorb", "leftovers", "sitrusberry", "lumberry", "lightclay", "damprock",
              "heatrock", "smoothrock", "icyrock", "expertbelt", "muscleband", "wiseglasses", "whiteherb", "mentalherb",
              "shedshell", "ironball", "quickclaw", "scopelens", "kingsrock", "widelens", "zoomlens", "brightpowder",
              "focusband", "shellbell", "bigroot", "metronome", "leppaberry"]
HAZARD_SC = ["stealthrock", "spikes", "toxicspikes", "stickyweb", "reflect", "lightscreen", "auroraveil", "tailwind"]
PSEUDO = ["trickroom", "gravity"]
RECOVERY = {"recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun", "synthesis", "milkdrink", "shoreup",
            "strengthsap", "wish", "rest"}
PROTECT = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap", "burningbulwark", "obstruct"}


class Vocab:
    """효과 처리기 목록과 id 표. 데이터(Dex)에서 만든다 — 새 요소가 생기면 뒤에 붙는다."""

    def __init__(self, dex: Dex, n_ability=56, n_item=29, n_move=25):
        def top(tbl, n):
            cnt = {}
            for v in tbl.values():
                for c in v.get("callbacks") or []:
                    cnt[c] = cnt.get(c, 0) + 1
            return [k for k, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]
        self.ab_cb, self.it_cb, self.mv_cb = top(dex.abilities, n_ability), top(dex.items, n_item), top(dex.moves, n_move)
        self.species = {k: i + 1 for i, k in enumerate(sorted(dex.species))}
        self.ability = {k: i + 1 for i, k in enumerate(sorted(dex.abilities))}
        self.item = {k: i + 1 for i, k in enumerate(sorted(dex.items))}
        self.move = {k: i + 1 for i, k in enumerate(sorted(dex.moves))}
        self.dims = {"mon": self._dim_mon(), "ability": self._dim_ability(), "item": self._dim_item(), "move": self._dim_move()}
        self.sizes = {"species": len(self.species) + 1, "ability": len(self.ability) + 1, "item": len(self.item) + 1,
                      "move": len(self.move) + 1}

    def _dim_mon(self):
        return 18 + 6 + 6 + 1 + 6 + 18 + 3

    def _dim_ability(self):
        return len(self.ab_cb) + 2 + 18 + 4 + 4 + len(ROLE_ABILITIES)

    def _dim_item(self):
        return len(self.it_cb) + 3 + 1 + len(ITEM_FLAGS) + 4

    def _dim_move(self):
        return (18 + 3 + 3 + 7 + len(MOVE_FLAGS) + len(self.mv_cb) + (1 + len(STATUSES) + 3) + 7 + 1 + 2 + 18
                + 6 + len(HAZARD_SC) + len(PSEUDO) + 4 + 3 + 9)

    def to_json(self):
        return {"ab_cb": self.ab_cb, "it_cb": self.it_cb, "mv_cb": self.mv_cb}


def _mons(dex: Dex, s: dict):
    m = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""), nature=s.get("nature", "serious"),
            sp=dict(s.get("sp") or {}), moves=list(s.get("moves") or [])).setup(dex)
    mm = Mon(species=s["species"], ability=s.get("ability", ""), item=s.get("item", ""), nature=s.get("nature", "serious"),
             sp=dict(s.get("sp") or {}), moves=list(s.get("moves") or [])).setup(dex)
    is_mega = mm.mega_evolve()
    return m, (mm if is_mega else m), is_mega


def mon_token(dex: Dex, m: Mon, atk: Mon, is_mega: bool) -> np.ndarray:
    sp = dex.species[m.species]
    types = list(atk.types)
    v = [1.0 if t in types else 0.0 for t in TYPES]
    v += [m.stats[k] / 200.0 for k in STATS]
    v += [sp["baseStats"][k] / 200.0 for k in STATS]
    v += [float(is_mega)] + ([atk.stats[k] / 200.0 for k in STATS] if is_mega else [0.0] * 6)
    imm = ABILITY_IMMUNE.get(atk.ability, "")
    v += [(-3.0 if t == imm else (math.log2(e) if (e := _eff(dex, t, types)) > 0 else -3.0)) for t in TYPES]
    v += [min(3.0, (sp.get("weightkg") or 50) / 100.0), 1.0 if sp.get("nfe") else 0.0, 1.0 if sp.get("isMega") else 0.0]
    return np.array(v, np.float32)


def ability_token(dex: Dex, vocab: Vocab, s: dict, atk: Mon, mults: dict) -> np.ndarray:
    ab = atk.ability
    cb = set((dex.abilities.get(ab) or {}).get("callbacks") or [])
    v = [1.0 if c in cb else 0.0 for c in vocab.ab_cb]
    ra = [mults[m][0] for m in mults] or [1.0]
    v += [max(ra) - 1.0, sum(ra) / len(ra) - 1.0]
    imm = ABILITY_IMMUNE.get(ab, "")
    v += [1.0 if t == imm else 0.0 for t in TYPES]
    v += [1.0 if WEATHER_SETTERS.get(ab) == w else 0.0 for w in ("rain", "sun", "sand", "snow")]
    v += [1.0 if SPEED_ABILITY.get(ab) == w else 0.0 for w in ("rain", "sun", "sand", "snow")]
    v += [1.0 if ab == r else 0.0 for r in ROLE_ABILITIES]
    return np.array(v, np.float32)


def item_token(dex: Dex, vocab: Vocab, s: dict, mults: dict) -> np.ndarray:
    item = to_id(s.get("item") or "")
    it = dex.items.get(item) or {}
    cb = set(it.get("callbacks") or [])
    v = [1.0 if c in cb else 0.0 for c in vocab.it_cb]
    v += [1.0 if it.get("megaStone") else 0.0, 1.0 if it.get("isBerry") else 0.0, 1.0 if not item else 0.0]
    ri = [mults[m][1] for m in mults] or [1.0]
    v += [max(ri) - 1.0]
    v += [1.0 if item == f else 0.0 for f in ITEM_FLAGS]
    v += [1.0 if (it.get("fling") or {}).get("basePower", 0) >= 60 else 0.0, 0.0, 0.0, 0.0]
    return np.array(v, np.float32)


def move_token(dex: Dex, vocab: Vocab, mid: str, atk: Mon, mult) -> np.ndarray:
    mo = dex.moves.get(mid) or {}
    types = list(atk.types)
    cat = mo.get("category", "Status")
    bp = mo.get("basePower") or 0
    v = [1.0 if mo.get("type") == t else 0.0 for t in TYPES]
    v += [1.0 if cat == c else 0.0 for c in ("Physical", "Special", "Status")]
    pr = mo.get("priority", 0) or 0
    v += [1.0 if pr < 0 else 0.0, 1.0 if pr == 0 else 0.0, 1.0 if pr > 0 else 0.0]
    tg = [0.0] * 7
    tg[TARGETS.get(mo.get("target", "normal"), 0)] = 1.0
    v += tg
    fl = mo.get("flags") or {}
    v += [1.0 if fl.get(f) else 0.0 for f in MOVE_FLAGS]
    cb = set(mo.get("callbacks") or [])
    v += [1.0 if c in cb else 0.0 for c in vocab.mv_cb]
    # 부가효과: 확률, 상태 6, 상대 랭크다운·자기 랭크업·풀죽음
    sec = [0.0] * (1 + len(STATUSES) + 3)
    for sd in ([mo["secondary"]] if mo.get("secondary") else []) + list(mo.get("secondaries") or []):
        ch = (sd.get("chance") or 100) / 100.0
        sec[0] += ch
        if sd.get("status") in STATUSES:
            sec[1 + STATUSES.index(sd["status"])] += ch
        if sd.get("boosts"):
            sec[-3] += ch
        if (sd.get("self") or {}).get("boosts"):
            sec[-2] += ch
        if sd.get("volatileStatus") == "flinch":
            sec[-1] += ch
    if cat == "Status" and mo.get("status") in STATUSES:
        sec[1 + STATUSES.index(mo["status"])] += 1.0
    v += sec
    # 자기/상대 랭크 변화 (변화기 boosts, self.boosts)
    bst = [0.0] * 7
    own = (mo.get("boosts") if mo.get("target") == "self" else None) or (mo.get("self") or {}).get("boosts") or {}
    if mid == "bellydrum":
        own = {"atk": 6}
    for k, i in (("atk", 0), ("def", 1), ("spa", 2), ("spd", 3), ("spe", 4)):
        bst[i] = float(own.get(k, 0))
    foe = mo.get("boosts") if mo.get("target") != "self" and cat == "Status" else None
    bst[5] = float(sum(foe.values())) if foe else 0.0
    bst[6] = float(mid in PROTECT)
    v += bst
    stab = 1.0 if mo.get("type") in types else 0.0
    v += [stab]
    ra, ri = mult
    v += [ra - 1.0, ri - 1.0]
    # 18 방어 타입 상대 화력 (위력 × 자속 × 배율 × 상성 × 공격 실능)
    if cat != "Status" and bp:
        st = atk.stats["atk" if cat == "Physical" else "spa"]
        # 상한으로 자르지 않고 로그로 눌러 센 타격끼리의 차이도 남긴다 (log1p: 0→0, 1→0.69, 4→1.61, 10→2.40)
        v += [math.log1p(bp * (1.5 if stab else 1.0) * ra * ri * _eff(dex, mo["type"], [t]) * st / (100 * 150)) for t in TYPES]
    else:
        v += [0.0] * 18
    acc = mo.get("accuracy")
    v += [bp / 100.0, 1.0 if acc is True else (acc or 100) / 100.0, (mo.get("pp") or 10) / 30.0,
          1.0 if mo.get("selfSwitch") else 0.0, (mo.get("drain") or [0, 1])[0] / max(1, (mo.get("drain") or [0, 1])[1]),
          1.0 if (mo.get("recoil") or mo.get("mindBlownRecoil") or mo.get("hasCrashDamage")) else 0.0]
    sc = mo.get("sideCondition") or ""
    v += [1.0 if sc == h else 0.0 for h in HAZARD_SC]
    pw = mo.get("pseudoWeather") or ""
    v += [1.0 if pw == p else 0.0 for p in PSEUDO]
    w = WEATHER_SETTERS.get(mid, "")
    v += [1.0 if w == x else 0.0 for x in ("rain", "sun", "sand", "snow")]
    v += [1.0 if mid in RECOVERY else 0.0, 1.0 if mo.get("terrain") else 0.0,
          float(mo.get("multihit") if isinstance(mo.get("multihit"), (int, float)) else (3 if mo.get("multihit") else 1)) / 3.0]
    # 필중(회피 무시) · 급소율↑/확정급소 · 일격기 · 방어 뚫기 · 특성 무시 · 방어 무시 · 얼음 녹임 · 잠든 채 사용 · 강제 교체
    v += [1.0 if acc is True else 0.0, 1.0 if ((mo.get("critRatio") or 1) > 1 or mo.get("willCrit")) else 0.0,
          1.0 if mo.get("ohko") else 0.0, 1.0 if mo.get("breaksProtect") else 0.0, 1.0 if mo.get("ignoreAbility") else 0.0,
          1.0 if mo.get("ignoreDefensive") else 0.0, 1.0 if mo.get("thawsTarget") else 0.0, 1.0 if mo.get("sleepUsable") else 0.0,
          1.0 if mo.get("forceSwitch") else 0.0]
    return np.array(v, np.float32)


def set_tokens(dex: Dex, s: dict, vocab: Vocab) -> dict:
    """한 세트 → {'mon': (vec, id), 'ability': (vec, id), 'item': (vec, id), 'moves': [(vec, id), ...], 'extra': 집계}."""
    s = normalize_set(dex, s)
    m, atk, is_mega = _mons(dex, s)
    moves = [x for x in m.moves if x in dex.moves]
    mults = {}
    for mid in moves:
        mo = dex.moves[mid]
        mults[mid] = move_multipliers(dex, atk, mid) if (mo["category"] != "Status" and mo.get("basePower")) else (1.0, 1.0)
    wp = weather_profile(dex, s)
    extra = np.array([wp["gain"][w] for w in ("rain", "sun", "sand", "snow")] + [1.0 if w in wp["sets"] else 0.0 for w in ("rain", "sun", "sand", "snow")], np.float32)
    return {
        "mon": (mon_token(dex, m, atk, is_mega), vocab.species.get(m.species, 0)),
        "ability": (ability_token(dex, vocab, s, atk, mults), vocab.ability.get(atk.ability, 0)),
        "item": (item_token(dex, vocab, s, mults), vocab.item.get(to_id(s.get("item") or ""), 0)),
        "moves": [(move_token(dex, vocab, mid, atk, mults[mid]), vocab.move.get(mid, 0)) for mid in moves],
        "extra": extra,
    }


def batch_sets(toks: list, vocab: Vocab):
    """세트 토큰 목록 → 인코더 입력. inputs[type] = [B, n, d], ids[type] = [B, n], mask[type] = [B, n] (True = 있음)."""
    B = len(toks)
    inputs = {"mon": np.zeros((B, 1, vocab.dims["mon"]), np.float32), "ability": np.zeros((B, 1, vocab.dims["ability"]), np.float32),
              "item": np.zeros((B, 1, vocab.dims["item"]), np.float32), "move": np.zeros((B, 4, vocab.dims["move"]), np.float32)}
    ids = {"mon": np.zeros((B, 1), np.int64), "ability": np.zeros((B, 1), np.int64), "item": np.zeros((B, 1), np.int64),
           "move": np.zeros((B, 4), np.int64)}
    mask = {"mon": np.ones((B, 1), bool), "ability": np.ones((B, 1), bool), "item": np.ones((B, 1), bool), "move": np.zeros((B, 4), bool)}
    extra = np.zeros((B, 8), np.float32)
    for b, t in enumerate(toks):
        for k in ("mon", "ability", "item"):
            inputs[k][b, 0], ids[k][b, 0] = t[k]
        for j, (vec, i) in enumerate(t["moves"][:4]):
            inputs["move"][b, j], ids["move"][b, j], mask["move"][b, j] = vec, i, True
        extra[b] = t["extra"]
    return inputs, ids, mask, extra


def selfcheck(dex: Dex | None = None) -> dict:
    dex = dex or Dex()
    vb = Vocab(dex)
    sc = set_tokens(dex, {"species": "scizor", "ability": "technician", "item": "lifeorb", "nature": "adamant",
                          "sp": {"atk": 32, "hp": 32}, "moves": ["bulletpunch", "uturn", "swordsdance", "closecombat"]}, vb)
    pel = set_tokens(dex, {"species": "pelipper", "ability": "drizzle", "item": "damprock", "nature": "modest",
                           "sp": {"spa": 32, "hp": 32}, "moves": ["hurricane", "weatherball", "uturn", "roost"]}, vb)
    errs = []
    for k in ("mon", "ability", "item"):
        if len(sc[k][0]) != vb.dims[k]:
            errs.append(f"{k} 길이 {len(sc[k][0])} ≠ {vb.dims[k]}")
    if any(len(v) != vb.dims["move"] for v, _ in sc["moves"]):
        errs.append("move 길이 불일치")
    # 테크니션 핫삼: 불릿펀치 토큰의 특성 배율 > 0, 인파이트는 0
    off = 18 + 3 + 3 + 7 + len(MOVE_FLAGS) + len(vb.mv_cb) + (1 + len(STATUSES) + 3) + 7 + 1
    bp_ra = sc["moves"][0][0][off]
    cc_ra = sc["moves"][3][0][off]
    if not (bp_ra > 0.3 and abs(cc_ra) < 1e-6):
        errs.append(f"테크니션 배율: 불릿펀치 {bp_ra:.2f}, 인파이트 {cc_ra:.2f}")
    # 잔비 패리퍼: 특성 토큰의 날씨 설치 = 비
    a = pel["ability"][0]
    o = len(vb.ab_cb) + 2 + 18
    if a[o] != 1.0:
        errs.append("잔비 → 비 설치 플래그 없음")
    inputs, ids, mask, extra = batch_sets([sc, pel], vb)
    if inputs["move"].shape != (2, 4, vb.dims["move"]) or not mask["move"].all():
        errs.append("배치 모양 이상")
    return {"ok": not errs, "errors": errs, "dims": vb.dims, "sizes": vb.sizes,
            "summary": f"하위 토큰 종/능력치 {vb.dims['mon']} · 특성 {vb.dims['ability']} · 도구 {vb.dims['item']} · 기술 {vb.dims['move']}칸"}
