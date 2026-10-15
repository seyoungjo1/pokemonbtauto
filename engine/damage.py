# -*- coding: utf-8 -*-
"""데미지 계산 — 쇼다운(sim/battle-actions.ts getDamage + 챔피언스 modifyDamage)과
같은 순서·같은 반올림으로 계산한다.

검증: tests/selftest.py 가 쇼다운 시뮬레이터로 만든 수천 개의 무작위 사례
(tests/parity_*.json)와 16단계 난수 결과를 한 칸씩 비교한다.

지원하지 않는 효과가 끼어 있으면 결과의 notes 에 '근사' 로 적는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield

from .dex import Dex, to_id
from .mon import BOOST_TABLE, FieldState, Mon, chain, mod4096, modify, trunc

SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}
MOLD_BREAKERS = {"moldbreaker", "teravolt", "turboblaze"}
ATE_ABILITIES = {"aerilate": "Flying", "pixilate": "Fairy", "refrigerate": "Ice",
                 "galvanize": "Electric", "dragonize": "Dragon"}
NO_ATE = {"judgment", "multiattack", "naturalgift", "revelationdance", "technoblast",
          "terrainpulse", "weatherball"}
TYPE_ITEMS = {
    "silkscarf": "Normal", "blackbelt": "Fighting", "sharpbeak": "Flying", "poisonbarb": "Poison",
    "softsand": "Ground", "hardstone": "Rock", "silverpowder": "Bug", "spelltag": "Ghost",
    "metalcoat": "Steel", "charcoal": "Fire", "mysticwater": "Water", "miracleseed": "Grass",
    "magnet": "Electric", "twistedspoon": "Psychic", "nevermeltice": "Ice", "dragonfang": "Dragon",
    "blackglasses": "Dark", "fairyfeather": "Fairy",
}
RESIST_BERRIES = {
    "occaberry": "Fire", "passhoberry": "Water", "wacanberry": "Electric", "rindoberry": "Grass",
    "yacheberry": "Ice", "chopleberry": "Fighting", "kebiaberry": "Poison", "shucaberry": "Ground",
    "cobaberry": "Flying", "payapaberry": "Psychic", "tangaberry": "Bug", "chartiberry": "Rock",
    "kasibberry": "Ghost", "habanberry": "Dragon", "colburberry": "Dark", "babiriberry": "Steel",
    "roseliberry": "Fairy",
}
PINCH = {"blaze": "Fire", "overgrow": "Grass", "torrent": "Water", "swarm": "Bug"}
CRIT_MULT = [0, 24, 8, 2, 1]
SUB_ABILITY, SUB_ITEM, SUB_VOLATILE, SUB_SIDE, SUB_FIELD = 7, 8, 2, 4, 5

# 데미지 계산에 들어가는 효과 중 이 엔진이 구현한 것 (그 외는 notes 에 근사로 표시)
SUPPORTED_ABILITIES = {
    "adaptability", "aerilate", "pixilate", "refrigerate", "galvanize", "dragonize", "analytic",
    "dryskin", "eartheater", "fairyaura", "filter", "solidrock", "prismarmor", "firemane",
    "flashfire", "fluffy", "friendguard", "furcoat", "guts", "heatproof", "hugepower", "purepower",
    "hustle", "ironfist", "lightningrod", "liquidvoice", "marvelscale", "megalauncher", "megasol",
    "minus", "plus", "moldbreaker", "teravolt", "turboblaze", "motordrive", "multiscale",
    "shadowshield", "overgrow", "blaze", "torrent", "swarm", "purifyingsalt", "reckless",
    "sandforce", "sapsipper", "scrappy", "sharpness", "sheerforce", "sniper", "solarpower",
    "soundproof", "strongjaw", "supremeoverlord", "technician", "thickfat", "toughclaws", "unaware",
    "voltabsorb", "waterabsorb", "stormdrain", "waterbubble", "levitate", "eelevate",
    "bulletproof", "infiltrator", "rivalry", "skilllink", "longreach", "goodasgold", "cloudnine",
    "airlock", "battlearmor", "shellarmor", "superluck", "merciless", "wellbakedbody",
    "windrider", "telepathy", "quickfeet", "chlorophyll", "swiftswim", "sandrush", "slushrush",
    "icescales", "punkrock", "tintedlens",
}
SUPPORTED_ITEMS = set(TYPE_ITEMS) | set(RESIST_BERRIES) | {
    "chilanberry", "lifeorb", "expertbelt", "muscleband", "wiseglasses", "lightball",
    "choicescarf", "ironball", "scopelens",
}
APPROX_MOVES = {"fickle beam", "beatup", "lashout", "avalanche", "payback", "assurance",
                "stompingtantrum", "temperflare", "ficklebeam"}


@dataclass
class Ctx:
    """데미지에 영향을 주는 주변 상황."""
    atk_side: int = 0
    def_side: int = 1
    actives: list = dfield(default_factory=list)   # 필드 위 모든 포켓몬 (구름보호·페어리오라 등)
    atk_ally: Mon | None = None
    def_ally: Mon | None = None
    spread: bool | None = None       # None = 더블에서 범위기면 자동으로 True
    helping_hand: bool = False
    moved_last: bool = False         # Analytic
    target_moved_first: bool = False  # Payback
    hurt_by_target: bool = False     # Avalanche
    target_hurt_this_turn: bool = False  # Assurance
    last_move_failed: bool = False   # Stomping Tantrum · Temper Flare
    stats_lowered: bool = False      # Lash Out


@dataclass
class DamageResult:
    move: str
    move_type: str = ""
    category: str = ""
    base_power: int = 0
    rolls: list = dfield(default_factory=list)        # 일반 (타당 16개, 다단히트는 타별 리스트)
    crit_rolls: list = dfield(default_factory=list)
    hits: list = dfield(default_factory=lambda: [1])  # 가능한 타수
    crit_chance: float = 0.0
    type_mod: int = 0
    immune: str = ""      # 타입·땅 무효 (쇼다운 getDamage 가 false)
    blocked: str = ""     # 특성 등으로 막힘 (저수·피뢰침 등) — rolls 는 계산값 그대로
    fixed: int | None = None
    notes: list = dfield(default_factory=list)

    @property
    def min(self):
        return min(self.rolls[0]) if self.rolls and self.rolls[0] else 0

    @property
    def max(self):
        return max(self.rolls[0]) if self.rolls and self.rolls[0] else 0


# ======================================================================== 도우미
class _Calc:
    def __init__(self, dex: Dex, atk: Mon, dfn: Mon, move_id: str, field: FieldState, ctx: Ctx):
        self.dex, self.atk, self.dfn, self.field, self.ctx = dex, atk, dfn, field, ctx
        self.m = dict(dex.moves[move_id])
        self.m["flags"] = dict(self.m.get("flags") or {})
        self.notes = []
        self.mold = atk.ability in MOLD_BREAKERS
        self.actives = ctx.actives or [atk, dfn]
        self.megasol = atk.ability == "megasol"
        self.base_power = 0
        self.type_mod = 0
        self.immune_reason = ""

    # ---- 특성 (틀깨기면 상대의 breakable 특성 무시)
    def ab(self, mon: Mon, ability: str) -> bool:
        if mon.ability != ability:
            return False
        if mon is self.dfn and self.mold and self.dex.abilities.get(ability, {}).get("flags", {}).get("breakable"):
            return False
        return True

    def ab_breakable_ok(self, mon: Mon) -> bool:
        a = self.dex.abilities.get(mon.ability, {})
        return not (mon is self.dfn and self.mold and a.get("flags", {}).get("breakable"))

    # ---- 날씨
    def field_weather(self) -> str:
        for p in self.actives:
            if p and not p.fainted and p.ability in ("cloudnine", "airlock"):
                return ""
        return self.field.weather

    def eff_weather(self, mon: Mon) -> str:
        if self.megasol:   # 메가솔라: 그 포켓몬이 기술을 쓰는 동안은 언제나 쾌청
            return "sunnyday"
        return self.field_weather()

    # ---- 땅에 있는가
    def grounded(self, mon: Mon):
        if self.field.gravity:
            return True
        if mon.item == "ironball":
            return True
        if mon.has_type("Flying"):
            return False
        if mon.ability in ("levitate", "eelevate") and self.ab_breakable_ok(mon):
            return None
        return True

    # ---- 스피드 (핸들러 순서·일렉트릭볼·자이로볼)
    def spe_stat(self, mon: Mon, side: int) -> int:
        s = mon.stats["spe"]
        b = mon.boost("spe")
        s = trunc(s * BOOST_TABLE[b]) if b >= 0 else trunc(s / BOOST_TABLE[-b])
        m = 4096
        w = self.field_weather()
        if mon.ability == "chlorophyll" and w == "sunnyday" or mon.ability == "swiftswim" and w == "raindance" \
                or mon.ability == "sandrush" and w == "sandstorm" or mon.ability == "slushrush" and w == "snowscape":
            m = chain(m, mod4096(2))
        if mon.ability == "quickfeet" and mon.status:
            m = chain(m, mod4096(1.5))
        if mon.item == "choicescarf":
            m = chain(m, mod4096(1.5))
        if self.field.sides[side].tailwind:
            m = chain(m, mod4096(2))
        if mon.status == "par" and mon.ability != "quickfeet":
            m = chain(m, mod4096(0.5))
        return modify(s, m)

    def action_speed(self, mon: Mon) -> int:
        side = self.ctx.atk_side if mon is self.atk or mon is self.ctx.atk_ally else self.ctx.def_side
        s = self.spe_stat(mon, side)
        return -s if self.field.trick_room else s

    # ---- 이벤트: 핸들러(우선도, 스피드, subOrder) 정렬 후 배율 연결
    def run_event(self, value: int, handlers: list) -> int:
        handlers.sort(key=lambda h: (-h[0], -h[1], h[2]))
        m = 4096
        for _, _, _, kind, x in handlers:
            if kind == "chain":
                m = chain(m, x)
            else:  # 'direct' — 핸들러가 값을 바로 바꾼다 (Hustle, 모래바람 특방 등)
                value = modify(value, x)
        if m != 4096:
            value = modify(value, m)
        return value

    def H(self, prio, holder, sub, num, den=1, kind="chain"):
        speed = self.action_speed(holder) if isinstance(holder, Mon) else 0
        return (prio, speed, sub, kind, mod4096(num, den))


# ======================================================================== 본체
def calc(dex: Dex, atk: Mon, dfn: Mon, move_id: str, field: FieldState | None = None,
         ctx: Ctx | None = None) -> DamageResult:
    field = field or FieldState()
    ctx = ctx or Ctx()
    move_id = dex.find("moves", move_id) or to_id(move_id)
    c = _Calc(dex, atk, dfn, move_id, field, ctx)
    m = c.m
    res = DamageResult(move=move_id, category=m["category"])
    if m["category"] == "Status":
        res.notes.append("변화기")
        return res

    _prepare_move(c)
    res.move_type, res.category = m["type"], m["category"]
    if move_id in APPROX_MOVES:
        res.notes.append("상황 의존 위력 — 기본값 사용")
    for mon, role in ((atk, "공격"), (dfn, "방어")):
        if mon.ability and mon.ability not in SUPPORTED_ABILITIES and _ability_matters(dex, mon.ability):
            res.notes.append(f"{role} 특성 {mon.ability} 미구현 — 근사")
        if mon.item and mon.item not in SUPPORTED_ITEMS and _item_matters(dex, mon.item):
            res.notes.append(f"{role} 도구 {mon.item} 미구현 — 근사")

    # 타수
    mh = m.get("multihit")
    if isinstance(mh, list):
        res.hits = [mh[1]] if c.ab(atk, "skilllink") else list(range(mh[0], mh[1] + 1))
    elif isinstance(mh, int):
        res.hits = [mh]

    # 급소 확률
    res.crit_chance = _crit_chance(c)

    # 특성으로 막힘 (쇼다운에선 TryHit 단계 — getDamage 와 별개)
    res.blocked = _blocked_by(c)

    n_hits = max(res.hits)
    # 전투무장·조가비갑옷이면 급소가 일어나지 않는다 (CriticalHit 이벤트가 false)
    crit_blocked = c.dfn.ability in ("battlearmor", "shellarmor") and c.ab_breakable_ok(c.dfn)
    for crit in (False, True):
        if crit and crit_blocked:
            res.crit_rolls = [list(r) for r in res.rolls]
            continue
        per_hit = []
        for hit in range(1, n_hits + 1):
            out = _get_damage(c, crit, hit)
            if out is False:
                res.immune = c.immune_reason
                return res
            if isinstance(out, tuple):        # 고정 데미지
                res.fixed = out[1]
                per_hit.append([out[1]] * 16)
                continue
            per_hit.append(out)
        if crit:
            res.crit_rolls = per_hit
        else:
            res.rolls = per_hit
    res.base_power = c.base_power
    res.type_mod = c.type_mod
    res.notes.extend(c.notes)
    return res


def _ability_matters(dex, a):
    cb = dex.abilities.get(a, {}).get("callbacks", [])
    return any(k for k in cb if any(s in k for s in (
        "BasePower", "ModifyAtk", "ModifySpA", "ModifyDef", "ModifySpD", "ModifyDamage", "ModifyType",
        "Effectiveness", "ModifySTAB", "CriticalHit", "ModifyCritRatio")))


def _item_matters(dex, i):
    cb = dex.items.get(i, {}).get("callbacks", [])
    return any(k for k in cb if any(s in k for s in (
        "BasePower", "ModifyAtk", "ModifySpA", "ModifyDef", "ModifySpD", "ModifyDamage", "Effectiveness")))


def _prepare_move(c: _Calc):
    """useMoveInner 의 ModifyType / ModifyMove 단계를 흉내낸다."""
    m, atk = c.m, c.atk
    mid = m["id"]
    w = c.eff_weather(atk)
    # 기술 자체의 타입·위력 변화
    if mid == "weatherball":
        t = {"sunnyday": "Fire", "raindance": "Water", "sandstorm": "Rock", "snowscape": "Ice"}.get(w)
        if t:
            m["type"] = t
            m["basePower"] *= 2
    elif mid == "terrainpulse":
        if c.grounded(atk) and c.field.terrain:
            m["type"] = {"electricterrain": "Electric", "grassyterrain": "Grass",
                         "mistyterrain": "Fairy", "psychicterrain": "Psychic"}[c.field.terrain]
            m["basePower"] *= 2
    elif mid == "aurawheel":
        m["type"] = "Dark" if c.atk.sp_data["name"] == "Morpeko-Hangry" else "Electric"
    elif mid == "ragingbull":
        m["type"] = {"Tauros-Paldea-Combat": "Fighting", "Tauros-Paldea-Blaze": "Fire",
                     "Tauros-Paldea-Aqua": "Water"}.get(atk.sp_data["name"], m["type"])
    elif mid == "shellsidearm":
        _shell_side_arm(c)
    # 특성에 의한 타입 변화
    if atk.ability in ATE_ABILITIES and m["type"] == "Normal" and mid not in NO_ATE:
        m["type"] = ATE_ABILITIES[atk.ability]
        m["typeChangerBoosted"] = True
    if atk.ability == "liquidvoice" and m["flags"].get("sound"):
        m["type"] = "Water"
    # ModifyMove
    if atk.ability == "sheerforce" and m.get("secondaries"):
        m["hasSheerForce"] = True
    if atk.ability == "longreach":
        m["flags"].pop("contact", None)
    if atk.ability == "scrappy":
        ii = m.get("ignoreImmunity")
        if ii is not True:
            ii = dict(ii or {})
            ii["Fighting"] = ii["Normal"] = True
            m["ignoreImmunity"] = ii
    if atk.ability == "infiltrator":
        m["infiltrates"] = True
    if mid == "expandingforce" and c.field.terrain == "psychicterrain" and c.grounded(atk):
        m["target"] = "allAdjacentFoes"
    # 범위기 (더블에서 상대가 둘 이상이면 0.75)
    if c.ctx.spread is None:
        c.spread = c.field.doubles and m.get("target") in SPREAD_TARGETS and _n_targets(c) > 1
    else:
        c.spread = bool(c.ctx.spread)


def _n_targets(c):
    n = 1 + (1 if c.ctx.def_ally is not None and not c.ctx.def_ally.fainted else 0)
    if c.m.get("target") == "allAdjacent" and c.ctx.atk_ally is not None and not c.ctx.atk_ally.fainted:
        n += 1
    return n


def _shell_side_arm(c):
    atk, dfn = c.atk, c.dfn

    def st(mon, s):
        v = mon.stats[s]
        b = mon.boost(s)
        return trunc(v * BOOST_TABLE[b]) if b >= 0 else trunc(v / BOOST_TABLE[-b])
    lv = atk.level
    phys = trunc(trunc(trunc(trunc(2 * lv / 5 + 2) * 90 * st(atk, "atk")) / st(dfn, "def")) / 50)
    spec = trunc(trunc(trunc(trunc(2 * lv / 5 + 2) * 90 * st(atk, "spa")) / st(dfn, "spd")) / 50)
    if phys > spec or (phys == spec and False):
        c.m["category"] = "Physical"
        c.m["flags"]["contact"] = 1


def _crit_chance(c: _Calc) -> float:
    m = c.m
    if c.dfn.ability in ("battlearmor", "shellarmor") and c.ab_breakable_ok(c.dfn):
        return 0.0
    if m.get("willCrit"):
        return 1.0
    r = m.get("critRatio", 1) or 0
    if c.atk.ability == "superluck":
        r += 1
    if c.atk.item == "scopelens":
        r += 1
    if c.atk.ability == "merciless" and c.dfn.status in ("psn", "tox"):
        r = 5
    r = max(0, min(4, r))
    return 0.0 if r == 0 else 1.0 / CRIT_MULT[r]


def _blocked_by(c: _Calc) -> str:
    m, d, t = c.m, c.dfn, c.m["type"]
    if not c.ab_breakable_ok(d):
        return ""
    a = d.ability
    table = {
        "voltabsorb": "Electric", "lightningrod": "Electric", "motordrive": "Electric",
        "waterabsorb": "Water", "stormdrain": "Water", "dryskin": "Water", "flashfire": "Fire",
        "wellbakedbody": "Fire", "sapsipper": "Grass", "eartheater": "Ground",
    }
    if table.get(a) == t:
        return a
    if a == "bulletproof" and m["flags"].get("bullet"):
        return a
    if a == "soundproof" and m["flags"].get("sound"):
        return a
    if a == "windrider" and m["flags"].get("wind"):
        return a
    if a == "telepathy" and c.ctx.atk_side == c.ctx.def_side:
        return a
    if c.field.terrain == "psychicterrain" and m.get("priority", 0) > 0 and c.grounded(d) \
            and c.ctx.atk_side != c.ctx.def_side:
        return "psychicterrain"
    return ""


def _get_damage(c: _Calc, crit: bool, hit: int):
    """sim/battle-actions.ts getDamage 와 같은 흐름. False=무효, (fixed, n)=고정 데미지, list=16단계."""
    m, atk, dfn, dex = c.m, c.atk, c.dfn, c.dex
    mid, mtype = m["id"], m["type"]
    c.immune_reason = ""

    # ---- 타입 무효
    ii = m.get("ignoreImmunity")
    if not (ii is True or (isinstance(ii, dict) and ii.get(mtype))):
        if mtype == "Ground":
            g = c.grounded(dfn)
            if not g:
                c.immune_reason = "levitate" if g is None else "type"
                return False
        elif any(dex.immune(mtype, t) for t in dfn.types):
            c.immune_reason = "type"
            return False

    if m.get("ohko"):
        return ("fixed", dfn.maxhp)
    if m.get("damage") == "level":
        return ("fixed", atk.level)
    if isinstance(m.get("damage"), int):
        return ("fixed", m["damage"])
    if mid == "superfang":
        return ("fixed", max(1, trunc(dfn.hp / 2)))
    if mid == "endeavor":
        return ("fixed", max(0, dfn.hp - atk.hp))
    if mid == "finalgambit":
        return ("fixed", atk.hp)
    if mid in ("counter", "mirrorcoat", "metalburst", "comeuppance"):
        c.notes.append("받은 데미지에 비례하는 기술 — 계산 제외")
        return ("fixed", 0)

    # ---- 위력 (basePowerCallback)
    bp = _base_power_callback(c, hit)
    if not bp:
        return ("fixed", 0)
    bp = max(1, trunc(bp))

    # ---- 위력 이벤트 (BasePower)
    bp = _base_power_event(c, bp)
    if not bp:
        return ("fixed", 0)
    bp = max(1, bp)
    c.base_power = bp

    # ---- 공격·방어 실수치
    attacker = dfn if m.get("overrideOffensivePokemon") == "target" else atk
    defender = atk if m.get("overrideDefensivePokemon") == "source" else dfn
    phys = m["category"] == "Physical"
    atk_stat = m.get("overrideOffensiveStat") or ("atk" if phys else "spa")
    def_stat = m.get("overrideDefensiveStat") or ("def" if phys else "spd")
    ab_ = attacker.boost(atk_stat)
    db_ = defender.boost(def_stat)
    # Unaware (천진)
    if c.ab(dfn, "unaware"):
        ab_ = 0
    if c.ab(atk, "unaware"):
        db_ = 0
    ign_neg_off = bool(m.get("ignoreNegativeOffensive")) or crit
    ign_pos_def = bool(m.get("ignorePositiveDefensive")) or crit
    if m.get("ignoreOffensive") or (ign_neg_off and ab_ < 0):
        ab_ = 0
    if m.get("ignoreDefensive") or (ign_pos_def and db_ > 0):
        db_ = 0
    A = _boosted(attacker.stats[atk_stat], ab_)
    D = _boosted(defender.stats[def_stat], db_)
    A = _modify_atk(c, A, "atk" if phys else "spa")
    D = _modify_def(c, D, def_stat)

    lv = atk.level
    base = trunc(trunc(trunc(trunc(2 * lv / 5 + 2) * bp * A) / D) / 50)
    return _modify_damage(c, base, crit)


def _boosted(stat, b):
    if b >= 0:
        return trunc(stat * BOOST_TABLE[b])
    return trunc(stat / BOOST_TABLE[-b])


def _base_power_callback(c: _Calc, hit: int):
    m, atk, dfn = c.m, c.atk, c.dfn
    mid, bp = m["id"], m["basePower"]
    if mid == "fling":
        it = c.dex.items.get(atk.item) or {}
        return (it.get("fling") or {}).get("basePower", 0)
    if mid == "acrobatics":
        return bp * 2 if not atk.item else bp
    if mid in ("eruption", "waterspout", "dragonenergy"):
        return bp * atk.hp / atk.maxhp
    if mid in ("flail", "reversal"):
        r = max(trunc(atk.hp * 48 / atk.maxhp), 1)
        return 200 if r < 2 else 150 if r < 5 else 100 if r < 10 else 80 if r < 17 else 40 if r < 33 else 20
    if mid in ("lowkick", "grassknot"):
        w = dfn.weighthg
        return 120 if w >= 2000 else 100 if w >= 1000 else 80 if w >= 500 else 60 if w >= 250 else 40 if w >= 100 else 20
    if mid in ("heavyslam", "heatcrash"):
        tw, pw = dfn.weighthg, atk.weighthg
        return 120 if pw >= tw * 5 else 100 if pw >= tw * 4 else 80 if pw >= tw * 3 else 60 if pw >= tw * 2 else 40
    if mid in ("hex", "infernalparade"):
        return bp * 2 if dfn.status else bp
    if mid in ("storedpower", "powertrip"):
        return bp + 20 * atk.positive_boosts()
    if mid == "electroball":
        t = c.spe_stat(dfn, c.ctx.def_side)
        ratio = trunc(c.spe_stat(atk, c.ctx.atk_side) / t) if t else 0
        return [40, 60, 80, 120, 150][min(ratio, 4)]
    if mid == "gyroball":
        s = c.spe_stat(atk, c.ctx.atk_side)
        p = trunc(25 * c.spe_stat(dfn, c.ctx.def_side) / s) + 1 if s else 1
        return min(p, 150)
    if mid == "hardpress":
        return trunc(trunc((100 * (100 * trunc(dfn.hp * 4096 / dfn.maxhp)) + 2048 - 1) / 4096) / 100) or 1
    if mid == "lastrespects":
        return 50 + 50 * c.field.sides[c.ctx.atk_side].fainted
    if mid == "ragefist":
        return min(350, 50 + 50 * atk.times_attacked)
    if mid == "risingvoltage":
        return bp * 2 if c.field.terrain == "electricterrain" and c.grounded(dfn) else bp
    if mid == "tripleaxel":
        return 20 * hit
    if mid == "avalanche":
        return bp * 2 if c.ctx.hurt_by_target else bp
    if mid == "payback":
        return bp * 2 if c.ctx.target_moved_first else bp
    if mid == "assurance":
        return bp * 2 if c.ctx.target_hurt_this_turn else bp
    if mid in ("stompingtantrum", "temperflare"):
        return bp * 2 if c.ctx.last_move_failed else bp
    return bp


def _base_power_event(c: _Calc, bp: int) -> int:
    m, atk, dfn, H = c.m, c.atk, c.dfn, c.H
    mid, t = m["id"], m["type"]
    hs = []
    # 기술 자체 (우선도 0, subOrder 0)
    if mid == "facade" and atk.status and atk.status != "slp":
        hs.append(H(0, atk, 0, 2))
    if mid == "knockoff" and dfn.item and _removable(c, dfn):
        hs.append(H(0, atk, 0, 1.5))
    if mid in ("venoshock", "barbbarrage") and dfn.status in ("psn", "tox"):
        hs.append(H(0, atk, 0, 2))
    if mid in ("solarbeam", "solarblade") and c.eff_weather(atk) in ("raindance", "sandstorm", "snowscape"):
        hs.append(H(0, atk, 0, 0.5))
    if mid == "expandingforce" and c.field.terrain == "psychicterrain" and c.grounded(atk):
        hs.append(H(0, atk, 0, 1.5))
    if mid == "mistyexplosion" and c.field.terrain == "mistyterrain" and c.grounded(atk):
        hs.append(H(0, atk, 0, 1.5))
    if mid == "gravapple" and c.field.gravity:
        hs.append(H(0, atk, 0, 1.5))
    if mid == "lashout" and c.ctx.stats_lowered:
        hs.append(H(0, atk, 0, 2))
    # 공격 측 특성
    a = atk.ability
    if a == "technician":
        if bp <= 60:   # 우선도 30 — 가장 먼저 실행되므로 배율이 아직 1
            hs.append(H(30, atk, SUB_ABILITY, 1.5))
    if a in ATE_ABILITIES and m.get("typeChangerBoosted"):
        hs.append(H(23, atk, SUB_ABILITY, 4915, 4096))
    if a == "ironfist" and m["flags"].get("punch"):
        hs.append(H(23, atk, SUB_ABILITY, 4915, 4096))
    if a == "reckless" and (m.get("recoil") or m.get("hasCrashDamage")):
        hs.append(H(23, atk, SUB_ABILITY, 4915, 4096))
    if a == "rivalry" and atk.gender and dfn.gender:
        hs.append(H(24, atk, SUB_ABILITY, 1.25 if atk.gender == dfn.gender else 0.75))
    if a == "toughclaws" and m["flags"].get("contact"):
        hs.append(H(21, atk, SUB_ABILITY, 5325, 4096))
    if a == "sheerforce" and (m.get("hasSheerForce") or m.get("hasSheerForceBoost")):
        hs.append(H(21, atk, SUB_ABILITY, 5325, 4096))
    if a == "sandforce" and c.field_weather() == "sandstorm" and t in ("Rock", "Ground", "Steel"):
        hs.append(H(21, atk, SUB_ABILITY, 5325, 4096))
    if a == "analytic" and c.ctx.moved_last:
        hs.append(H(21, atk, SUB_ABILITY, 5325, 4096))
    if a == "supremeoverlord":
        n = min(5, c.field.sides[c.ctx.atk_side].fainted)
        if n:
            hs.append(H(21, atk, SUB_ABILITY, [4096, 4506, 4915, 5325, 5734, 6144][n], 4096))
    if a == "strongjaw" and m["flags"].get("bite"):
        hs.append(H(19, atk, SUB_ABILITY, 1.5))
    if a == "megalauncher" and m["flags"].get("pulse"):
        hs.append(H(19, atk, SUB_ABILITY, 1.5))
    if a == "sharpness" and m["flags"].get("slicing"):
        hs.append(H(19, atk, SUB_ABILITY, 1.5))
    if a == "punkrock" and m["flags"].get("sound"):
        hs.append(H(0, atk, SUB_ABILITY, 5325, 4096))
    # 방어 측 특성 (onSourceBasePower)
    if c.ab(dfn, "dryskin") and t == "Fire":
        hs.append(H(17, dfn, SUB_ABILITY, 1.25))
    # 필드 위 페어리오라
    if t == "Fairy":
        for p in c.actives:
            if p and not p.fainted and p.ability == "fairyaura":
                hs.append(H(20, p, SUB_ABILITY, 5448, 4096))
                break
    # 도구
    it = atk.item
    if it == "muscleband" and m["category"] == "Physical":
        hs.append(H(16, atk, SUB_ITEM, 4505, 4096))
    if it == "wiseglasses" and m["category"] == "Special":
        hs.append(H(16, atk, SUB_ITEM, 4505, 4096))
    if TYPE_ITEMS.get(it) == t:
        hs.append(H(15, atk, SUB_ITEM, 4915, 4096))
    # 도우미 (더블)
    if c.ctx.helping_hand:
        hs.append(H(10, atk, SUB_VOLATILE, 1.5))
    # 필드
    terr = c.field.terrain
    if terr == "electricterrain" and t == "Electric" and c.grounded(atk):
        hs.append(H(6, None, SUB_FIELD, 5325, 4096))
    if terr == "psychicterrain" and t == "Psychic" and c.grounded(atk):
        hs.append(H(6, None, SUB_FIELD, 5325, 4096))
    if terr == "grassyterrain":
        if mid in ("earthquake", "bulldoze", "magnitude") and c.grounded(dfn):
            hs.append(H(6, None, SUB_FIELD, 0.5))
        elif t == "Grass" and c.grounded(atk):
            hs.append(H(6, None, SUB_FIELD, 5325, 4096))
    if terr == "mistyterrain" and t == "Dragon" and c.grounded(dfn):
        hs.append(H(6, None, SUB_FIELD, 0.5))
    return c.run_event(bp, hs)


def _removable(c: _Calc, mon: Mon) -> bool:
    it = c.dex.items.get(mon.item)
    if not it:
        return False
    ms = it.get("megaStone") or {}
    if ms and (mon.sp_data["name"] in ms or mon.base_species in ms):
        return False
    return True


def _modify_atk(c: _Calc, A: int, stat: str) -> int:
    m, atk, dfn, H = c.m, c.atk, c.dfn, c.H
    t = m["type"]
    hs = []
    a = atk.ability
    if stat == "atk":
        if a in ("hugepower", "purepower"):
            hs.append(H(5, atk, SUB_ABILITY, 2))
        if a == "guts" and atk.status:
            hs.append(H(5, atk, SUB_ABILITY, 1.5))
        if a == "hustle":
            hs.append(H(5, atk, SUB_ABILITY, 1.5, kind="direct"))
    if stat == "spa":
        if a == "solarpower" and c.eff_weather(atk) == "sunnyday":
            hs.append(H(5, atk, SUB_ABILITY, 1.5))
        if a in ("plus", "minus") and c.ctx.atk_ally is not None and c.ctx.atk_ally.ability in ("plus", "minus"):
            hs.append(H(5, atk, SUB_ABILITY, 1.5))
    if a in PINCH and PINCH[a] == t and atk.hp <= atk.maxhp / 3:
        hs.append(H(5, atk, SUB_ABILITY, 1.5))
    if a == "firemane" and t == "Fire":
        hs.append(H(5, atk, SUB_ABILITY, 1.5))
    if a == "waterbubble" and t == "Water":
        hs.append(H(0, atk, SUB_ABILITY, 2))
    if "flashfire" in atk.volatiles and t == "Fire" and a == "flashfire":
        hs.append(H(5, atk, SUB_VOLATILE, 1.5))
    if atk.item == "lightball" and atk.base_species == "Pikachu":
        hs.append(H(1, atk, SUB_ITEM, 2))
    # 상대 특성 (onSourceModifyAtk/SpA)
    if c.ab(dfn, "thickfat") and t in ("Fire", "Ice"):
        hs.append(H(6 if stat == "atk" else 5, dfn, SUB_ABILITY, 0.5))
    if c.ab(dfn, "heatproof") and t == "Fire":
        hs.append(H(6 if stat == "atk" else 5, dfn, SUB_ABILITY, 0.5))
    if c.ab(dfn, "purifyingsalt") and t == "Ghost":
        hs.append(H(6 if stat == "atk" else 5, dfn, SUB_ABILITY, 0.5))
    if c.ab(dfn, "waterbubble") and t == "Fire":
        hs.append(H(5, dfn, SUB_ABILITY, 0.5))
    return c.run_event(A, hs)


def _modify_def(c: _Calc, D: int, stat: str) -> int:
    dfn, H = c.dfn, c.H
    hs = []
    if stat == "def":
        if c.ab(dfn, "furcoat"):
            hs.append(H(6, dfn, SUB_ABILITY, 2))
        if c.ab(dfn, "marvelscale") and dfn.status:
            hs.append(H(6, dfn, SUB_ABILITY, 1.5))
        if dfn.has_type("Ice") and c.eff_weather(dfn) == "snowscape":
            hs.append(H(10, None, SUB_FIELD, 1.5, kind="direct"))
    if stat == "spd":
        if dfn.has_type("Rock") and c.eff_weather(dfn) == "sandstorm":
            hs.append(H(10, None, SUB_FIELD, 1.5, kind="direct"))
    return c.run_event(D, hs)


def _modify_damage(c: _Calc, base: int, crit: bool) -> list:
    m, atk, dfn, dex, H = c.m, c.atk, c.dfn, c.dex, c.H
    t, mid = m["type"], m["id"]
    base += 2
    if c.spread:
        base = modify(base, mod4096(0.75))
    # 날씨 (priorityEvent — 처음 값을 돌려준 핸들러에서 끝)
    wm = 4096
    w = c.eff_weather(dfn)
    if mid == "hydrosteam" and c.eff_weather(atk) == "sunnyday":
        wm = mod4096(1.5)
    elif w == "sunnyday":
        wm = mod4096(1.5) if t == "Fire" else mod4096(0.5) if t == "Water" else 4096
    elif w == "raindance":
        wm = mod4096(1.5) if t == "Water" else mod4096(0.5) if t == "Fire" else 4096
    if wm != 4096:
        base = modify(base, wm)
    if crit:
        base = trunc(base * 1.5)

    # 타입 상성
    tm = 0
    for dt in dfn.types:
        e = dex.effectiveness(t, dt)
        if mid == "freezedry" and dt == "Water":
            e = 1
        if mid == "flyingpress":
            e += dex.effectiveness("Flying", dt)
        if dfn.item == "ironball" and t == "Ground" and dt == "Flying" and not c.field.gravity:
            e = 0
        tm += e
    tm = max(-6, min(6, tm))
    c.type_mod = tm

    stab = 4096
    if t != "???" and (m.get("forceSTAB") or atk.has_type(t)):
        stab = mod4096(2) if atk.ability == "adaptability" else mod4096(1.5)

    rolls = []
    for r in range(16):
        d = trunc(trunc(base * (100 - r)) / 100)
        if stab != 4096:
            d = modify(d, stab)
        if tm > 0:
            for _ in range(tm):
                d *= 2
        elif tm < 0:
            for _ in range(-tm):
                d = trunc(d / 2)
        if atk.status == "brn" and m["category"] == "Physical" and atk.ability != "guts" and mid != "facade":
            d = modify(d, mod4096(0.5))
        d = _modify_damage_event(c, d, crit, tm)
        if not d:
            d = 1
        rolls.append(d & 0xFFFF)
    return rolls


def _modify_damage_event(c: _Calc, d: int, crit: bool, tm: int) -> int:
    m, atk, dfn, H = c.m, c.atk, c.dfn, c.H
    t = m["type"]
    hs = []
    # 공격 측
    if atk.item == "lifeorb":
        hs.append(H(0, atk, SUB_ITEM, 5324, 4096))
    if atk.item == "expertbelt" and tm > 0:
        hs.append(H(0, atk, SUB_ITEM, 4915, 4096))
    if atk.ability == "sniper" and crit:
        hs.append(H(0, atk, SUB_ABILITY, 1.5))
    if atk.ability == "tintedlens" and tm < 0:
        hs.append(H(0, atk, SUB_ABILITY, 2))
    # 방어 측 (onSourceModifyDamage)
    if (c.ab(dfn, "multiscale") or dfn.ability == "shadowshield") and dfn.hp >= dfn.maxhp:
        hs.append(H(0, dfn, SUB_ABILITY, 0.5))
    if (c.ab(dfn, "filter") or c.ab(dfn, "solidrock") or dfn.ability == "prismarmor") and tm > 0:
        hs.append(H(0, dfn, SUB_ABILITY, 0.75))
    if c.ab(dfn, "fluffy"):
        f = 1.0
        if t == "Fire":
            f *= 2
        if m["flags"].get("contact"):
            f /= 2
        hs.append(H(0, dfn, SUB_ABILITY, f))
    if c.ab(dfn, "icescales") and m["category"] == "Special":
        hs.append(H(0, dfn, SUB_ABILITY, 0.5))
    if c.ab(dfn, "punkrock") and m["flags"].get("sound"):
        hs.append(H(0, dfn, SUB_ABILITY, 0.5))
    if RESIST_BERRIES.get(dfn.item) == t and tm > 0:
        hs.append(H(0, dfn, SUB_ITEM, 0.5))
    if dfn.item == "chilanberry" and t == "Normal":
        hs.append(H(0, dfn, SUB_ITEM, 0.5))
    # 벽 (상대 진영)
    side = c.field.sides[c.ctx.def_side]
    if not crit and not m.get("infiltrates") and c.ctx.atk_side != c.ctx.def_side:
        wall = mod4096(2732, 4096) if c.field.doubles else mod4096(0.5)
        cat = m["category"]
        if side.reflect and cat == "Physical" or side.lightscreen and cat == "Special":
            hs.append((0, 0, SUB_SIDE, "chain", wall))
        elif side.auroraveil:
            hs.append((0, 0, SUB_SIDE, "chain", wall))
    # 프렌드가드 (상대 파트너)
    if c.ctx.def_ally is not None and not c.ctx.def_ally.fainted and c.ctx.def_ally.ability == "friendguard" \
            and not (c.mold):
        hs.append(H(0, c.ctx.def_ally, SUB_ABILITY, 0.75))
    return c.run_event(d, hs)


# ======================================================================== 결과 해석
def percent_range(res: DamageResult, mon: Mon, crit=False) -> tuple:
    rolls = res.crit_rolls if crit else res.rolls
    if not rolls:
        return (0.0, 0.0)
    lo = sum(min(h) for h in rolls[:min(res.hits)])
    hi = sum(max(h) for h in rolls[:max(res.hits)])
    return (100.0 * lo / mon.maxhp, 100.0 * hi / mon.maxhp)


def damage_distribution(res: DamageResult, crit_mix=True) -> dict:
    """한 번 맞을 때의 데미지 분포 {데미지: 확률}. 급소·다단히트 타수 확률을 섞는다."""
    if res.immune or not res.rolls:
        return {0: 1.0}
    hit_probs = _hit_count_probs(res)
    cc = res.crit_chance if crit_mix else 0.0
    total = {}
    for n, pn in hit_probs.items():
        dist = {0: 1.0}
        for h in range(n):
            per = {}
            for rolls, p in ((res.rolls[min(h, len(res.rolls) - 1)], 1 - cc),
                             (res.crit_rolls[min(h, len(res.crit_rolls) - 1)] if res.crit_rolls else [], cc)):
                if p <= 0 or not rolls:
                    continue
                for v in rolls:
                    per[v] = per.get(v, 0) + p / 16
            nd = {}
            for a, pa in dist.items():
                for b, pb in per.items():
                    nd[a + b] = nd.get(a + b, 0) + pa * pb
            dist = nd
        for k, v in dist.items():
            total[k] = total.get(k, 0) + v * pn
    return total


def _hit_count_probs(res: DamageResult) -> dict:
    hs = res.hits
    if len(hs) == 1:
        return {hs[0]: 1.0}
    if hs == [2, 3, 4, 5]:
        return {2: 0.35, 3: 0.35, 4: 0.15, 5: 0.15}
    return {h: 1 / len(hs) for h in hs}


def ko_chance(res: DamageResult, hp: int, n_hits: int = 1) -> float:
    """n 번 맞혀서 hp 이상 줄 확률 (명중률 제외, 같은 기술 반복 가정)."""
    if res.immune or not res.rolls:
        return 0.0
    one = damage_distribution(res)
    dist = {0: 1.0}
    for _ in range(n_hits):
        nd = {}
        for a, pa in dist.items():
            if a >= hp:
                nd[a] = nd.get(a, 0) + pa
                continue
            for b, pb in one.items():
                nd[a + b] = nd.get(a + b, 0) + pa * pb
        dist = nd
    return sum(p for d, p in dist.items() if d >= hp)
