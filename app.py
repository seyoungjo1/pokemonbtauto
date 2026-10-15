#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""포켓몬 챔피언스 배틀 AI — 웹 화면 서버.

실행: run.bat (윈도우) 또는 python app.py → 브라우저가 http://127.0.0.1:8730 으로 열린다.
"""
import importlib.util
import json
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8730
REQUIRED = {"flask": "flask", "numpy": "numpy"}


def _installed(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


def ensure_packages():
    missing = [pkg for mod, pkg in REQUIRED.items() if not _installed(mod)]
    if missing:
        print("필요한 패키지를 설치합니다:", ", ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing], check=False)
        # 방금 설치한 패키지를 이 프로세스에서 바로 쓰도록 경로 갱신
        import site
        try:
            site.addsitedir(site.getusersitepackages())
        except Exception:
            pass
        importlib.invalidate_caches()


ensure_packages()

from flask import Flask, jsonify, request, send_file, render_template, Response  # noqa: E402

sys.path.insert(0, str(HERE))
from engine.dex import Dex, to_id, DATA_DIR  # noqa: E402
from engine import advisor, builder, model as M  # noqa: E402
from engine.damage import Ctx, calc, ko_chance, percent_range  # noqa: E402
from engine.mon import FieldState, Mon, SideState  # noqa: E402
from engine.team import export_paste, parse_paste  # noqa: E402
from engine.psbridge import find_ps_dir  # noqa: E402

app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))
app.config["JSON_AS_ASCII"] = False

USER = HERE / "userdata"
CACHE = USER / "cache" / "sprites"
TEAMS = USER / "teams"
BATTLES = USER / "battles"
CONFIG = HERE / "config.json"
for d in (CACHE, TEAMS, BATTLES):
    d.mkdir(parents=True, exist_ok=True)

_DEX = {}


def cfg() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cfg(c):
    CONFIG.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")


def data_dirs():
    root = HERE / "data"
    return sorted([p for p in root.iterdir() if (p / "species.json").is_file()]) if root.is_dir() else []


def dex() -> Dex:
    mod = cfg().get("mod") or "champions"
    path = HERE / "data" / mod
    if not (path / "species.json").is_file():
        path = DATA_DIR
    key = str(path)
    if key not in _DEX:
        _DEX[key] = Dex(path)
    return _DEX[key]


def ok(**kw):
    return jsonify({"ok": True, **kw})


def err(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


# ======================================================================== 화면
@app.route("/")
def index():
    return render_template("index.html", version=app_version())


def app_version():
    p = HERE / "version.txt"
    return p.read_text(encoding="utf-8").strip() if p.is_file() else "dev"


LEAGUE_KO = {"showdown_random": "쇼다운 기본 AI", "max_damage": "최대 데미지 AI", "heuristic": "선생님 AI",
             "best": "이전 최고 신경망", "random": "완전 무작위"}


def _overnight_state():
    p = USER / "overnight" / "state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except Exception:
        return {}


@app.route("/api/meta")
def api_meta():
    d = dex()
    models = {}
    from engine import nets as _nets
    state = _overnight_state()
    for gt in ("singles", "doubles"):
        p = _nets.path_for(gt)
        if not p.is_file():
            continue
        try:
            nd = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = nd.get("info") or {}
        last = next((e for e in reversed(state.get("progress", [])) if e.get("gameType") == gt and e.get("kept")), None)
        models[gt] = {"created": nd.get("created"), "games": info.get("games") or nd.get("games"),
                      "decisions": info.get("decisions"), "metrics": info.get("metrics") or {},
                      "inputs": nd.get("ds", 0) + nd.get("do", 0) // 2, "surgery": info.get("surgery"),
                      "league": {k: {**v, "ko": LEAGUE_KO.get(k, k), "wins": round(v.get("winrate", 0) * v.get("played", 0))}
                                 for k, v in ((last or {}).get("league") or {}).items()}}
    return ok(version=app_version(), data=d.meta, mod=cfg().get("mod") or "champions",
              dataDirs=[{"mod": p.name, "meta": json.loads((p / "meta.json").read_text(encoding="utf-8"))}
                        for p in data_dirs()],
              formats=d.formats, typesKo=d.types_ko,
              natures=[{"id": k, "ko": v.get("ko"), "name": v["name"], "plus": v["plus"], "minus": v["minus"]}
                       for k, v in d.natures.items()],
              models=models)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    c = cfg()
    body = request.get_json(force=True) or {}
    for k in ("mod", "lang", "gameType"):
        if k in body:
            c[k] = body[k]
    save_cfg(c)
    return ok(settings=c)


# ======================================================================== 검색 · 정보
@app.route("/api/search")
def api_search():
    d = dex()
    kind = request.args.get("kind", "species")
    q = request.args.get("q", "")
    within = None
    if kind == "moves" and request.args.get("species"):
        sid = d.find("species", request.args["species"])
        sp = d.species.get(sid or "", {})
        within = d.learnsets.get(sid) or d.learnsets.get(to_id(sp.get("changesFrom") or sp.get("baseSpecies") or ""))
    if kind == "abilities" and request.args.get("species"):
        sp = d.species.get(d.find("species", request.args["species"]) or "", {})
        within = [to_id(a) for a in sp.get("abilities", {}).values()]
    res = d.search(kind, q, limit=int(request.args.get("limit", 30)), within=within)
    if kind == "species" and request.args.get("noMega"):
        res = [r for r in res if not r.get("isMega")]
    out = []
    for r in res:
        o = {"id": r["id"], "name": r["name"], "ko": r.get("ko", r["name"])}
        if kind == "species":
            o.update(types=r["types"], baseStats=r["baseStats"], isMega=r.get("isMega"))
        elif kind == "moves":
            o.update(type=r["type"], category=r["category"], basePower=r.get("basePower"),
                     accuracy=r.get("accuracy"), desc=r.get("shortDesc", ""))
        else:
            o.update(desc=r.get("shortDesc", ""))
        out.append(o)
    return ok(results=out)


@app.route("/api/species/<sid>")
def api_species(sid):
    d = dex()
    sid = d.find("species", sid)
    if not sid:
        return err("없는 포켓몬")
    sp = d.species[sid]
    base = to_id(sp.get("changesFrom") or sp.get("baseSpecies") or sid)
    learn = d.learnsets.get(sid) or d.learnsets.get(base) or []
    megas = [to_id(f) for f in d.species.get(base, sp).get("otherFormes", []) if d.species.get(to_id(f), {}).get("isMega")]
    return ok(species={**sp, "abilitiesKo": {k: d.abilities.get(to_id(v), {}).get("ko", v) for k, v in sp["abilities"].items()},
                       "abilityIds": {k: to_id(v) for k, v in sp["abilities"].items()},
                       "learnset": [{"id": m, "ko": d.moves[m]["ko"], "type": d.moves[m]["type"],
                                     "category": d.moves[m]["category"], "basePower": d.moves[m].get("basePower")}
                                    for m in learn if m in d.moves],
                       "megas": [{"id": m, "ko": d.species[m]["ko"], "item": d.find("items", d.species[m].get("requiredItem"))}
                                 for m in megas],
                       "sets": d.sets.get(sid, {}).get("sets", [])})


@app.route("/api/names", methods=["POST"])
def api_names():
    """id 목록 → 한글 이름 (화면 표시용)."""
    d = dex()
    body = request.get_json(force=True) or {}
    out = {}
    for kind, ids in body.items():
        tbl = getattr(d, kind, {})
        out[kind] = {i: (tbl.get(i) or {}).get("ko", i) for i in ids}
    return ok(names=out)


# ======================================================================== 그림
SPRITE_SOURCES = [
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/home/{pid}.png",
    "https://play.pokemonshowdown.com/sprites/home-centered/{sid}.png",
    "https://play.pokemonshowdown.com/sprites/gen5/{sid}.png",
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{pid}.png",
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{pid}.png",
]
PLACEHOLDER = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><circle cx="48" cy="48" r="40" '
               'fill="#e9edf3" stroke="#c4ccd8" stroke-width="4"/><path d="M8 48h80" stroke="#c4ccd8" stroke-width="4"/>'
               '<circle cx="48" cy="48" r="11" fill="#fff" stroke="#c4ccd8" stroke-width="4"/></svg>')


def fetch_sprite(sid: str) -> Path | None:
    d = dex()
    sp = d.species.get(sid)
    if not sp:
        return None
    out = CACHE / f"{sid}.png"
    if out.is_file() and out.stat().st_size > 100:
        return out
    miss = CACHE / f"{sid}.miss"
    if miss.is_file() and time.time() - miss.stat().st_mtime < 86400:
        return None
    pid = sp.get("pokeapiId") if sp.get("spriteExact") else None
    for tmpl in SPRITE_SOURCES:
        if "{pid}" in tmpl and not pid:
            continue
        url = tmpl.format(pid=pid, sid=sp.get("spriteid") or sid)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pokemonbtauto"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = r.read()
            if data[:4] == b"\x89PNG":
                out.write_bytes(data)
                return out
        except Exception:
            continue
    # 폼 그림이 없으면 기본 폼 번호로
    base_pid = sp.get("pokeapiId")
    if base_pid and not sp.get("spriteExact"):
        for tmpl in SPRITE_SOURCES:
            if "{pid}" not in tmpl:
                continue
            try:
                with urllib.request.urlopen(tmpl.format(pid=base_pid, sid=""), timeout=8) as r:
                    data = r.read()
                if data[:4] == b"\x89PNG":
                    out.write_bytes(data)
                    return out
            except Exception:
                continue
    miss.write_text("x")
    return None


@app.route("/sprite/<sid>")
def sprite(sid):
    p = fetch_sprite(to_id(sid))
    if p:
        resp = send_file(p, mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=604800"
        return resp
    return Response(PLACEHOLDER, mimetype="image/svg+xml")


SPRITE_JOB = {"running": False, "done": 0, "total": 0, "failed": 0}


@app.route("/api/sprites/download", methods=["POST"])
def api_sprites_download():
    if SPRITE_JOB["running"]:
        return ok(job=SPRITE_JOB)
    ids = list(dex().species)
    SPRITE_JOB.update(running=True, done=0, total=len(ids), failed=0)

    def work():
        for sid in ids:
            if fetch_sprite(sid) is None:
                SPRITE_JOB["failed"] += 1
            SPRITE_JOB["done"] += 1
        SPRITE_JOB["running"] = False
    threading.Thread(target=work, daemon=True).start()
    return ok(job=SPRITE_JOB)


@app.route("/api/sprites/status")
def api_sprites_status():
    return ok(job=SPRITE_JOB, cached=len(list(CACHE.glob("*.png"))))


# ======================================================================== 팀
@app.route("/api/team/parse", methods=["POST"])
def api_team_parse():
    team = parse_paste(dex(), (request.get_json(force=True) or {}).get("text", ""))
    return ok(team=team)


@app.route("/api/team/export", methods=["POST"])
def api_team_export():
    body = request.get_json(force=True) or {}
    return ok(text=export_paste(dex(), body.get("team") or [], body.get("lang", "ko")))


@app.route("/api/team/list")
def api_team_list():
    out = []
    for f in sorted(TEAMS.glob("*.json"), key=lambda p: -p.stat().st_mtime):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
            out.append({"id": f.stem, "name": t.get("name", f.stem), "team": t.get("team", []),
                        "updated": t.get("updated")})
        except Exception:
            pass
    return ok(teams=out)


@app.route("/api/team/save", methods=["POST"])
def api_team_save():
    body = request.get_json(force=True) or {}
    tid = to_id(body.get("id") or body.get("name") or "") or datetime.now().strftime("team%Y%m%d%H%M%S")
    data = {"name": body.get("name") or tid, "team": body.get("team") or [], "updated": datetime.now().isoformat()}
    (TEAMS / f"{tid}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return ok(id=tid)


@app.route("/api/team/delete", methods=["POST"])
def api_team_delete():
    tid = to_id((request.get_json(force=True) or {}).get("id", ""))
    (TEAMS / f"{tid}.json").unlink(missing_ok=True)
    return ok()


@app.route("/api/team/stats", methods=["POST"])
def api_team_stats():
    """세트 목록 → 실능력치 (편집 화면 미리보기)."""
    d = dex()
    out = []
    for s in (request.get_json(force=True) or {}).get("team") or []:
        try:
            m = builder.mon_from_set(d, s)
            out.append({"stats": m.stats, "species": m.species, "mega": m.mega})
        except Exception as e:
            out.append({"error": str(e)})
    return ok(stats=out)


# ======================================================================== 추천
@app.route("/api/advise", methods=["POST"])
def api_advise():
    st = request.get_json(force=True) or {}
    try:
        return ok(**advisor.advise(dex(), st))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return err(f"{type(e).__name__}: {e}", 500)


@app.route("/api/bring", methods=["POST"])
def api_bring():
    b = request.get_json(force=True) or {}
    gt = b.get("gameType", "singles")
    team = [s for s in (b.get("team") or []) if s.get("species") in dex().species]
    opp = [dex().find("species", x) for x in (b.get("oppPreview") or []) if dex().find("species", x)]
    out = builder.recommend_bring(dex(), team, opp, gt)
    # 학습된 선출망: 내 선출(선봉 포함) × 상대 선출 전부의 예측 승률표 → 균형해
    import numpy as np
    from engine import bringnet as BN
    net = BN.load(gt)
    if net is not None and len(team) >= (4 if gt == "doubles" else 3) and len(opp) >= 3:
        try:
            mu, M, p, q, v = BN.decide(net, dex(), team, opp, gt)
            ko = lambda s: dex().species[s["species"]]["ko"]
            rows = []
            for i in np.argsort(-p)[:6]:
                if p[i] < 0.01 and rows:
                    break
                o = mu.oa[int(i)]
                worst = int(np.argmin(M[i]))
                rows.append({"order": [int(x) for x in o], "names": [ko(team[x]) for x in o], "prob": round(float(p[i]), 3),
                             "value": round(float(M[i] @ q), 3), "worst": round(float(M[i].min()), 3),
                             "worstVs": [dex().species[opp[x]]["ko"] for x in mu.ob[worst]]})
            likely = [{"names": [dex().species[opp[x]]["ko"] for x in mu.ob[int(j)]], "prob": round(float(q[j]), 3)}
                      for j in np.argsort(-q)[:3]]
            out["net"] = {"value": round(v, 3), "rows": rows, "oppLikely": likely, "pairs": int(M.size),
                          "info": net.info.get("arena")}
        except Exception as e:
            out["netError"] = f"{type(e).__name__}: {e}"
    return ok(**out)


@app.route("/api/build/<what>", methods=["POST"])
def api_build(what):
    b = request.get_json(force=True) or {}
    d = dex()
    s = b.get("set") or {}
    if not s.get("species"):
        return err("포켓몬을 먼저 고르세요")
    if what == "spread":
        return ok(**builder.recommend_spread(d, s, goal=b.get("goal", "auto")))
    if what == "moves":
        return ok(**builder.recommend_moves(d, s))
    if what == "items":
        return ok(**builder.recommend_items(d, s, used_items=b.get("usedItems") or []))
    return err("알 수 없는 추천")


@app.route("/api/calc", methods=["POST"])
def api_calc():
    b = request.get_json(force=True) or {}
    d = dex()
    try:
        a = builder.mon_from_set(d, b["attacker"]) if not b["attacker"].get("noMega") else Mon.from_dict(d, b["attacker"])
        t = builder.mon_from_set(d, b["defender"]) if not b["defender"].get("noMega") else Mon.from_dict(d, b["defender"])
        for m, k in ((a, "attacker"), (t, "defender")):
            src = b[k]
            if src.get("hpPct") is not None:
                m.set_hp_pct(float(src["hpPct"]))
            m.status = src.get("status", "") or ""
            m.boosts = {x: int(v) for x, v in (src.get("boosts") or {}).items() if v}
        f = b.get("field") or {}
        field = FieldState(game_type=f.get("gameType", "singles"), weather=f.get("weather", ""),
                           terrain=f.get("terrain", ""), gravity=bool(f.get("gravity")), trick_room=bool(f.get("trickRoom")),
                           sides=[SideState(**{k: bool(v) for k, v in (f.get("atkSide") or {}).items() if k in SideState.__dataclass_fields__}),
                                  SideState(**{k: bool(v) for k, v in (f.get("defSide") or {}).items() if k in SideState.__dataclass_fields__})])
        out = []
        for mid in (b.get("moves") or a.moves):
            r = calc(d, a, t, mid, field, Ctx(helping_hand=bool(f.get("helpingHand")), spread=f.get("spread")))
            lo, hi = percent_range(r, t)
            clo, chi = percent_range(r, t, crit=True)
            out.append({"move": d.moves.get(r.move, {}).get("ko", mid), "id": r.move, "type": r.move_type,
                        "category": r.category, "bp": r.base_power,
                        "rolls": r.rolls[0] if r.rolls else [], "min": round(lo, 1), "max": round(hi, 1),
                        "critMin": round(clo, 1), "critMax": round(chi, 1),
                        "ko": [round(ko_chance(r, t.hp, n), 3) for n in (1, 2, 3, 4)],
                        "immune": r.immune, "blocked": r.blocked, "notes": r.notes, "critChance": r.crit_chance})
        return ok(results=out, attacker={"stats": a.stats, "hp": a.hp}, defender={"stats": t.stats, "hp": t.hp, "maxhp": t.maxhp})
    except Exception as e:
        return err(f"{type(e).__name__}: {e}")


# ======================================================================== 대전 기록 · 사후 분석
@app.route("/api/battle/save", methods=["POST"])
def api_battle_save():
    """대전 진행 상황(턴마다의 상태·고른 행동·예측 승률)과 결과를 저장한다."""
    b = request.get_json(force=True) or {}
    bid = to_id(b.get("id") or "") or datetime.now().strftime("b%Y%m%d%H%M%S")
    b["id"] = bid
    b["saved"] = datetime.now().isoformat()
    (BATTLES / f"{bid}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    return ok(id=bid)


@app.route("/api/battle/list")
def api_battle_list():
    out = []
    for f in sorted(BATTLES.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:200]:
        try:
            b = json.loads(f.read_text(encoding="utf-8"))
            out.append({"id": b["id"], "saved": b.get("saved"), "result": b.get("result"),
                        "gameType": b.get("gameType"), "turns": len(b.get("turns") or []),
                        "opp": b.get("oppPreview") or [], "myTeam": [s.get("species") for s in b.get("myTeam") or []]})
        except Exception:
            pass
    return ok(battles=out)


@app.route("/api/battle/report")
def api_battle_report():
    """내 실제 대전들: 예측 승률이 결과와 얼마나 맞았는지(보정), 자주 만난 상대, 승률."""
    d = dex()
    rows, wins, total = [], 0, 0
    opp_count, calib = {}, [[0, 0] for _ in range(5)]
    for f in BATTLES.glob("*.json"):
        try:
            b = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if b.get("result") not in ("win", "lose"):
            continue
        total += 1
        y = 1 if b["result"] == "win" else 0
        wins += y
        for s in b.get("oppPreview") or []:
            opp_count[s] = opp_count.get(s, 0) + 1
        for t in b.get("turns") or []:
            p = t.get("winNow")
            if p is None:
                continue
            k = min(4, int(p * 5))
            calib[k][0] += 1
            calib[k][1] += y
    top_opp = sorted(opp_count.items(), key=lambda kv: -kv[1])[:15]
    return ok(total=total, wins=wins,
              calibration=[{"bin": f"{i * 20}~{i * 20 + 20}%", "n": n, "actual": (w / n if n else None)}
                           for i, (n, w) in enumerate(calib)],
              topOpp=[{"species": s, "ko": d.species.get(s, {}).get("ko", s), "n": n} for s, n in top_opp])


# ======================================================================== 학습 · 셀프테스트 정보
@app.route("/api/ai/info")
def api_ai_info():
    led = {}
    p = HERE / "features" / "ledger.json"
    if p.is_file():
        led = json.loads(p.read_text(encoding="utf-8"))
    hist = []
    hp = HERE / "reports" / "training_history.json"
    if hp.is_file():
        hist = json.loads(hp.read_text(encoding="utf-8"))[-40:]
    st = {}
    sp = HERE / "reports" / "selftest.json"
    if sp.is_file():
        st = json.loads(sp.read_text(encoding="utf-8"))
    # 학습 데이터 현황: 사람 기보(리플레이) · 나와 둔 대전 · 패치 감지
    data = {"replays": 0, "replayGames": {}, "vsUser": 0, "patch": None}
    ix = USER / "replays" / "index.json"
    if ix.is_file():
        try:
            data["replays"] = sum(len(v) for v in json.loads(ix.read_text(encoding="utf-8")).get("seen", {}).values())
        except Exception:
            pass
    data["vsUser"] = len(list((USER / "selfplay").glob("*/games_*_vsuser_*.jsonl.gz")))
    pd = HERE / "reports" / "patch_diff.json"
    if pd.is_file():
        try:
            j = json.loads(pd.read_text(encoding="utf-8"))
            data["patch"] = {"changes": j.get("changes", 0), "from": j.get("from"), "to": j.get("to"),
                             "focus": list((j.get("focus") or {}).items())[:12]}
        except Exception:
            pass
    return ok(ledger=led, history=hist, selftest=st, overnight=_overnight_state().get("progress", [])[-40:], data=data)


SELFTEST = {"running": False, "result": None}


@app.route("/api/selftest", methods=["POST"])
def api_selftest():
    if SELFTEST["running"]:
        return ok(running=True)
    SELFTEST["running"] = True

    def work():
        try:
            from tests import selftest
            SELFTEST["result"] = selftest.run(verbose=False)
        except Exception as e:
            SELFTEST["result"] = {"ok": False, "error": str(e)}
        SELFTEST["running"] = False
    threading.Thread(target=work, daemon=True).start()
    return ok(running=True)


@app.route("/api/selftest/status")
def api_selftest_status():
    return ok(**SELFTEST)


# ======================================================================== 시뮬레이션 작업 (파티 짜기 · 선출 시뮬레이션)
import shutil as _shutil  # noqa: E402
import uuid as _uuid  # noqa: E402

JOBS = {}
BOX = USER / "box.json"


def sim_available():
    return bool(_shutil.which("node")) and find_ps_dir() is not None


@app.route("/api/sim/status")
def api_sim_status():
    return ok(available=sim_available(), node=bool(_shutil.which("node")), ps=str(find_ps_dir() or ""),
              hint="" if sim_available() else "시뮬레이션 기능은 Node.js 와 쇼다운이 필요합니다. 밤샘학습.bat 을 한 번 실행하면 준비됩니다.")


def start_job(kind, fn):
    jid = _uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "running": True, "progress": [], "result": None, "error": None,
           "started": time.time()}
    JOBS[jid] = job

    def log(msg):
        job["progress"].append(str(msg))
        del job["progress"][:-60]

    def run():
        try:
            job["result"] = fn(log, job)
        except Exception as e:
            import traceback
            traceback.print_exc()
            job["error"] = f"{type(e).__name__}: {e}"
        job["running"] = False
    threading.Thread(target=run, daemon=True).start()
    return jid


@app.route("/api/job/<jid>")
def api_job(jid):
    j = JOBS.get(jid)
    if not j:
        return err("없는 작업")
    return ok(job={k: v for k, v in j.items()})


@app.route("/api/box", methods=["GET", "POST"])
def api_box():
    if request.method == "POST":
        body = request.get_json(force=True) or {}
        BOX.write_text(json.dumps({"species": body.get("species") or [], "sets": body.get("sets") or {}},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        return ok()
    if BOX.is_file():
        return ok(**json.loads(BOX.read_text(encoding="utf-8")))
    return ok(species=[], sets={})


def _teamopt(gt, box, log, seed=7, n_opp=10, games=2):
    from engine.psbridge import Bridge
    from engine.selfplay import pick_format
    from engine.teamopt import TeamOptimizer
    br = Bridge()
    fmt = pick_format(br, gt)
    opt = TeamOptimizer(dex(), br, fmt, gt, pool=box or None, n_opp=n_opp, games=games, k_orders=4, seed=seed, log=log)
    return br, opt


def _describe_team(team):
    d = dex()
    out = []
    for s in team:
        out.append({**s, "ko": d.species[s["species"]]["ko"],
                    "itemKo": d.items.get(s.get("item") or "", {}).get("ko", ""),
                    "abilityKo": d.abilities.get(s.get("ability") or "", {}).get("ko", ""),
                    "natureKo": d.natures.get(s.get("nature") or "", {}).get("ko", ""),
                    "movesKo": [d.moves.get(m, {}).get("ko", m) for m in s.get("moves") or []]})
    return out


@app.route("/api/team/build", methods=["POST"])
def api_team_build():
    if not sim_available():
        return err("시뮬레이션 기능은 Node.js 와 쇼다운이 필요합니다. 밤샘학습.bat 을 한 번 실행하면 준비됩니다.")
    b = request.get_json(force=True) or {}
    gt = b.get("gameType", "singles")
    box = [dex().find("species", x) for x in (b.get("box") or []) if dex().find("species", x)]
    locked = b.get("locked") or []
    steps = int(b.get("steps", 20))

    def fn(log, job):
        br, opt = _teamopt(gt, box, log)
        try:
            log(f"메타 상대 파티 {len(opt.opponents)}개와 '선출 → 배틀' 로 평가합니다.")
            start = opt.initial_team(fixed=locked)
            job["partial"] = _describe_team(start)

            def prog(step, total, score):
                job["step"] = [step, total, round(score, 4)]
            res = opt.optimize(start, steps=steps, locked=tuple(range(len(locked))), progress=prog)
            return {"team": _describe_team(res["team"]), "score": res["score"], "perOpp": res["perOpp"],
                    "history": res["history"], "evals": res["evals"],
                    "opponents": [[dex().species[s["species"]]["ko"] for s in t] for t in opt.opponents]}
        finally:
            br.close()
    return ok(job=start_job("build", fn))


@app.route("/api/team/one", methods=["POST"])
def api_team_one():
    if not sim_available():
        return err("시뮬레이션 기능은 Node.js 와 쇼다운이 필요합니다. 밤샘학습.bat 을 한 번 실행하면 준비됩니다.")
    b = request.get_json(force=True) or {}
    gt = b.get("gameType", "singles")
    box = [dex().find("species", x) for x in (b.get("box") or []) if dex().find("species", x)]
    team = [s for s in (b.get("team") or []) if s.get("species")][:5]

    def fn(log, job):
        br, opt = _teamopt(gt, box, log)
        try:
            log(f"후보 {len(opt.pool)}마리를 하나씩 넣어 대전시킵니다.")
            res = opt.recommend_one(team, top=6)
            return {"rows": [{**r, "set": _describe_team([r["set"]])[0]} for r in res]}
        finally:
            br.close()
    return ok(job=start_job("one", fn))


@app.route("/api/bring/sim", methods=["POST"])
def api_bring_sim():
    if not sim_available():
        return err("시뮬레이션 기능은 Node.js 와 쇼다운이 필요합니다. 밤샘학습.bat 을 한 번 실행하면 준비됩니다.")
    b = request.get_json(force=True) or {}
    gt = b.get("gameType", "singles")

    def fn(log, job):
        from engine import bring as BR
        from engine.psbridge import Bridge
        from engine.selfplay import pick_format
        br = Bridge()
        try:
            fmt = pick_format(br, gt)
            log("내 선출 후보 × 상대 선출 후보를 끝까지 대전시켜 승률표를 채웁니다…")
            return BR.recommend(dex(), br, fmt, b.get("team") or [], b.get("oppPreview") or [], gt,
                                k_mine=10, k_opp=10, worlds=6, games=6, seed=int(time.time()) % 9999, budget_ms=60000)
        finally:
            br.close()
    return ok(job=start_job("bring", fn))


# ======================================================================== AI 와 대전
PLAY = {"s": None}
PLAY_LOCK = threading.Lock()


@app.route("/api/play/levels")
def api_play_levels():
    from engine.playsession import AI_LEVELS
    from engine import nets
    have = {gt: nets.load_net(gt) is not None for gt in ("singles", "doubles")}
    return ok(levels=[[k, v] for k, v in AI_LEVELS.items()], nets=have, sim=sim_available())


@app.route("/api/play/start", methods=["POST"])
def api_play_start():
    if not sim_available():
        return err("AI 와 대전하려면 Node.js 와 쇼다운이 필요합니다. 밤샘학습.bat 을 한 번 실행하면 준비됩니다.")
    from engine.playsession import PlaySession
    b = request.get_json(force=True) or {}
    gt = b.get("gameType", "singles")
    team = b.get("team") or []
    with PLAY_LOCK:
        if PLAY["s"]:
            try:
                PLAY["s"].close()
            except Exception:
                pass
            PLAY["s"] = None
        if not team:   # 팀이 없으면 메타 팀을 하나 만들어 준다
            from engine.teamgen import TeamGen
            tg = TeamGen(dex(), seed=int(time.time()) % 99991, game_type=gt)
            team = (tg.meta_team(6) if tg.usage else []) or tg.team(6)
        try:
            PLAY["s"] = PlaySession(dex(), gt, team, ai=b.get("ai", "search"))
        except ValueError as e:
            return err(str(e))
        return ok(state=PLAY["s"].state())


@app.route("/api/play/choose", methods=["POST"])
def api_play_choose():
    b = request.get_json(force=True) or {}
    with PLAY_LOCK:
        s = PLAY["s"]
        if not s:
            return err("진행 중인 대전이 없습니다.")
        try:
            return ok(state=s.choose(b.get("choice", "")))
        except ValueError as e:
            return err(str(e))


@app.route("/api/play/state")
def api_play_state():
    s = PLAY["s"]
    return ok(state=s.state() if s else None)


@app.route("/api/play/forfeit", methods=["POST"])
def api_play_forfeit():
    with PLAY_LOCK:
        s = PLAY["s"]
        if s:
            s.forfeit()
            return ok(state=s.state())
    return ok(state=None)


# ======================================================================== 업데이트
@app.route("/api/update/check", methods=["POST"])
def api_update_check():
    import updater
    try:
        return ok(**updater.check())
    except Exception as e:
        return err(f"업데이트 확인 실패: {e}")


@app.route("/api/update/apply", methods=["POST"])
def api_update_apply():
    import updater
    logs = []
    try:
        changed = updater.apply(log=logs.append)
        _DEX.clear()
        return ok(changed=changed, log=logs)
    except Exception as e:
        return err(f"업데이트 실패: {e}")


def open_browser():
    time.sleep(1.2)
    try:
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    print(f"포켓몬 챔피언스 배틀 AI — http://127.0.0.1:{PORT}")
    if "--no-browser" not in sys.argv:
        threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
