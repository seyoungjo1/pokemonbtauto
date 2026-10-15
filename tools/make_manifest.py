#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""manifest.json 만들기 — 업데이트할 파일 목록과 SHA-256, 데이터·모델 버전.

GitHub Actions 가 main 에 무언가 올라올 때마다 돌린다 (사용자 PC 의 updater.py 가 이것을 본다).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["app.py", "updater.py", "run.bat", "밤샘학습.bat", "requirements.txt", "version.txt", "README.md", "FEATURES.md"]
INCLUDE_DIRS = ["engine", "templates", "static", "data", "models", "features", "tests", "tools", "reports"]
# models/history 의 지난 신경망은 리그 상대이므로 사용자 PC 에도 내려받는다
SKIP_SUFFIX = {".pyc"}
SKIP_PARTS = {"__pycache__"}


def main():
    files = {}
    paths = [ROOT / p for p in INCLUDE if (ROOT / p).is_file()]
    for d in INCLUDE_DIRS:
        if (ROOT / d).is_dir():
            paths += [p for p in (ROOT / d).rglob("*") if p.is_file()]
    for p in sorted(paths):
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix in SKIP_SUFFIX or SKIP_PARTS & set(p.parts) or rel.endswith("_candidate.json"):
            continue
        data = p.read_bytes()
        files[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    data_versions = {}
    for meta in sorted((ROOT / "data").glob("*/meta.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        data_versions[meta.parent.name] = {"builtAt": m.get("builtAt"), "psVersion": m.get("psVersion"),
                                           "psCommit": m.get("psCommit"), "counts": m.get("counts")}
    models = {}
    for mp in sorted((ROOT / "models").glob("*.json")):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            models[mp.stem] = {"created": m.get("created"), "games": m.get("games"),
                               "league": {k: v.get("winrate") for k, v in (m.get("league") or {}).items()}}
        except Exception:
            pass
    version = (ROOT / "version.txt").read_text(encoding="utf-8").strip() if (ROOT / "version.txt").is_file() else "0"
    out = {"version": version, "files": files, "dataVersions": data_versions, "models": models}
    (ROOT / "manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"manifest.json: 파일 {len(files)}개 · 데이터 {list(data_versions)} · 모델 {list(models)}")


if __name__ == "__main__":
    main()
