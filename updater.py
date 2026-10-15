#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""자동 업데이트 + 실행기 (dsdrive 방식).

GitHub 의 manifest.json(파일 목록과 SHA-256)을 내 PC 의 파일과 비교해, 바뀐 파일만 받는다.
프로그램 파일뿐 아니라 챔피언스 데이터(data/<규칙>/)와 학습 모델(models/)도 같은 방식으로 갱신된다.

  - 전부 받은 뒤에만 덮어쓴다 → 중간에 끊겨도 일부만 바뀌는 일이 없다.
  - 오프라인이면 기존 파일로 그냥 실행한다.
  - 공개 저장소라 토큰이 필요 없다. 비공개로 바꾸면 token.txt 에 GitHub 토큰을 한 줄로 넣는다.
"""
import hashlib
import json
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = "seyoungjo1"
REPO = "pokemonbtauto"
BRANCH = "main"
TOKEN_FILE = HERE / "token.txt"
MANIFEST = HERE / "manifest.json"
KEEP_LOCAL = {"config.json", "token.txt"}   # 사용자 파일은 절대 덮어쓰지 않음


def _token():
    if TOKEN_FILE.is_file():
        t = TOKEN_FILE.read_text(encoding="utf-8").strip()
        return t or None
    return None


def fetch(path: str) -> bytes:
    tok = _token()
    if tok:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}"
        headers = {"Authorization": f"token {tok}", "Accept": "application/vnd.github.raw"}
    else:
        url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{path}"
        headers = {}
    headers["User-Agent"] = "pokemonbtauto-updater"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as r:
        return r.read()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def local_manifest() -> dict:
    if MANIFEST.is_file():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": "0", "files": {}}


def _model_created(path: Path) -> str:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("created", "")
    except Exception:
        return ""


def check() -> dict:
    remote = json.loads(fetch("manifest.json").decode("utf-8"))
    changed = []
    for path, info in remote.get("files", {}).items():
        if path in KEEP_LOCAL:
            continue
        # 내 PC 에서 밤샘 학습한 신경망이 더 새것이면 저장소 것으로 덮어쓰지 않는다
        if path.startswith("models/") and path.endswith(".json") and (HERE / path).is_file():
            stem = Path(path).stem
            remote_created = (remote.get("models", {}).get(stem) or {}).get("created", "")
            if _model_created(HERE / path) >= remote_created:
                continue
        p = HERE / path
        if not p.is_file() or p.stat().st_size != info["size"] or sha256(p) != info["sha256"]:
            changed.append(path)
    loc = local_manifest()
    return {"local": loc.get("version"), "remote": remote.get("version"), "changed": changed,
            "update_available": bool(changed), "dataVersions": remote.get("dataVersions", {}),
            "models": remote.get("models", {}), "localData": loc.get("dataVersions", {})}


def apply(log=print) -> list:
    remote_raw = fetch("manifest.json")
    remote = json.loads(remote_raw.decode("utf-8"))
    info = check()
    if not info["changed"]:
        log(f"최신입니다 (v{remote.get('version')}).")
        return []
    log(f"업데이트: v{info['local']} → v{remote.get('version')} · 파일 {len(info['changed'])}개")
    blobs = {}
    for path in info["changed"]:
        data = fetch(path)
        if hashlib.sha256(data).hexdigest() != remote["files"][path]["sha256"]:
            raise RuntimeError(f"내려받은 파일이 목록과 다릅니다: {path}")
        blobs[path] = data
        log(f"  받음 {path}")
    for path, data in blobs.items():
        dest = HERE / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".new")
        tmp.write_bytes(data)
        tmp.replace(dest)
    MANIFEST.write_bytes(remote_raw)
    log("업데이트 완료.")
    return info["changed"]


def main():
    try:
        apply()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            print(f"업데이트 확인 실패 (HTTP {e.code}) — 기존 파일로 실행합니다.")
        else:
            print(f"업데이트 확인 실패 (HTTP {e.code}) — 기존 파일로 실행합니다.")
    except Exception as e:
        print(f"업데이트 확인 실패 ({type(e).__name__}) — 기존 파일로 실행합니다.")
    # 새 패키지가 필요해졌을 수 있으니 확인
    req = HERE / "requirements.txt"
    if req.is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", str(req)],
                       check=False)
    subprocess.run([sys.executable, str(HERE / "app.py")])


if __name__ == "__main__":
    main()
