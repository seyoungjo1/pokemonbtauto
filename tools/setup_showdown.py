#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""쇼다운 시뮬레이터 준비 — 최신 master 를 받아 build/pokemon-showdown 에 빌드한다.

  python tools/setup_showdown.py [--force]

이미 있고 7일이 안 지났으면 건너뛴다. Node.js(npm) 가 필요하다.
git 이 없어도 되도록 GitHub zip 으로 받는다.
"""
import io
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "build" / "pokemon-showdown"
ZIP = "https://codeload.github.com/smogon/pokemon-showdown/zip/refs/heads/master"
STAMP = DEST / ".pba_stamp.json"


def run(cmd, cwd):
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, shell=sys.platform.startswith("win"))
    if r.returncode != 0:
        raise SystemExit(f"실패: {' '.join(cmd)}")


def main():
    force = "--force" in sys.argv
    if not force and STAMP.is_file() and (DEST / "dist" / "sim").is_dir():
        age = time.time() - json.loads(STAMP.read_text(encoding="utf-8")).get("t", 0)
        if age < 7 * 86400:
            print(f"쇼다운 준비됨 ({age / 86400:.1f}일 전 빌드) — 건너뜀")
            return
    if not shutil.which("node") or not shutil.which("npm"):
        raise SystemExit("Node.js 가 없습니다. https://nodejs.org 에서 LTS 버전을 설치한 뒤 다시 실행하세요.")
    print("쇼다운 최신판 내려받는 중… (약 20MB)", flush=True)
    data = urllib.request.urlopen(urllib.request.Request(ZIP, headers={"User-Agent": "pokemonbtauto"}), timeout=300).read()
    tmp = ROOT / "build" / "ps_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    src = next(tmp.iterdir())
    shutil.rmtree(DEST, ignore_errors=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(DEST))
    shutil.rmtree(tmp, ignore_errors=True)
    print("빌드 중… (처음엔 몇 분 걸립니다)", flush=True)
    run(["npm", "install", "--no-audit", "--no-fund"], DEST)
    run(["node", "build"], DEST)
    STAMP.write_text(json.dumps({"t": time.time()}), encoding="utf-8")
    print("쇼다운 준비 완료:", DEST)


if __name__ == "__main__":
    main()
