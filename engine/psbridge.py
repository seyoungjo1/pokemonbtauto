# -*- coding: utf-8 -*-
"""tools/ps_bridge.js (쇼다운 시뮬레이터) 를 자식 프로세스로 띄워 대화한다.

쇼다운 폴더는 다음 순서로 찾는다: 인자 → 환경변수 PS_DIR → build/pokemon-showdown
(Node.js 가 필요하다. 학습은 GitHub Actions 나 Node 가 깔린 PC 에서 돌린다.)
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_ps_dir(ps_dir=None) -> Path | None:
    for p in (ps_dir, os.environ.get("PS_DIR"), ROOT / "build" / "pokemon-showdown"):
        if p and (Path(p) / "dist" / "sim").is_dir():
            return Path(p)
    return None


class Bridge:
    def __init__(self, ps_dir=None):
        ps = find_ps_dir(ps_dir)
        if ps is None:
            raise RuntimeError("쇼다운 폴더를 찾지 못했습니다 (PS_DIR 또는 build/pokemon-showdown).")
        node = shutil.which("node")
        if not node:
            raise RuntimeError("Node.js 가 없습니다. 자기대전 학습에는 Node.js 가 필요합니다.")
        self.proc = subprocess.Popen(
            [node, str(ROOT / "tools" / "ps_bridge.js"), str(ps)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1)

    def call(self, **msg) -> dict:
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()
            raise RuntimeError("쇼다운 다리가 종료됨: " + err[-2000:])
        out = json.loads(line)
        if out.get("error") and "id" not in out:
            raise RuntimeError(out["error"])
        return out

    def formats(self) -> list:
        return self.call(op="formats")["formats"]

    def pack(self, team_sets: list) -> str:
        return self.call(op="pack", team=team_sets)["packed"]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
