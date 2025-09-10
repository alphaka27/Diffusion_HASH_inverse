"""Python Package Customize"""
# sitecustomize.py
import sys
import pathlib

# 프로젝트 루트 경로
ROOT = pathlib.Path(__file__).resolve().parent

# src/를 sys.path에 추가
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
