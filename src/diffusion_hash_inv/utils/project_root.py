"""
Find the project root directory by searching for specific marker files.
"""
# src/diffusion_hash_inv/utils/project_root.py
import sys
from pathlib import Path

def get_project_root(marker_files=("pyproject.toml", ".git")) -> Path:
    """
    Jupyter/Script 어디서 실행해도 프로젝트 루트를 찾아줌.
    marker_files 중 하나라도 있으면 거기를 루트로 간주.
    """
    current = Path.cwd().resolve()  # notebook에서는 cwd 기준
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")

def add_src_to_path():
    """프로젝트 루트 밑의 src/를 sys.path에 자동 추가"""
    root = get_project_root()
    src = root / "src"

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src

def add_root_to_path():
    """프로젝트 루트를 sys.path에 자동 추가"""
    root = get_project_root()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
