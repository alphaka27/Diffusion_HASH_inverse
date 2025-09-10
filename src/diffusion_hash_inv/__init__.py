"""
diffusion_hash_inverse: 해시/랜덤 비트 유틸 패키지
핵심 객체를 패키지 루트에서 바로 가져올 수 있도록 re-export 합니다.
"""

from importlib.metadata import version, PackageNotFoundError

# 서브패키지 주요 심볼 re-export
from .hashing import SHA256, ValidateHash
from .generator import GenerateRandom
from .utils import FileIO
from .utils import add_root_to_path, add_src_to_path

__all__ = [
    "SHA256",
    "ValidateHash",
    "GenerateRandom",
    "FileIO",
    "add_root_to_path",
    "add_src_to_path"
]

# 패키지 버전
try:
    __version__ = version("diffusion-hash-inv")
except PackageNotFoundError:
    # 개발환경(로컬)에서 pyproject 설치 전인 경우 대비
    __version__ = "0.0.0.dev"
