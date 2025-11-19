"""
diffusion_hash_inverse: 해시/랜덤 비트 유틸 패키지
핵심 객체를 패키지 루트에서 바로 가져올 수 있도록 re-export 합니다.
"""

from importlib.metadata import version, PackageNotFoundError

from diffusion_hash_inv.common import BaseCalc
from diffusion_hash_inv.common import Logs, Metadata, BaseLogs, StepLogs

from diffusion_hash_inv.utils import FileIO
from diffusion_hash_inv.utils import JSONFormat
from diffusion_hash_inv.utils import XLSXFormat

from diffusion_hash_inv.hashing import MD5
from diffusion_hash_inv.hashing import SHA256

__all__ = [
    "BaseCalc",
    "Logs",
    "Metadata",
    "BaseLogs",
    "StepLogs",
    "FileIO",
    "JSONFormat",
    "XLSXFormat",
    "MD5",
    "SHA256"
]

# dev 패키지 버전
try:
    __version__ = version("diffusion-hash-inv")
except PackageNotFoundError:
    # 개발환경(로컬)에서 pyproject 설치 전인 경우 대비
    __version__ = "0.0.0.dev"
