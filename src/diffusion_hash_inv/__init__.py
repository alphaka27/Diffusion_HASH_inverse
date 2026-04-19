"""
diffusion_hash_inverse: 해시/랜덤 비트 유틸 패키지
핵심 객체를 패키지 루트에서 바로 가져올 수 있도록 re-export 합니다.
"""

from importlib.metadata import version, PackageNotFoundError

from diffusion_hash_inv.core import BaseCalc
from diffusion_hash_inv.logger import Logs

from diffusion_hash_inv.main import RuntimeState, RuntimeConfig
from diffusion_hash_inv.main import MainEP

__all__ = [
    "BaseCalc",
    "Logs",
    "RuntimeState",
    "RuntimeConfig",
    "MainEP",
]

# dev 패키지 버전
try:
    __version__ = version("diffusion-hash-inv")

except PackageNotFoundError:
    # 개발환경(로컬)에서 pyproject 설치 전인 경우 대비
    __version__ = "0.0.0.dev"
