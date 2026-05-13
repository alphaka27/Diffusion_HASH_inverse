"""
난수/패스워드 생성 관련 모듈
- GenerateRandom: 비트 길이 기반 난수 생성
- nist_pwgen_utf8, random_n_char 모듈은 네임스페이스로 공개
"""

__all__ = [
    "NBitsGenerator",
    "GenerateRandomNChar",
]


def __getattr__(name):
    if name == "GenerateRandomNChar":
        from .random_n_char import GenerateRandomNChar
        return GenerateRandomNChar
    if name == "NBitsGenerator":
        from .n_bits_gen import NBitsGenerator
        return NBitsGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
