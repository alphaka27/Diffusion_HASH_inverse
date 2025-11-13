"""
Converting string to hex representation and vice versa.
"""

def to_hex32_scalar(x) -> str:
    """단일 32-bit 값 → 8자리 hex"""
    return "0x" + f"{int(x):08x}"

def to_hex32_concat(seq, endian='big') -> str:
    """시퀀스(8워드 등) → 64자리 hex"""
    if endian == 'little':
        seq = seq[::-1]
    return ''.join(f"{int(x):08x}" for x in seq)
