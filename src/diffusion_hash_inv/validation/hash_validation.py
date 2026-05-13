"""
Hash algorithm Validation Module
"""

# TODO
# - Change hexdigest to digest
# - Fix validation algorithm fit to hashlib.digest output

import hashlib
from diffusion_hash_inv.logger import Logs


def validate(
    test_hash: bytes,
    message: bytes,
    hash_alg: str = "sha256",
    verbose_flag: bool = True,
) -> tuple[bool, bytes]:
    """
    Validate a generated hash digest against hashlib.
    """
    if test_hash is None:
        raise ValueError("test_hash must be provided")
    if message is None:
        raise ValueError("message must be provided")

    if verbose_flag:
        print(f"Validating {hash_alg.upper()} hash...\nFor message: {message}\n")
    try:
        right_hash = getattr(hashlib, hash_alg.lower())()
    except AttributeError as exc:
        raise ValueError(f"Unsupported hash algorithm: {hash_alg}") from exc

    right_hash.update(message)
    _right_value = right_hash.digest()
    is_valid = test_hash == _right_value
    if verbose_flag:
        print(f"Generated Hash: {Logs.bytes_to_str(test_hash)}")
        print(f"Correct   Hash: {Logs.bytes_to_str(_right_value)}")
        print(f"Validation Result: {'Passed' if is_valid else 'Failed'}\n")

    return is_valid, _right_value
