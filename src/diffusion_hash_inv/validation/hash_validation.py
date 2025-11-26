"""
Hash algorithm Validation Module
"""

# TODO
# - Change hexdigest to digest
# - Fix validation algorithm fit to hashlib.digest output

import hashlib
from diffusion_hash_inv.common import Logs


def validate(test_hash = None, message = None, hash_alg = "sha256", verbose_flag = True):
    """
    Validate the SHA-256 hash of the given message.
    """
    if verbose_flag:
        print(f"Validating {hash_alg.upper()} hash...\nFor message: {message}\n")
    right_hash = getattr(hashlib, hash_alg)()
    right_hash.update(message)
    _right_value = right_hash.digest()
    is_valid = test_hash == _right_value
    if verbose_flag:
        print(f"Generated Hash: {Logs.bytes_to_str(test_hash)}")
        print(f"Correct   Hash: {Logs.bytes_to_str(_right_value)}")
        print(f"Validation Result: {'Passed' if is_valid else 'Failed'}\n")

    return is_valid, _right_value
