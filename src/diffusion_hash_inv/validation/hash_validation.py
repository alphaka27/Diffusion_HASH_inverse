"""
Hash algorithm Validation Module
"""

import hashlib
import numpy as np

try:
    import diffusion_hash_inv.hashing as TestHash
except ImportError as e:
    print(f"Error importing TestHash: {e}")


def validate(test_hash = None, message = None, hash_alg = "sha256", verbose_flag = True):
    """
    Validate the SHA-256 hash of the given message.
    """
    right_hash = getattr(hashlib, hash_alg.lower())()
    right_hash.update(message)
    _right_value = right_hash.hexdigest()
    print(message)
    if verbose_flag:
        for _i in range(0, len(_right_value), 8):
            print(f"Chunk {_i // 8}: {_right_value[_i:_i + 8]}")
    b = bytes.fromhex(_right_value)
    out = np.frombuffer(b, dtype='>u4').astype(np.uint32, copy=True)
    if verbose_flag:
        print()
        print("In Byte representation")
        print(f"Correct HASH: \n{out}")

        print(f"Generated hash: \n{test_hash}")

    for _i, _test in enumerate(test_hash):
        if _test == out[_i]:
            pass
        else:
            print("Hash validation failed.")
            return False, test_hash, _right_value

    print("Hash validation successful.")
    return True, test_hash, None
