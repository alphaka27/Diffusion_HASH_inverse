"""
Encoding Validation Module
"""

from typing import Tuple
from diffusion_hash_inv.core import RGB
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.utils import Byte2RGB

def encoding_validate(
    byte_data: bytes | str,
    rgb_data: RGB | Tuple[RGB, ...],
    encoder: Byte2RGB,
):
    """
    Validate the encoding of byte data to RGB format.
    """

    decoded_byte = encoder.rgb_decoder(rgb_data)
    decoded_byte = Logs.bytes_to_str(decoded_byte) if isinstance(decoded_byte, bytes) \
        else decoded_byte
    byte_data = Logs.bytes_to_str(byte_data) if isinstance(byte_data, bytes) else byte_data
    if encoder.main_cfg.verbose_flag:
        print(f"Decoded Byte: {decoded_byte}\nOriginal Byte: {byte_data}\n")

    if decoded_byte == byte_data:
        if encoder.main_cfg.verbose_flag:
            print("Encoding validation successful: Decoded byte matches original byte.")
        return True

    if encoder.main_cfg.verbose_flag:
        print("Encoding validation failed: Decoded byte does not match original byte.")
    return False

if __name__ == "__main__":
    pass
