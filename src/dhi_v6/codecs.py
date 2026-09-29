"""NumPy 이미지 codec. 슬롯 배치와 글꼴은 v3.1 encoding/{bgv,cgge}.py에서 옮겼다."""
import hashlib

import numpy as np

from . import data

_GLYPH_BYTES = bytes.fromhex(
    """
    183c3c1818001800 3636000000000000 36367f367f363600 0c3e031e301f0c00
    006333180c666300 1c361c6e3b336e00 0606030000000000 180c0606060c1800
    060c1818180c0600 00663cff3c660000 000c0c3f0c0c0000 00000000000c0c06
    0000003f00000000 00000000000c0c00 6030180c06030100 3e63737b6f673e00
    0c0e0c0c0c0c3f00 1e33301c06333f00 1e33301c30331e00 383c36337f307800
    3f031f3030331e00 1c06031f33331e00 3f3330180c0c0c00 1e33331e33331e00
    1e33333e30180e00 000c0c00000c0c00 000c0c00000c0c06 180c0603060c1800
    00003f00003f0000 060c1830180c0600 1e3330180c000c00 3e637b7b7b031e00
    0c1e33333f333300 3f66663e66663f00 3c66030303663c00 1f36666666361f00
    7f46161e16467f00 7f46161e16060f00 3c66030373667c00 3333333f33333300
    1e0c0c0c0c0c1e00 7830303033331e00 6766361e36666700 0f06060646667f00
    63777f7f6b636300 63676f7b73636300 1c36636363361c00 3f66663e06060f00
    1e3333333b1e3800 3f66663e36666700 1e33070e38331e00 3f2d0c0c0c0c1e00
    3333333333333f00 33333333331e0c00 6363636b7f776300 63631e1c1c366300
    3333331e0c0c1e00 7f6331184c667f00 1e06060606061e00 03060c1830604000
    1e18181818181e00 081c366300000000 00000000000000ff 0c0c180000000000
    00001e303e336e00 0706063e66663b00 00001e3303331e00 3830303e33336e00
    00001e333f031e00 1c36060f06060f00 00006e33333e301f 0706366e66666700
    0c000e0c0c0c1e00 300030303033331e 070666361e366700 0e0c0c0c0c0c1e00
    0000337f7f6b6300 00001f3333333300 00001e3333331e00 00003b66663e060f
    00006e33333e3078 00003b6e66060f00 00003e031e301f00 080c3e0c0c2c1800
    0000333333336e00 00003333331e0c00 0000636b7f7f3600 000063361c366300
    00003333333e301f 00003f190c263f00 380c0c070c0c3800 1818180018181800
    070c0c380c0c0700 6e3b000000000000
    """
)

GLYPHS = ((np.frombuffer(_GLYPH_BYTES, dtype=np.uint8).reshape(94, 8, 1)
           >> np.arange(8)) & 1).astype(np.float64)
BYTE_BITS = ((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float64)
SHAPES = {"bgv": (2, 32, 128), "cgge": (2, 32, 64)}


def glyph_table_checksum():
    return hashlib.sha256(_GLYPH_BYTES).hexdigest()


def _lengths(lengths):
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
            or np.any((lengths < 4) | (lengths > 31))):
        raise ValueError("Lengths must be integers in 4..31")
    return lengths


def _image(slots):
    n, _, height, width = slots.shape
    return slots.reshape(n, 4, 8, height, width).transpose(0, 1, 3, 2, 4).reshape(n, 32, 8 * width)


def _slots(channel, width):
    return channel.reshape(len(channel), 4, 8, 8, width).transpose(0, 1, 3, 2, 4).reshape(len(channel), 32, 8, width)


def structure(lengths, representation):
    lengths = _lengths(lengths)
    shape = SHAPES[representation]
    cell_width = shape[-1] // 8
    first = int(representation == "bgv")
    slot = np.arange(32)[None, :]
    active = slot < lengths[:, None] + first
    payload_slots = active & (slot >= first)
    mask = _image(np.broadcast_to(active[:, :, None, None], (len(lengths), 32, 8, cell_width)))
    payload = np.zeros((len(lengths), *shape), dtype=bool)
    payload[:, 0] = _image(np.broadcast_to(payload_slots[:, :, None, None], (len(lengths), 32, 8, cell_width)))
    fixed = np.full((len(lengths), *shape), -1, dtype=np.float32)
    fixed[:, 1] = 2 * mask.astype(np.float32) - 1
    if first:
        header = BYTE_BITS[lengths].reshape(-1, 2, 4).repeat(4, axis=1).repeat(4, axis=2)
        fixed[:, 0, :8, :16] = 2 * header - 1
    return fixed, payload


def encode(payload, lengths, representation, src):
    if representation == "cgge" and src != "P":
        raise ValueError("CGGE requires Printable source")
    tokens = data.encode(payload, lengths, src)
    lengths = _lengths(lengths)
    active = np.arange(31) < lengths[:, None]
    payload = np.where(active, tokens[:, :31] + data.SOURCES[src]["byte_min"], 0)
    fixed, mask = structure(lengths, representation)
    if representation == "bgv":
        values = np.concatenate((lengths[:, None], payload), axis=1)
        slots = BYTE_BITS[values].reshape(-1, 32, 2, 4).repeat(4, axis=2).repeat(4, axis=3)
    else:
        slots = np.zeros((len(lengths), 32, 8, 8), dtype=np.float32)
        slots[:, :31] = GLYPHS[np.clip(payload - 33, 0, 93)]
    fixed[:, 0] = np.where(mask[:, 0], 2 * _image(slots) - 1, fixed[:, 0])
    return fixed


def _features(images, lengths, representation, src):
    lengths = _lengths(lengths)
    images = np.asarray(images)
    if representation == "cgge" and src != "P":
        raise ValueError("CGGE requires Printable source")
    if images.shape != (len(lengths), *SHAPES[representation]):
        raise ValueError("Invalid image shape")
    if not np.isfinite(images).all():
        raise FloatingPointError("Non-finite image")
    if np.any((images < -1) | (images > 1)):
        raise ValueError("Image must be clipped to [-1, 1]")
    slots = _slots(images[:, 0].astype(np.float64), SHAPES[representation][-1] // 8)
    if representation == "bgv":
        features = (slots[:, 1:].reshape(-1, 31, 2, 4, 4, 4).mean(axis=(3, 5)).reshape(-1, 31, 8) + 1) / 2
        spec = data.SOURCES[src]
        prototypes = BYTE_BITS[spec["byte_min"]:spec["byte_max"] + 1]
    else:
        features = (slots[:, :31].reshape(-1, 31, 64) + 1) / 2
        prototypes = GLYPHS.reshape(94, 64)
    return features, prototypes, lengths


def decode(images, lengths, representation, src):
    """Return prototype messages, maximum distances and diagnostic strict flags."""
    features, prototypes, lengths = _features(images, lengths, representation, src)
    payload = np.zeros((len(lengths), 31), dtype=np.uint8)
    distances = np.zeros((len(lengths), 31), dtype=np.float64)
    # Bound temporary distance matrices independently of generation batch size.
    for offset in range(0, len(lengths), 256):
        u = features[offset:offset + 256]
        d = (u * u).sum(axis=-1, keepdims=True) + prototypes.sum(axis=1) - 2 * (u @ prototypes.T)
        if representation == "cgge":
            d /= 64
        nearest = d.argmin(axis=-1)
        payload[offset:offset + 256] = nearest + data.SOURCES[src]["byte_min"]
        distances[offset:offset + 256] = np.take_along_axis(d, nearest[..., None], axis=-1)[..., 0]
    active = np.arange(31) < lengths[:, None]
    margins = np.where(active, distances, 0).max(axis=1)
    if representation == "bgv":
        strict_bytes = ((features >= .5) * (1 << np.arange(7, -1, -1))).sum(axis=-1)
        spec = data.SOURCES[src]
        strict = np.all(~active | ((strict_bytes >= spec["byte_min"]) & (strict_bytes <= spec["byte_max"])), axis=1)
    else:
        strict = margins <= .1
    payload[~active] = 0
    return data.messages(payload, lengths), margins, strict


def strict_decode(images, lengths, representation, src):
    decoded, _, flags = decode(images, lengths, representation, src)
    if representation == "bgv":
        features, _, lengths = _features(images, lengths, representation, src)
        payload = ((features >= .5) * (1 << np.arange(7, -1, -1))).sum(axis=-1).astype(np.uint8)
        decoded = data.messages(payload, lengths)
    return [message if ok else None for message, ok in zip(decoded, flags, strict=True)]
