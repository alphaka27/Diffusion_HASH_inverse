import json
from pathlib import Path

from PIL import Image

from diffusion_hash_inv.models.sample_decoding import (
    _byte2rgb_decoder,
    decode_sample_image,
    write_decode_comparison,
)


def _write_encoded_png(path: Path, payload: bytes) -> None:
    decoder = _byte2rgb_decoder("golay24")
    encoded = decoder.rgb_encoder(payload)
    pixels = encoded if isinstance(encoded, tuple) else (encoded,)
    image = Image.new("RGB", (len(pixels), 1))
    image.putdata([pixel.as_tuple for pixel in pixels])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_legacy_encoded_png(path: Path, payload: bytes) -> None:
    decoder = _byte2rgb_decoder("legacy-bin")
    encoded = decoder.rgb_encoder(payload)
    pixels = encoded if isinstance(encoded, tuple) else (encoded,)
    image = Image.new("RGB", (len(pixels), 1))
    image.putdata([pixel.as_tuple for pixel in pixels])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def test_byte2rgb_uses_rgb_octants_as_bit_positions() -> None:
    decoder = _byte2rgb_decoder("golay24")

    encoded = decoder.rgb_encoder(b"\x12")
    assert isinstance(encoded, tuple)
    assert len(encoded) == 24
    assert [decoder.rgb_octant_decoder(pixel) for pixel in encoded[:8]] == [
        7,
        6,
        5,
        3,
        3,
        2,
        6,
        0,
    ]
    assert decoder.rgb_decoder(encoded) == b"\x12"


def test_byte2rgb_golay_corrects_three_bit_errors() -> None:
    decoder = _byte2rgb_decoder("golay24")
    encoded = list(decoder.rgb_encoder(b"\x12"))
    assert len(encoded) == 24

    for index in (0, 9, 23):
        bit_position = index % decoder.data_bits_per_byte
        octant = decoder.rgb_octant_decoder(encoded[index])
        wrong_octant = bit_position if octant != bit_position else bit_position ^ 0b111
        encoded[index] = decoder._rgb_from_octant(wrong_octant)

    assert decoder.rgb_decoder(tuple(encoded)) == b"\x12"


def test_decode_sample_image_records_rgb_colors(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _write_encoded_png(image_path, b"\x12\x34")

    decoded = decode_sample_image(image_path, _byte2rgb_decoder("golay24"))

    assert decoded["supported"] is True
    assert decoded["complete"] is True
    assert decoded["encoding"] == "rgb-bit-position-golay24"
    assert decoded["pixel_count"] == 48
    assert decoded["decoded_bit_count"] == 48
    assert decoded["decoded_byte_count"] == 2
    assert decoded["ecc"]["code"] == "extended-golay-24-12"
    assert decoded["ecc"]["corrected_codeword_count"] == 0
    assert decoded["ecc"]["uncorrectable_codeword_count"] == 0
    assert decoded["hex"] == "0x1234"
    assert decoded["unique_rgb_color_count"] == 8
    assert decoded["rgb_colors"][0]["rgb"] == [191, 191, 191]
    assert decoded["rgb_colors"][0]["bit_position"] == 0
    assert decoded["rgb_colors"][0]["decoded_rgb_space"] == 7
    assert decoded["rgb_colors"][0]["decoded_bit"] == 0
    assert decoded["rgb_colors"][0]["decoded_bits"] == "0"


def test_decode_sample_image_defaults_to_legacy_bin(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    _write_legacy_encoded_png(image_path, b"\x12\x34")

    decoded = decode_sample_image(image_path)

    assert decoded["supported"] is True
    assert decoded["complete"] is True
    assert decoded["encoding"] == "rgb-bin-legacy"
    assert decoded["pixel_count"] == 2
    assert decoded["decoded_byte_count"] == 2
    assert decoded["ecc"]["code"] == "none"
    assert decoded["hex"] == "0x1234"
    assert decoded["rgb_colors"][0]["decoded_byte"] == 0x12
    assert decoded["rgb_colors"][0]["decoded_byte_hex"] == "0x12"
    assert decoded["rgb_colors"][0]["decoded_bit"] is None


def test_write_decode_comparison_records_legacy_decoded_byte_hamming_distance(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    final_path = tmp_path / "final.png"
    output_path = tmp_path / "decode_comparison.json"
    _write_legacy_encoded_png(source_path, b"\x12\x34")
    _write_legacy_encoded_png(final_path, b"\x12\x35")

    write_decode_comparison([source_path], [final_path], output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    record = payload["records"][0]
    assert payload["decoded_byte_comparable"] == 1
    assert payload["decoded_byte_matches"] == 0
    assert payload["decoded_byte_all_match"] is False
    assert record["decoded_byte_comparable"] is True
    assert record["decoded_byte_match"] is False
    assert record["decoded_byte_count_delta"] == 0
    assert record["decoded_byte_hamming_distance_bits"] == 1
    assert record["decoded_byte_hamming_distance_bytes"] == 1
    assert record["hamming_distance_bits"] == 1
    assert record["source"]["hex"] == "0x1234"
    assert record["final"]["hex"] == "0x1235"
    assert record["source"]["encoding"] == "rgb-bin-legacy"
    assert record["final"]["encoding"] == "rgb-bin-legacy"
