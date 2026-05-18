"""Decode and compare saved diffusion sample images."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from diffusion_hash_inv.config import Byte2RGBConfig, HashConfig, MainConfig
from diffusion_hash_inv.core import RGB
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.utils.byte2rgb import Byte2RGB


def _byte2rgb_decoder() -> Byte2RGB:
    return Byte2RGB(
        main_config=MainConfig(
            verbose_flag=False,
            clean_flag=False,
            debug_flag=False,
            make_image_flag=False,
        ),
        hash_config=HashConfig(hash_alg="md5", length=8),
        rgb_config=Byte2RGBConfig(seed_flag=False, input_seed=42),
    )


def _bytes_to_bits(value: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in value)


def _hamming_distance_bits(left: bytes, right: bytes) -> int | None:
    if len(left) != len(right):
        return None
    return sum((left_byte ^ right_byte).bit_count() for left_byte, right_byte in zip(left, right))


def _hamming_distance_bytes(left: bytes, right: bytes) -> int | None:
    if len(left) != len(right):
        return None
    return sum(left_byte != right_byte for left_byte, right_byte in zip(left, right))


def _decode_rgb_pixel(rgb: RGB, decoder: Byte2RGB) -> int | None:
    return decoder.rgb_octant_decoder(rgb)


def _rgb_color_records(
    image: Image.Image,
    decoder: Byte2RGB | None,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    color_counts: Counter[tuple[int, int, int]] = Counter()
    records: list[dict[str, Any]] = []
    decoded_values: list[int] = []
    undecoded_records: list[dict[str, Any]] = []

    for y in range(image.height):
        for x in range(image.width):
            index = y * image.width + x
            rgb_tuple = tuple(int(value) for value in image.getpixel((x, y)))
            color_counts[rgb_tuple] += 1
            decoded_octant = (
                _decode_rgb_pixel(RGB.from_tuple(rgb_tuple), decoder)
                if decoder is not None
                else None
            )
            bit_span = (
                decoder.data_bits_per_byte
                if decoder is not None
                else Byte2RGB.data_bits_per_byte
            )
            bit_position = index % bit_span
            decoded_bit = (
                None if decoded_octant is None else int(decoded_octant == bit_position)
            )
            if decoded_octant is not None:
                decoded_values.append(decoded_octant)
            record = {
                "index": index,
                "x": x,
                "y": y,
                "rgb": list(rgb_tuple),
                "bit_position": bit_position,
                "decoded_bit": decoded_bit,
                "decoded_bits": None if decoded_bit is None else str(decoded_bit),
                "decoded_rgb_space": decoded_octant,
                "decoded_rgb_space_bits": (
                    None if decoded_octant is None else f"{decoded_octant:03b}"
                ),
                "decoded_bit_chunk": decoded_octant,
                "decoded_byte": None,
                "decoded_byte_hex": None,
            }
            records.append(record)
            if decoded_octant is None:
                undecoded_records.append(record)

    unique_colors = [
        {"rgb": list(rgb), "count": count}
        for rgb, count in sorted(color_counts.items())
    ]
    return records, decoded_values, unique_colors, undecoded_records


def decode_sample_image(path: Path, decoder: Byte2RGB | None = None) -> dict[str, Any]:
    decoder = decoder or _byte2rgb_decoder()
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        width, height = int(rgb_image.width), int(rgb_image.height)
        supported = image.mode in ("RGB", "RGBA")
        pixel_records, decoded_values, unique_colors, undecoded_records = _rgb_color_records(
            rgb_image,
            decoder if supported else None,
        )
        if image.mode not in ("RGB", "RGBA"):
            return {
                "supported": False,
                "reason": f"Byte2RGB decode requires RGB/RGBA image mode, got {image.mode}",
                "mode": image.mode,
                "size": [int(image.width), int(image.height)],
                "pixel_count": int(image.width * image.height),
                "decoded_byte_count": 0,
                "invalid_pixel_count": int(image.width * image.height),
                "hex": None,
                "bits": None,
                "rgb_colors": pixel_records,
                "unique_rgb_colors": unique_colors,
                "unique_rgb_color_count": len(unique_colors),
                "undecoded_rgb_colors": undecoded_records,
            }

    decoded_byte_values, ecc_records = decoder.decode_octants(decoded_values)
    decoded = Logs.iter_to_bytes(decoded_byte_values, byteorder=decoder.hash_cfg.byteorder)
    decoded_byte_count = len(decoded)
    decoded_rgb_space_count = len(decoded_values)
    trailing_bit_count = decoded_rgb_space_count % decoder.code_bits_per_byte
    invalid_pixel_count = len(pixel_records) - decoded_rgb_space_count
    uncorrectable_codeword_count = sum(
        1 for record in ecc_records if record.get("uncorrectable")
    )
    corrected_codeword_count = sum(
        1 for record in ecc_records if record.get("corrected")
    )
    return {
        "supported": True,
        "complete": (
            invalid_pixel_count == 0
            and trailing_bit_count == 0
            and uncorrectable_codeword_count == 0
        ),
        "encoding": "rgb-bit-position-golay24",
        "mode": "RGB",
        "size": [width, height],
        "pixel_count": len(pixel_records),
        "decoded_bit_count": decoded_rgb_space_count,
        "decoded_bit_chunk_count": decoded_rgb_space_count,
        "decoded_rgb_space_count": decoded_rgb_space_count,
        "trailing_bit_count": trailing_bit_count,
        "trailing_bit_chunk_count": trailing_bit_count,
        "decoded_byte_count": decoded_byte_count,
        "ecc": {
            "code": "extended-golay-24-12",
            "data_bits_per_byte": decoder.data_bits_per_byte,
            "code_bits_per_byte": decoder.code_bits_per_byte,
            "max_correctable_bits_per_byte": decoder.golay_max_correctable_bits,
            "codeword_count": len(ecc_records),
            "corrected_codeword_count": corrected_codeword_count,
            "uncorrectable_codeword_count": uncorrectable_codeword_count,
            "records": ecc_records,
        },
        "invalid_pixel_count": invalid_pixel_count,
        "hex": Logs.bytes_to_str(decoded),
        "bits": _bytes_to_bits(decoded),
        "rgb_colors": pixel_records,
        "unique_rgb_colors": unique_colors,
        "unique_rgb_color_count": len(unique_colors),
        "undecoded_rgb_colors": undecoded_records,
    }


def write_decode_comparison_summary(
    job_results: dict[str, Any],
    output_path: Path,
) -> Path:
    """Aggregate per-sample source/final decode info and Hamming distances across all jobs.

    ``job_results`` is the ``results`` dict produced by the notebook runner —
    each value must contain a ``"sample_decode_comparison"`` key whose value is
    the path to that job's ``decode_comparison.json`` file.
    """
    overview: dict[str, list[str]] = {}
    detail: dict[str, Any] = {}

    for job_key, result in job_results.items():
        decode_comparison_path = result.get("sample_decode_comparison")
        if not decode_comparison_path:
            continue
        path = Path(decode_comparison_path)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records = []
        job_overview: list[str] = []
        for r in data.get("records", []):
            src = r.get("source") or {}
            fin = r.get("final") or {}
            final_bytes = fin.get("decoded_byte_count")
            full_bytes = src.get("decoded_byte_count")
            job_overview.append(
                f"{final_bytes}/{full_bytes}"
                if final_bytes is not None and full_bytes is not None
                else "?/?"
            )
            records.append(
                {
                    "index": r["index"],
                    "source": {
                        "hex": src.get("hex"),
                        "bits": src.get("bits"),
                        "decoded_byte_count": full_bytes,
                        "invalid_pixel_count": src.get("invalid_pixel_count"),
                    },
                    "final": {
                        "hex": fin.get("hex"),
                        "bits": fin.get("bits"),
                        "decoded_byte_count": final_bytes,
                        "invalid_pixel_count": fin.get("invalid_pixel_count"),
                    },
                    "comparable": r.get("comparable"),
                    "same_length": r.get("same_length"),
                    "match": r.get("match"),
                    "decoded_byte_match": r.get("decoded_byte_match"),
                    "hamming_distance_bits": r.get("hamming_distance_bits"),
                    "decoded_byte_hamming_distance_bits": r.get(
                        "decoded_byte_hamming_distance_bits"
                    ),
                    "decoded_byte_hamming_distance_bytes": r.get(
                        "decoded_byte_hamming_distance_bytes"
                    ),
                }
            )
        overview[job_key] = job_overview
        detail[job_key] = {
            "total": data.get("total"),
            "comparable": data.get("comparable"),
            "matches": data.get("matches"),
            "all_match": data.get("all_match"),
            "decoded_byte_comparable": data.get("decoded_byte_comparable"),
            "decoded_byte_matches": data.get("decoded_byte_matches"),
            "decoded_byte_all_match": data.get("decoded_byte_all_match"),
            "samples": records,
        }

    overview_block: dict[str, Any] = {
        "_description": {
            "format": "final_decoded_bytes / source_decoded_bytes",
            "final_decoded_bytes": (
                "Number of bytes successfully decoded from the diffusion model's generated image."
            ),
            "source_decoded_bytes": (
                "Number of bytes decoded from the original source image "
                "(the expected full length used as ground truth)."
            ),
            "interpretation": (
                "X/Y means the model recovered X out of Y bytes. "
                "X < Y indicates invalid/missing pixels caused by diffusion noise; "
                "X == Y is a prerequisite for Hamming distance computation."
            ),
        },
        **overview,
    }
    payload = {"overview": overview_block, **detail}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_decode_comparison(
    source_files: Sequence[Path],
    final_files: Sequence[Path],
    output_path: Path,
) -> Path:
    if len(source_files) != len(final_files):
        raise ValueError(
            "source_files and final_files must have the same length, "
            f"got {len(source_files)} and {len(final_files)}"
        )

    decoder = _byte2rgb_decoder()
    records: list[dict[str, Any]] = []
    for index, (source_path, final_path) in enumerate(zip(source_files, final_files)):
        source_decoded = decode_sample_image(source_path, decoder)
        final_decoded = decode_sample_image(final_path, decoder)
        comparable = (
            bool(source_decoded.get("supported"))
            and bool(final_decoded.get("supported"))
            and bool(source_decoded.get("complete"))
            and bool(final_decoded.get("complete"))
        )
        source_bytes = Logs.str_to_bytes(source_decoded["hex"]) if source_decoded["hex"] else b""
        final_bytes = Logs.str_to_bytes(final_decoded["hex"]) if final_decoded["hex"] else b""
        same_length = len(source_bytes) == len(final_bytes)
        decoded_byte_comparable = (
            bool(source_decoded.get("supported"))
            and bool(final_decoded.get("supported"))
            and same_length
        )
        decoded_byte_match = (
            decoded_byte_comparable
            and source_bytes == final_bytes
        )
        decoded_byte_hamming_distance_bits = (
            _hamming_distance_bits(source_bytes, final_bytes)
            if decoded_byte_comparable
            else None
        )
        decoded_byte_hamming_distance_bytes = (
            _hamming_distance_bytes(source_bytes, final_bytes)
            if decoded_byte_comparable
            else None
        )
        match = comparable and same_length and source_bytes == final_bytes
        records.append(
            {
                "index": index,
                "source_file": str(source_path),
                "final_file": str(final_path),
                "comparable": comparable,
                "same_length": same_length,
                "decoded_byte_comparable": decoded_byte_comparable,
                "decoded_byte_count_delta": len(source_bytes) - len(final_bytes),
                "decoded_byte_match": decoded_byte_match,
                "match": match if comparable and same_length else None,
                "decoded_byte_hamming_distance_bits": decoded_byte_hamming_distance_bits,
                "decoded_byte_hamming_distance_bytes": decoded_byte_hamming_distance_bytes,
                "hamming_distance_bits": decoded_byte_hamming_distance_bits,
                "source": source_decoded,
                "final": final_decoded,
            }
        )

    comparable_records = [
        record for record in records if record["comparable"] and record["same_length"]
    ]
    matched_records = [record for record in comparable_records if record["match"]]
    decoded_byte_comparable_records = [
        record for record in records if record["decoded_byte_comparable"]
    ]
    decoded_byte_matched_records = [
        record for record in decoded_byte_comparable_records if record["decoded_byte_match"]
    ]

    overview_entries: list[str] = []
    for r in records:
        final_bytes = (r.get("final") or {}).get("decoded_byte_count")
        full_bytes = (r.get("source") or {}).get("decoded_byte_count")
        overview_entries.append(
            f"{final_bytes}/{full_bytes}"
            if final_bytes is not None and full_bytes is not None
            else "?/?"
        )

    payload = {
        "overview": {
            "_description": {
                "format": "final_decoded_bytes / source_decoded_bytes",
                "final_decoded_bytes": (
                    "Number of bytes successfully decoded from the diffusion model's"
                    " generated image."
                ),
                "source_decoded_bytes": (
                    "Number of bytes decoded from the original source image"
                    " (the expected full length used as ground truth)."
                ),
                "interpretation": (
                    "X/Y means the model recovered X out of Y bytes. "
                    "X < Y indicates invalid/missing pixels caused by diffusion noise; "
                    "X == Y is a prerequisite for Hamming distance computation."
                ),
            },
            "samples": overview_entries,
        },
        "total": len(records),
        "comparable": len(comparable_records),
        "matches": len(matched_records),
        "all_match": len(records) > 0 and len(matched_records) == len(records),
        "decoded_byte_comparable": len(decoded_byte_comparable_records),
        "decoded_byte_matches": len(decoded_byte_matched_records),
        "decoded_byte_all_match": (
            len(records) > 0 and len(decoded_byte_matched_records) == len(records)
        ),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
