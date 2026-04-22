from pathlib import Path

import pytest

from diffusion_hash_inv.config import HashConfig, MainConfig, OutputConfig
from diffusion_hash_inv.core import BaseCalc


def test_hash_config_exposes_md5_runtime_properties() -> None:
    config = HashConfig(hash_alg="md5", length=128)

    assert config.byteorder == "little"
    assert config.ws_bits == 32
    assert config.bs_bits == 512
    assert config.ws_bytes == 4
    assert config.bs_bytes == 64
    assert config.mask == 0xFFFFFFFF
    assert config.hierarchy == ("Step", "Round", "Loop")


def test_hash_config_rejects_invalid_algorithm_and_length() -> None:
    with pytest.raises(ValueError, match="Unsupported hash algorithm"):
        HashConfig(hash_alg="sha1", length=128)

    with pytest.raises(ValueError, match="positive multiple of 8"):
        HashConfig(hash_alg="md5", length=127)


def test_main_and_output_config_behave_as_expected() -> None:
    seeded = MainConfig(
        message_flag=True,
        verbose_flag=False,
        clean_flag=True,
        debug_flag=False,
        make_xlsx_flag=False,
        seed_flag=True,
    )
    explicit_root = Path("/tmp/diffusion-hash-test-root")
    output = OutputConfig(root_dir=explicit_root)
    project_root = OutputConfig.get_project_root()

    assert seeded.seed != 0
    seeded.reset_clean_flag()
    assert seeded.clean_flag is False
    assert output.data_dir == explicit_root / "data"
    assert output.output_dir == explicit_root / "output"
    assert (project_root / "pyproject.toml").exists()

    with pytest.raises(ValueError, match="MainConfig has no attribute"):
        getattr(seeded, "missing_field")


def test_base_calc_operations_track_overflow_and_mutation() -> None:
    calc = BaseCalc(HashConfig(hash_alg="md5", length=128))
    calc.example = 7

    assert calc.word_to_int(b"\x01\x00\x00\x00") == 1
    assert calc.block_to_int(bytes(64)) == 0
    assert calc.get_variable("example") == 7

    calc.set_variable("example", b"\x08\x00\x00\x00")
    assert calc.get_variable("example") == 8

    assert calc.modular_add(calc.mask, 1) == 0
    assert calc.total_overflow_count == 1
    assert calc.loop_overflow_count == 1
    assert calc.rotl(0x12345678, 8) == 0x34567812
    assert calc.rotr(0x12345678, 8) == 0x78123456
    assert calc.shl(0x11, 4) == 0x110
    assert calc.shr(0x110, 4) == 0x11
    assert calc.modular_not(0) == calc.mask
    assert calc.modular_and(0x0F0F, 0x00FF) == 0x000F
    assert calc.modular_or(0x0F00, 0x00F0) == 0x0FF0
    assert calc.modular_xor(0xAAAA, 0x00FF) == 0xAA55

    calc.clear_overflow()
    assert calc.total_overflow_count == 0
    assert calc.loop_overflow_count == 0
