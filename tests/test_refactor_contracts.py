import pytest

from diffusion_hash_inv import cli
from diffusion_hash_inv.config import HashConfig, MainConfig, MessageConfig, OutputConfig
from diffusion_hash_inv.generator import random_n_bits, random_n_char
from diffusion_hash_inv.generator.random_n_char import GenerateRandomNChar
from diffusion_hash_inv.hash_main import build_arg_parser, config_from_args
from diffusion_hash_inv.logger import BaseLogs, Logs, StepLogs
from diffusion_hash_inv.main import RuntimeConfig
from diffusion_hash_inv.utils import FileIO, bytes_to_binary_block, bytes_to_hex_block
from diffusion_hash_inv.validation import validate


def test_console_cli_lists_hash_and_mlx_commands(capsys) -> None:
    cli.main(["--help"])

    output = capsys.readouterr().out
    assert "hash" in output
    assert "mlx-toy" in output
    assert "mlx-conditional" in output


def test_console_cli_keeps_legacy_hash_options(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli.hash_main, "main", lambda argv: calls.append(argv))

    cli.main(["-i", "0"])
    cli.main(["hash", "-i", "1"])

    assert calls == [["-i", "0"], ["-i", "1"]]


def test_hash_main_config_from_args_uses_cli_values() -> None:
    args = build_arg_parser().parse_args(
        ["--hash-alg", "md5", "-e", "4", "--sequential", "--make-image", "-i", "2"]
    )

    config = config_from_args(args)

    assert args.iteration == 2
    assert config.hash.hash_alg_upper == "MD5"
    assert config.hash.length == 16
    assert config.message.length == 16
    assert config.message.random_flag is False
    assert config.main.make_image_flag is True


def test_runtime_config_update_returns_self_and_validates_lengths() -> None:
    config = RuntimeConfig.set_default()

    assert config.hash_update(HashConfig(hash_alg="md5", length=128)) is config

    with pytest.raises(ValueError, match="Configuration mismatch"):
        config.message_update(MessageConfig(length=256))


def test_logs_clear_accepts_runtime_state_aliases() -> None:
    baselogs = BaseLogs()
    steplogs = StepLogs()
    baselogs.message["Hex"] = "0x00"
    steplogs.logs["1st Step"] = "0x00"

    Logs.clear(baselogs=baselogs, steplogs=steplogs)

    assert baselogs.message == {}
    assert steplogs.logs == {}


def test_byte_formatters_are_shared_by_generators() -> None:
    assert bytes_to_hex_block(b"\x01\x02\x03", word_bytes=2) == "0102 0003"
    assert bytes_to_binary_block(b"\x01\xff") == "00000001 11111111"


def test_generator_cli_length_resolution_and_clear_flags() -> None:
    bits_parser = random_n_bits.build_arg_parser()
    chars_parser = random_n_char.build_arg_parser()

    assert random_n_bits.resolve_bit_length(bits_parser.parse_args([])) == 512
    assert random_n_bits.resolve_bit_length(bits_parser.parse_args(["-e", "5"])) == 32
    assert random_n_char.resolve_bit_length(chars_parser.parse_args(["-l", "64"])) == 64
    assert bits_parser.parse_args(["-c"]).clear is True
    assert bits_parser.parse_args(["-C"]).clear is False


def test_random_char_generator_works_with_main_config_without_seed_fields(tmp_path) -> None:
    main_config = MainConfig(
        verbose_flag=False,
        clean_flag=False,
        debug_flag=False,
        make_image_flag=False,
    )
    generator = GenerateRandomNChar(
        main_config,
        FileIO(main_config, OutputConfig(root_dir=tmp_path)),
    )

    generated = generator.generate(8)

    assert isinstance(generated, str)
    assert len(generated) == 8
    assert generator.normalize("A", form="none") == b"A"


def test_hash_validate_rejects_missing_inputs_and_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="test_hash"):
        validate(None, b"message", "md5", verbose_flag=False)

    with pytest.raises(ValueError, match="Unsupported hash algorithm"):
        validate(b"digest", b"message", "unknown", verbose_flag=False)
