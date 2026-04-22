from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from diffusion_hash_inv.config.hash_config import HashConfig, MD5Constants, SHA256Constants
from diffusion_hash_inv.config.main_config import HeaderConstants, MainConfig, OutputConfig
from diffusion_hash_inv.core.base_calc import BaseCalc
from diffusion_hash_inv.hashing.md5 import MD5, MD5Calc, MD5Logic
from diffusion_hash_inv.logger.logger import (
    BaseLogs,
    LogHelper,
    Logs,
    MD5Logger,
    MD5RoundTrace,
    MD5Step4Trace,
    Metadata,
    StepLogs,
    TimeHelper,
)


def test_hash_config_and_constants_cover_error_branches() -> None:
    with pytest.raises(ValueError, match="hash_alg must be specified"):
        HashConfig(hash_alg=None, length=128)
    with pytest.raises(ValueError, match="length must be specified"):
        HashConfig(hash_alg="md5", length=None)
    with pytest.raises(ValueError, match="word_size must be a positive multiple of 8"):
        MD5Constants(word_size=7)
    with pytest.raises(ValueError, match="block_size must be a positive multiple of 8"):
        MD5Constants(block_size=510)
    with pytest.raises(ValueError, match="word_size must be a positive multiple of 8"):
        SHA256Constants(word_size=0)
    with pytest.raises(ValueError, match="block_size must be a positive multiple of 8"):
        SHA256Constants(block_size=513)

    sha_cfg = HashConfig(hash_alg="sha256", length=256)
    assert sha_cfg.byteorder == "big"
    assert sha_cfg.hierarchy == ("Round", "Step", "Loop")
    assert set(HashConfig._get_classes()) == {MD5Constants, SHA256Constants}
    assert sha_cfg.byteorder == sha_cfg.__getattribute__("byteorder")

    md5_cfg = HashConfig(hash_alg="md5", length=128)
    assert len(md5_cfg.s) == 64

    stub = SimpleNamespace(
        byteorder=None,
        word_size=32,
        block_size=512,
        mask=0xFFFFFFFF,
        hierarchy=("Step", "Round", "Loop"),
    )
    object.__setattr__(md5_cfg, "constants", stub)
    with pytest.raises(ValueError, match="byteorder is not set"):
        _ = md5_cfg.byteorder

    stub.byteorder = "middle"
    with pytest.raises(ValueError, match="byteorder must be either"):
        _ = md5_cfg.byteorder

    stub.byteorder = "little"
    stub.word_size = None
    with pytest.raises(ValueError, match="word_size is not set"):
        _ = md5_cfg.ws_bits

    stub.word_size = -8
    with pytest.raises(ValueError, match="word_size must be a positive multiple of 8"):
        _ = md5_cfg.ws_bytes

    stub.word_size = 32
    stub.block_size = None
    with pytest.raises(ValueError, match="block_size is not set"):
        _ = md5_cfg.bs_bits

    stub.block_size = -8
    with pytest.raises(ValueError, match="block_size must be a positive multiple of 8"):
        _ = md5_cfg.bs_bytes

    stub.block_size = 512
    stub.mask = None
    with pytest.raises(ValueError, match="mask is not set"):
        _ = md5_cfg.mask

    stub.mask = 0xFFFFFFFF
    stub.hierarchy = None
    with pytest.raises(ValueError, match="hierarchy is not set"):
        _ = md5_cfg.hierarchy

    with pytest.raises(AttributeError, match="no attribute 'unknown'"):
        getattr(md5_cfg, "unknown")


def test_main_output_config_and_header_constants_cover_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    main_cfg = MainConfig(
        message_flag=True,
        verbose_flag=False,
        clean_flag=True,
        debug_flag=False,
        make_xlsx_flag=False,
        seed_flag=False,
    )
    object.__setattr__(main_cfg, "message_flag", None)
    with pytest.raises(ValueError, match="not initialized"):
        _ = main_cfg.message_flag

    output_cfg = OutputConfig(root_dir=Path("/tmp/diffhash-root"))
    object.__setattr__(output_cfg, "data_dir", None)
    with pytest.raises(ValueError, match="not initialized"):
        _ = output_cfg.data_dir
    with pytest.raises(ValueError, match="has no attribute"):
        getattr(output_cfg, "missing")

    fake_cwd = Path("/tmp/diffhash-no-root")
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: fake_cwd))
    with pytest.raises(FileNotFoundError, match="프로젝트 루트를 찾을 수 없습니다"):
        OutputConfig.get_project_root(marker_files=("missing.toml", ".missing-git"))

    header = HeaderConstants()
    assert header.header_length == 48


def test_base_calc_bytes_paths_and_assertions() -> None:
    calc = BaseCalc(HashConfig(hash_alg="md5", length=128))
    calc.example = 1

    assert calc.modular_add(b"\xff\xff\xff\xff", 1) == 0
    assert calc.rotl(b"\x78\x56\x34\x12", 8) == 0x34567812
    assert calc.rotr(b"\x78\x56\x34\x12", 8) == 0x78123456
    assert calc.shr(b"\x10\x01\x00\x00", 4) == 0x11
    assert calc.shl(b"\x11\x00\x00\x00", 4) == 0x110
    assert calc.modular_not(b"\x00\x00\x00\x00") == calc.mask
    assert calc.modular_and(b"\x0f\x00\x00\x00", b"\xff\x00\x00\x00") == 0x000F
    assert calc.modular_or(b"\x00\x0f\x00\x00", b"\xf0\x00\x00\x00") == 0x0FF0
    assert calc.modular_xor(b"\xaa\xaa\x00\x00", b"\xff\x00\x00\x00") == 0xAA55

    calc.example = [1]
    with pytest.raises(AssertionError, match="not an integer or string"):
        calc.get_variable("example")
    with pytest.raises(AssertionError, match="not a valid attribute"):
        calc.set_variable("missing", 1)
    with pytest.raises(AssertionError, match="Input bytes length must be equal"):
        calc.word_to_int(b"\x00")
    with pytest.raises(AssertionError, match="Input bytes length must be equal"):
        calc.block_to_int(b"\x00")


def test_logger_helpers_and_step_logs_cover_fallback_paths() -> None:
    base_logs = BaseLogs()
    base_logs.set_message(b"abc", message_mode=False)
    assert base_logs.message == {"Hex": "0x616263"}
    assert BaseLogs.keys() == ["Message", "Generated hash", "Correct   hash"]

    with pytest.raises(AssertionError, match="message must be provided"):
        base_logs.update(generated_hash=b"\x00", correct_hash=b"\x00")
    with pytest.raises(AssertionError, match="is_message must be a boolean"):
        base_logs.update(
            message=b"x",
            is_message="yes",
            generated_hash=b"\x00",
            correct_hash=b"\x00",
        )

    step_logs = StepLogs()
    assert step_logs._int_to_hex(15) == "0x0f"
    with pytest.raises(ValueError, match="negative integers are not supported"):
        step_logs._int_to_hex(-1)
    with pytest.raises(AssertionError, match="index must be positive"):
        StepLogs.index_label(0, "Loop")
    with pytest.raises(ValueError, match="path must not be empty"):
        step_logs.set_value((), "x")

    step_logs.set_value(("Custom", None, "Leaf"), 7)
    step_logs.update(step_index=2, step_result={"ok": 1}, round_idx=3)
    step_logs.update_step(5, None)
    step_logs.update_step(5, {"ok": 2}, level2=("Round", 4), level3=("Loop", "tail"))
    with pytest.raises(AssertionError, match="overflow count must be integer"):
        step_logs.update_overflow("bad")  # type: ignore[arg-type]

    logs, _ = step_logs.getter()
    assert logs["Custom"]["Leaf"] == "0x07"
    assert logs["2nd Step"]["3rd Round"] == {"ok": "0x01"}
    assert logs["5th Step"]["4th Round"]["tail"] == {"ok": "0x02"}

    assert LogHelper.str_strip("0x1234") == "1234"
    assert LogHelper.str_to_bytes("1234") == b"\x12\x34"
    assert LogHelper.iter_to_bytes((0x12, 0x34), byteorder="big") == b"\x34\x12"
    assert LogHelper.bytes_to_int(b"\x12\x34", byteorder="big") == (0x34, 0x12)
    assert LogHelper.idx_setter(None, "Loop") is None
    assert LogHelper.idx_setter(0, "Round") == "Round 1"
    assert LogHelper.json_file_namer("md5", 128, "2026-03-25 12:34:56.000000+09:00", 3, 12).endswith(
        "_03.json"
    )

    timestamp = TimeHelper.get_current_timestamp()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}[+-]\d{2}:\d{2}", timestamp)
    started = TimeHelper.perftimer_start()
    elapsed = TimeHelper.perftimer_end(started)
    assert elapsed >= 0
    assert Logs.perftimer_str(1_000_000_000) == "1 s"
    assert Logs.perftimer_str(0) == "0 ns"

    metadata = Metadata(hash_alg="md5")
    Logs.clear()
    Logs.clear(metadata=metadata)
    assert metadata.started_at == ""


def test_md5_logger_decorators_cover_no_log_and_step4_paths() -> None:
    with pytest.raises(AssertionError, match="step_index must be positive"):
        MD5Logger.step(0)

    class DummyStep:
        def __init__(self, logs=None):
            self.logs = logs
            self.total_overflow_count = 5

        @MD5Logger.step(step_index=2, update_overflow=True)
        def run(self):
            return {"value": 1}

    no_log_dummy = DummyStep()
    assert no_log_dummy.run() == {"value": 1}

    logged_dummy = DummyStep(StepLogs(wordsize=32, byteorder="little", hierarchy=("Step",)))
    assert logged_dummy.run() == {"value": 1}
    logged_data, logged_meta = logged_dummy.logs.getter()
    assert logged_data["2nd Step"] == {"value": "0x01000000"}
    assert logged_meta["overflow_count"] == 5

    class DummyStep4:
        def __init__(self, logs=None):
            self.logs = logs

        @MD5Logger.step(step_index=4)
        def run(self):
            return MD5Step4Trace(
                updated_hash={"A": 1},
                rounds=[
                    MD5RoundTrace(
                        loop_start={"A": 1},
                        loop_states=[{"A": 2}],
                        loop_end={"A": 3},
                    )
                ],
            )

    assert DummyStep4().run() == {"A": 1}
    step4_logged = DummyStep4(StepLogs())
    step4_logged.run()
    assert step4_logged.logs.getter()[0]["4th Step"]["1st Round"]["Loop End"] == {"A": "0x03"}


def test_md5_logic_and_digest_cover_auxiliary_paths(capsys: pytest.CaptureFixture[str]) -> None:
    hash_cfg = HashConfig(hash_alg="md5", length=24)
    main_cfg = MainConfig(
        message_flag=True,
        verbose_flag=True,
        clean_flag=False,
        debug_flag=False,
        make_xlsx_flag=False,
        seed_flag=False,
    )
    logs = StepLogs(wordsize=32, byteorder="little", hierarchy=("Step", "Round", "Loop"))
    md5 = MD5(main_cfg, hash_cfg, logs)
    logic = MD5Logic(hash_cfg)
    calc = MD5Calc(hash_cfg)

    with logic._step4_outer(1, 2, 3, 4) as loop_params:
        loop_params["A"] = 5
        loop_params["B"] = 6
        loop_params["C"] = 7
        loop_params["D"] = 8

    assert loop_params["A"] == 6
    assert loop_params["B"] == 8
    assert loop_params["C"] == 10
    assert loop_params["D"] == 12

    assert calc.f_func(0x0F0F0F0F, 0x33333333, 0xAAAAAAAA) == 0xA3A3A3A3
    assert calc.g_func(0x0F0F0F0F, 0x33333333, 0xAAAAAAAA) == 0x1B1B1B1B
    assert calc.h_func(0x0F0F0F0F, 0x33333333, 0xAAAAAAAA) == 0x96969696
    assert calc.i_func(0x0F0F0F0F, 0x33333333, 0xAAAAAAAA) == 0x6C6C6C6C

    assert len(logic.step1(b"abc")) == 56
    blocks = logic.step2(logic.step1(b"abc"), 24)
    assert len(blocks) == 1
    assert len(blocks[0]) == 16
    assert logic.step3()["A"] == hash_cfg.init_hash["A"]

    digest = md5.digest(b"abc")
    hexdigest = md5.hexdigest(b"abc")
    out = capsys.readouterr().out
    assert "Original data length (bits): 24" in out
    assert digest.hex() == "900150983cd24fb0d6963f7d28e17f72"
    assert hexdigest == "0x900150983cd24fb0d6963f7d28e17f72"
