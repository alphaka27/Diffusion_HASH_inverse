import json
import re

from diffusion_hash_inv.config import HashConfig, MainConfig
from diffusion_hash_inv.core import BaseLogs, Metadata, StepLogs
from diffusion_hash_inv.hashing.md5 import MD5
from diffusion_hash_inv.utils.formatter import JSONFormat


def _assert_state_dict(state: dict) -> None:
    assert set(state.keys()) == {"A", "B", "C", "D"}
    for value in state.values():
        assert isinstance(value, str)
        assert value.startswith("0x")


def test_md5_step4_log_format_for_512bit_input() -> None:
    msg = b"A" * 64  # 512-bit input -> 2 blocks after MD5 padding
    hash_cfg = HashConfig(hash_alg="md5", length=512)
    main_cfg = MainConfig(
        message_flag=True,
        verbose_flag=False,
        clean_flag=False,
        debug_flag=False,
        make_xlsx_flag=False,
    )
    steplogs = StepLogs(
        wordsize=hash_cfg.ws_bits,
        byteorder=hash_cfg.byteorder,
        hierarchy=hash_cfg.hierarchy,
    )
    md5 = MD5(main_cfg, hash_cfg, steplogs)

    digest_bytes = md5.digest(msg)
    logs, step_meta = steplogs.getter()

    assert "4th Step" in logs
    step4 = logs["4th Step"]

    # 4th Step top-level keys should only contain rounds.
    assert all(key.endswith("Round") for key in step4.keys())

    expected_rounds = ("1st Round", "2nd Round")
    assert tuple(step4.keys()) == expected_rounds

    for round_key in expected_rounds:
        round_data = step4[round_key]
        round_keys = list(round_data.keys())

        # Key order: Loop Start -> loops -> Loop End
        assert round_keys[0] == "Loop Start"
        assert round_keys[-1] == "Loop End"

        assert "1st Loop" in round_data
        assert "64th Loop" in round_data
        assert "21st Loop" in round_data
        assert "21th Loop" not in round_data

        loop_keys = [k for k in round_keys if re.fullmatch(r"\d+(st|nd|rd|th) Loop", k)]
        assert len(loop_keys) == 64

        _assert_state_dict(round_data["Loop Start"])
        _assert_state_dict(round_data["Loop End"])
        for loop_key in loop_keys:
            _assert_state_dict(round_data[loop_key])

    # Round continuity: second round starts from first round end state.
    assert step4["2nd Round"]["Loop Start"] == step4["1st Round"]["Loop End"]

    # Step metadata and JSON dump compatibility.
    assert step_meta["overflow_count"] >= 0
    assert step_meta["word_size"] == 32
    assert step_meta["byteorder"] == "little"

    metadata = Metadata(
        hash_alg="md5",
        is_message=True,
        started_at="2026-02-10 00:00:00.000000+00:00",
        input_bits_len=512,
    )
    metadata.hash_property(byteorder=hash_cfg.byteorder, hierarchy=hash_cfg.hierarchy)
    metadata.time_logger(0)

    baselogs = BaseLogs()
    baselogs.update(
        message=msg,
        is_message=True,
        generated_hash=digest_bytes,
        correct_hash=digest_bytes,
    )

    dumped = JSONFormat.dumps(metadata=metadata, baselogs=baselogs, steplogs=steplogs)
    dumped_json = json.loads(dumped)
    assert "Logs" in dumped_json
    assert "4th Step" in dumped_json["Logs"]
