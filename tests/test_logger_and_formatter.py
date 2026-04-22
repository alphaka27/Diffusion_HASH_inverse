import json

from diffusion_hash_inv.logger import BaseLogs, Logs, Metadata, StepLogs
from diffusion_hash_inv.utils.formatter import JSONFormat


def test_step_logs_normalize_nested_values_and_metadata() -> None:
    step_logs = StepLogs(wordsize=32, byteorder="little", hierarchy=("Step", "Round", "Loop"))

    step_logs.update(step_index=1, step_result=b"\x01\x02")
    step_logs.update(step_index=2, step_result=[b"\xAA\xBB", b"\xCC\xDD"])
    step_logs.update_loop(
        step_index=4,
        round_idx=2,
        loop_index=11,
        step_result={"A": 1, "B": 2},
    )
    step_logs.update_step(
        "Named Step",
        {"payload": 3},
        level2=("Round", "Custom Round"),
        level3=("Loop", 5),
    )
    logs, metadata = step_logs.getter()

    assert StepLogs.index_label(21, "Loop") == "21st Loop"
    assert logs["1st Step"] == "0x0102"
    assert logs["2nd Step"]["1st Block"] == "0xaabb"
    assert logs["2nd Step"]["2nd Block"] == "0xccdd"
    assert logs["4th Step"]["2nd Round"]["11th Loop"] == {"A": "0x01000000", "B": "0x02000000"}
    assert logs["Named Step"]["Custom Round"]["5th Loop"] == {"payload": "0x03000000"}
    assert metadata["word_size"] == 32
    assert metadata["byteorder"] == "little"
    assert metadata["hierarchy"] == ["Step", "Round", "Loop"]
    assert metadata["overflow_count"] == 0


def test_json_format_round_trip_and_log_clear() -> None:
    metadata = Metadata(
        hash_alg="md5",
        is_message=True,
        started_at="2026-03-25 12:00:00.000000+09:00",
        input_bits_len=24,
    )
    metadata.hash_property(byteorder="little", hierarchy=("Step", "Round", "Loop"))
    metadata.time_logger(1_002_003)

    base_logs = BaseLogs()
    base_logs.update(
        message=b"\xffA",
        is_message=True,
        generated_hash=b"\x01\x02",
        correct_hash=b"\x01\x02",
    )

    step_logs = StepLogs(wordsize=32, byteorder="little", hierarchy=("Step", "Round", "Loop"))
    step_logs.update(step_index=1, step_result=b"\x10\x20")

    dumped = JSONFormat.dumps(metadata=metadata, baselogs=base_logs, steplogs=step_logs)
    parsed = json.loads(dumped)
    loaded = JSONFormat.loads(dumped)

    assert parsed["Message"]["Bit String"] == "\ufffdA"
    assert parsed["Generated hash"] == "0x0102"
    assert loaded["Hash Algorithm"] == "md5"
    assert loaded["Byte order"] == "little"
    assert loaded["Length"] == 24
    assert loaded["Logs"]["1st Step"] == "0x1020"
    assert Logs.iter_to_bytes((0x12, 0x34), byteorder="little") == b"\x12\x34"
    assert Logs.bytes_to_int(b"\x12\x34", byteorder="little") == (0x12, 0x34)
    assert Logs.byte_to_int(b"\xFF") == 255
    assert Logs.int_to_bytes(0x3412, 16, byteorder="little") == b"\x12\x34"
    assert Logs.perftimer_str(1_002_003) == "1 ms, 2 us, 3 ns"

    Logs.clear(metadata=metadata, base_logs=base_logs, step_logs=step_logs)
    assert metadata.started_at == ""
    assert metadata.elapsed_time == ""
    assert base_logs.getter() == {
        "Message": {},
        "Generated hash": b"",
        "Correct   hash": b"",
    }
    assert step_logs.getter()[0] == {}
