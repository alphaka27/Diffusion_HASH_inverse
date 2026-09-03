"""Regression tests for the JSON hash-execution tracer."""

from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from diffusion_hash_inv import generate_bytes, generate_message, select_characters
from diffusion_hash_inv.cli import main as cli_main
from diffusion_hash_inv.hashing import (
    HashTracer,
    register_hash_function,
    trace_hash,
)


class HashTraceTests(unittest.TestCase):
    def test_md5_matches_hashlib_and_records_all_steps(self) -> None:
        trace = trace_hash("md5", b"abc")

        self.assertEqual(trace["digest"], hashlib.md5(b"abc").hexdigest())
        block = trace["intermediate"]["blocks"][0]
        self.assertEqual(len(block["rounds"]), 4)
        self.assertTrue(all(len(round_["steps"]) == 16 for round_ in block["rounds"]))
        self.assertEqual(block["rounds"][0]["steps"][0]["state_before"], trace["intermediate"]["initial_state"])

    def test_sha256_matches_hashlib_and_records_schedule_and_rounds(self) -> None:
        message = b"A" * 64  # Produces two padded SHA-256 blocks.
        trace = trace_hash("SHA-256", message)

        self.assertEqual(trace["digest"], hashlib.sha256(message).hexdigest())
        self.assertEqual(trace["preprocessing"]["block_count"], 2)
        for block in trace["intermediate"]["blocks"]:
            self.assertEqual(len(block["message_schedule"]), 64)
            self.assertEqual(len(block["message_schedule_expansion"]), 48)
            self.assertEqual(len(block["compression_rounds"]), 64)

    def test_trace_is_written_as_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "trace.json"
            returned = trace_hash("sha256", "한글 text", output_path)
            with output_path.open(encoding="utf-8") as trace_file:
                persisted = json.load(trace_file)

        self.assertEqual(persisted, returned)
        self.assertEqual(returned["input"]["encoding"], "utf-8")

    def test_a_custom_implementation_is_available_through_the_same_entry_point(self) -> None:
        class ExampleTracer(HashTracer):
            algorithm = "example"

            def trace(self, message: bytes) -> dict[str, object]:
                return {"schema_version": 1, "algorithm": self.algorithm, "digest": message.hex()}

        register_hash_function("example", ExampleTracer)
        trace = trace_hash("example", b"\x00\xff")

        self.assertEqual(trace["digest"], "00ff")

    def test_generated_message_is_repeatable_with_a_seed_and_hashable(self) -> None:
        characters = select_characters("lowercase", "digits")
        message = generate_message(32, characters, seed="test-seed")

        self.assertEqual(message, generate_message(32, characters, seed="test-seed"))
        self.assertEqual(len(message), 32)
        self.assertTrue(set(message).issubset(characters))
        self.assertEqual(trace_hash("sha256", message)["digest"], hashlib.sha256(message.encode()).hexdigest())
        with self.assertRaises(ValueError):
            generate_message(-1, characters)
        with self.assertRaises(ValueError):
            generate_message(1, "")
        with self.assertRaises(ValueError):
            select_characters("unknown")

    def test_generated_bytes_are_repeatable_with_a_seed_and_hashable(self) -> None:
        value = generate_bytes(32, seed="test-seed")

        self.assertEqual(value, generate_bytes(32, seed="test-seed"))
        self.assertEqual(len(value), 32)
        self.assertEqual(trace_hash("sha256", value)["digest"], hashlib.sha256(value).hexdigest())
        with self.assertRaises(ValueError):
            generate_bytes(-1)

    def test_command_line_accepts_input_text_and_writes_a_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "traces"
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    ["--algorithm", "md5", "--input-text", "hello", "--output", str(output_directory)]
                )
            output_path = next((output_directory / "trace").iterdir())
            with output_path.open(encoding="utf-8") as trace_file:
                trace = json.load(trace_file)

        self.assertEqual(exit_code, 0)
        self.assertRegex(output_path.name, r"md5_5_\d{8}T\d{6}\.json")
        self.assertEqual(trace["input"]["type"], "text")
        self.assertEqual(trace["input"]["hex"], b"hello".hex())
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["digest"], hashlib.md5(b"hello").hexdigest())
        self.assertEqual(Path(result["output"]), output_path)

    def test_command_line_generates_a_seeded_input_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "traces"
            with redirect_stdout(StringIO()):
                exit_code = cli_main(
                    [
                        "--input-length",
                        "24",
                        "--character-groups",
                        "lowercase",
                        "digits",
                        "--seed",
                        "test-seed",
                        "--output",
                        str(output_directory),
                    ]
                )
            output_path = next((output_directory / "trace").iterdir())
            with output_path.open(encoding="utf-8") as trace_file:
                trace = json.load(trace_file)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            bytes.fromhex(trace["input"]["hex"]).decode(),
            generate_message(24, select_characters("lowercase", "digits"), seed="test-seed"),
        )


if __name__ == "__main__":
    unittest.main()
