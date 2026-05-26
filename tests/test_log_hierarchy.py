from pathlib import Path

from diffusion_hash_inv.logger import Logs


class _LogReader:
    def __init__(self, log: dict) -> None:
        self.log = log

    def file_reader(self, _path: Path) -> dict:
        return self.log


def test_iter_logs_with_hierarchy_adds_block_level(tmp_path: Path) -> None:
    log = {
        "Hierarchy": ["Step", "Round", "Loop"],
        "Message": {"Hex": "0x00"},
        "Logs": {
            "2nd Step": {
                "1st Block": [
                    "0x00000000",
                    "0x00000001",
                ],
            },
        },
    }
    hierarchy: list[str] = []
    log_path = tmp_path / "MD5_16_2026-04-25 20-40-00_00000.json"

    parsed_logs = list(Logs.iter_logs_with_hierarchy(_LogReader(log), hierarchy, [log_path]))
    _, _, step_logs = Logs.log_parser(parsed_logs[0])

    assert hierarchy == ["Step", "Round", "Loop", "Block"]
    assert Logs.steplogs_parser(step_logs, hierarchy) == (
        {"2nd Step/1st Block": ["0x00000000", "0x00000001"]},
    )
