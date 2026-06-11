"""
n-bits character generator for hash input.
n is must be a multiple of 8.
"""

from typing import Optional, TYPE_CHECKING
import unicodedata
import random

from diffusion_hash_inv.config import RuntimeConfig
from diffusion_hash_inv.utils.file_io import FileIO
from diffusion_hash_inv.logger import Logs
if TYPE_CHECKING:
    from diffusion_hash_inv.config import MessageConfig, HashConfig


class NCharsGenerator:
    """
    Generate a random string of N characters.
    """

    def __init__(self, runtime_config: RuntimeConfig,
                io_controller: FileIO, program_start_time: str) -> None:
        self.runtime_config = runtime_config
        self.msg_cfg: MessageConfig = runtime_config.message
        self.hash_cfg: HashConfig = runtime_config.hash

        self.io_controller = io_controller

        self.program_start = program_start_time

    def help(self):
        """
        Provide help information for the password generator.
        """
        description = "Generate a random string of N characters\n"
        alphabet_info = (
            f"Included characters Length: {len(self.msg_cfg.candidate_list)}\n"
        )
        print(description +
            alphabet_info +
            self.msg_cfg.candidate_list,
            end="\n\n")

    def generate(self, value: Optional[int] = None) -> bytes:
        """
        Generate n bits data and print its hexadecimal and binary representations.

        Args:
            value (Optional[int]):
                If random_flag is False, this value will be used to generate the data.
        """
        candidate_list = self.msg_cfg.candidate_list
        if self.msg_cfg.random_flag:
            _pwd = ''.join(random.choice(candidate_list) for _ in range(self.msg_cfg.length))
        else:
            assert value is not None, "Value must be provided when random_flag is False"
            random.seed(value)
            _pwd = ''.join(random.choice(candidate_list) for _ in range(self.msg_cfg.length))

        return self.normalize(_pwd)

    def normalize(self, s: str, form: str = "NFKC") -> bytes:
        """
        Normalize a string to the specified Unicode normalization form.

        Args:
            s (str): The input string to normalize.
            form (str): The Unicode normalization form to use. 
                        Must be one of "NFKC", "NFKD", "NFC", or "none".

        Returns:
            bytes: The normalized string encoded as UTF-8 bytes.
        """
        assert form in ["NFKC", "NFKD", "NFC", "none"], "Invalid normalization form"
        s = unicodedata.normalize(form.upper(), s)
        return s.encode("utf-8")

    def main(self, value: Optional[int] = None) -> bytes:
        """
        Main function to generate n bits data.
        """
        msg = None
        main_start_time = Logs.get_current_timestamp()
        perf_timer_start = Logs.perftimer_start()
        msg = self.generate(value)
        perf_timer_end = Logs.perftimer_end(perf_timer_start)

        assert msg is not None, "Message generation failed."
        filename = f"_{self.msg_cfg.length}_chars_{self.program_start[:19]}.bin"
        if self.msg_cfg.random_flag:
            filename = "random" + filename
        else:
            filename = "fixed" + filename
        self.io_controller.file_writer(filename,
                                    msg,
                                    length=self.msg_cfg.length,
                                    timestamp=main_start_time,
                                    elapsed_time=perf_timer_end,
                                    byteorder=self.hash_cfg.constants.byteorder)

        return msg
