"""
Hash algorithm main module
"""

import argparse
import sys

from diffusion_hash_inv.logger import Logs, Metadata, BaseLogs, StepLogs
from diffusion_hash_inv.config import MainConfig, HashConfig, OutputConfig
from diffusion_hash_inv.generator import GenerateRandomNChar
from diffusion_hash_inv.utils import FileIO, RGBImgMaker
from diffusion_hash_inv.validation import validate
from diffusion_hash_inv import hashing

class Main:
    """
    Entry point for hash generation and validation
    """

    def __init__(self, main_config: MainConfig, \
                hash_config: HashConfig, \
                output_config: OutputConfig):
        self.start_time = Logs.get_current_timestamp()

        self.main_cfg = main_config
        self.hash_cfg = hash_config
        self.output_cfg = output_config
        print("Main configuration, Hash configuration, and Output configuration loaded.")
        print("=========================")
        print("Main Configuration:", self.main_cfg)
        print("Hash Configuration:", self.hash_cfg)
        print("Output Configuration:", self.output_cfg)


        self.alg_name = self.hash_cfg.hash_alg.upper()

        self.io_controller = FileIO(self.main_cfg, self.output_cfg)

    def message_generator(self, length:int, byteorder: str) -> bytes:
        """
        Generate random message for hashing
        """
        timer = Logs.perftimer_start()

        assert self.main_cfg.message_flag, "Bits generation is temporarily unavailable."
        generator = None

        if self.main_cfg.message_flag:
            generator = GenerateRandomNChar(self.main_cfg, self.io_controller)
        else:
            raise NotImplementedError("Bits generation is temporarily unavailable.")
            # generator = GenerateRandomNBits()
        assert generator is not None, "Generator is not initialized."

        _msg = generator.main(length, byteorder, timer_start=timer)
        return _msg

    @staticmethod
    def get_hasher(alg_name: str, main_cfg: MainConfig, hash_cfg: HashConfig, steplogs: StepLogs):
        """
        Get the hashing algorithm instance based on name
        """
        try:
            algo = getattr(hashing, alg_name)(main_cfg, hash_cfg, steplogs)
        except AttributeError as e:
            raise ValueError(f"Unsupported algo: {alg_name}") from e

        return algo

    def main(self, length:int, iteration: int = 0):
        """
        Main entry point for hash generation and validation
        """
        metadata = Metadata(hash_alg=self.alg_name, is_message=self.main_cfg.message_flag, \
                            input_bits_len=length, started_at=self.start_time)
        metadata.hash_property(byteorder=self.hash_cfg.byteorder, \
                            hierarchy=self.hash_cfg.hierarchy)

        baselogs = BaseLogs()
        steplogs = StepLogs(wordsize=self.hash_cfg.ws_bits, byteorder=self.hash_cfg.byteorder, \
                            hierarchy=self.hash_cfg.hierarchy)

        assert length > 0, "Length must be positive."
        assert length % 8 == 0, "Length must be multiple of 8."

        assert iteration >= 0, "Iteration count must be non-negative integer."

        if self.main_cfg.verbose_flag:
            print(f"Running {self.alg_name} Hash with length {length} "
                f"for {iteration} iterations.")

        algo = self.get_hasher(self.alg_name, self.main_cfg, self.hash_cfg, steplogs)

        for _i in range(iteration):
            perf_timer_start = Logs.perftimer_start()
            print(f"--- Iteration {_i + 1}/{iteration} ---")
            algo.reset()
            Logs.clear(baselogs=baselogs, steplogs=steplogs)

            json_file_name = Logs.json_file_namer(self.alg_name, length, \
                                                self.start_time, _i + 1, iteration)

            input_msg = self.message_generator(length, algo.byteorder)
            generated_hash = algo.digest(input_msg)
            valid, right_hash = \
                validate(generated_hash, input_msg, self.alg_name, self.main_cfg.verbose_flag)

            if not valid:
                print(f"Iteration {_i + 1}/{iteration} Hash validation failed. Exiting.")
                print(f"Generated Hash: {Logs.bytes_to_str(generated_hash)}")
                print(f"Correct   Hash: {Logs.bytes_to_str(right_hash)}")
                raise RuntimeError(f"Hash validation failed at iteration {_i + 1}.")

            print(f"Iteration {_i + 1}/{iteration} completed.\n")
            # Logs.stdout_writer(generated_hash, valid, correct_hash, iteration_index=_i)
            perf_timer_end = Logs.perftimer_end(perf_timer_start)
            metadata.time_logger(perf_timer_end)

            baselogs.update(message=input_msg, \
                            generated_hash=generated_hash, \
                            correct_hash=right_hash, \
                            is_message=self.main_cfg.message_flag)

            if self.main_cfg.debug_flag and _i == 0:
                breakpoint()

            self.io_controller.file_writer(filename=json_file_name, content={"metadata": metadata, \
                    "baselogs": baselogs, "steplogs": steplogs}, length=length)

    def run(self, length:int, iteration: int = 0):
        """
        Run the main process
        """
        if iteration == 0:
            sys.exit()

        _start_total = Logs.perftimer_start()

        self.main(length, iteration)

        _end_total = Logs.perftimer_end(_start_total)
        elapsed_time = Logs.perftimer_str(_end_total)
        print("Hash Calculation time:", elapsed_time)
        print("Process completed.")
        print("=========================")

        print("RGB Image Maker Module Loaded.")
        _img_make_start = Logs.perftimer_start()

        rgb_encoder = RGBImgMaker(self.main_cfg, self.hash_cfg, self.io_controller)
        rgb_encoder.main()

        _img_make_end = Logs.perftimer_end(_img_make_start)
        img_elapsed_time = Logs.perftimer_str(_img_make_end)
        print("RGB Image Maker execution time:", img_elapsed_time)
        print("RGB Image Maker process completed.")
        print("=========================")
        print()
        print("Total Execution Time:", Logs.perftimer_str(_end_total + _img_make_end))




if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Hash Generation and Image Creation Script")
    parser.add_argument('-l', '--length', type=int, default=argparse.SUPPRESS,
                        help='Length of random bits to generate (default: 512)')
    parser.add_argument('-e', '--exponentiation', type=int, default=argparse.SUPPRESS,
                        help='2 to the power of <exponentiation> (default: 9)')

    parser.add_argument('-i', '--iteration', type=int, default=0,
                        help='Running iterations (default: 0)')

    parser.add_argument('--hash_alg', type=str, default='md5',
                        help='Hash algorithm to use (default: md5)')

    gv = parser.add_mutually_exclusive_group()
    gv.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                    help='Enable verbose output')
    gv.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                    help='Suppress output')
    parser.set_defaults(verbose=True)

    gm = parser.add_mutually_exclusive_group()
    gm.add_argument('-m', '--message', action="store_true",
                    dest='message', help='Message input mode')
    gm.add_argument('-b', '--bit', action="store_false",
                    dest='message', help='Bit string input mode')
    parser.set_defaults(message=True)

    gc = parser.add_mutually_exclusive_group()
    gc.add_argument('-c', '--clear', action='store_true',
                    dest='clear', help='Clear generated files')
    gc.add_argument('-C', '--no-clear', action='store_false', dest='clear',
                    help='Do not clear generated files (default)')

    parser.add_argument('--make-xlsx', action='store_true',
                        dest='make_xlsx', help='Convert JSON logs to XLSX after completion')

    parser.set_defaults(clear=False)
    parser.set_defaults(make_xlsx=False)

    _args = parser.parse_args()

    LENGTH = 256
    if hasattr(_args, 'length'):
        LENGTH = _args.length
    else:
        pass

    if hasattr(_args, 'exponentiation'):
        EXP = _args.exponentiation
        LENGTH = 2 ** EXP
    else:
        pass
    DEBUG = False

    _main_flags = MainConfig(
        message_flag=_args.message,
        verbose_flag=_args.verbose,
        clean_flag=_args.clear,
        debug_flag=DEBUG,
        make_xlsx_flag=_args.make_xlsx,
        seed_flag=False,  # Enable random seed generation for reproducibility
    )
    _hash_flags = HashConfig(
        hash_alg=_args.hash_alg,
        length=LENGTH,
    )
    _output_config = OutputConfig()

    main_app = Main(_main_flags, _hash_flags, _output_config)
    main_app.run(length=LENGTH, iteration=_args.iteration)
