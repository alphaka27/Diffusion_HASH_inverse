"""
Hash algorithm main module
"""

import argparse
import sys
from dataclasses import dataclass

from diffusion_hash_inv.common import Logs, Metadata, BaseLogs
from diffusion_hash_inv.generator import GenerateRandom, GenerateRandomNChar
from diffusion_hash_inv.utils import FileIO, JSONToXLSXConverter
from diffusion_hash_inv.validation.hash_validation import validate
from diffusion_hash_inv import hashing

@dataclass
class Flags:
    """
    Command line flags
    """
    is_message: bool
    is_verbose: bool
    is_clean: bool
    is_debug: bool
    make_xlsx: bool

class Main:
    """
    Entry point for hash generation and validation
    """
    def __init__(self, *flags, hash_alg: str = "sha256"):
        _is_m, _is_v, _is_c, _is_d, _make_xlsx = flags
        self.flags = \
            Flags(is_message=_is_m, is_verbose=_is_v, is_clean=_is_c, \
                is_debug=_is_d, make_xlsx=_make_xlsx)

        self.alg_name = hash_alg

        self.file_io = FileIO(verbose_flag=self.flags.is_verbose)
        if self.flags.is_clean:
            self.file_io.file_clean(clear_flag=self.flags.is_clean, \
                                    verbose_flag=self.flags.is_verbose)
        self.flags.is_clean = False

        self.start_time = Logs.get_current_timestamp()

    def message_generator(self, length:int, byteorder: str) -> bytes:
        """
        Generate random message for hashing
        """
        timer = Logs.perftimer_start()
        if self.flags.is_message:
            generator = GenerateRandomNChar(verbose_flag=self.flags.is_verbose, \
                                                start_timestamp=self.start_time)
        else:
            generator = GenerateRandom(verbose_flag=self.flags.is_verbose, \
                                    start_timestamp=self.start_time)
        _msg = generator.main(length, byteorder, timer_start=timer)
        return _msg

    def get_hasher(self):
        """
        Get the hashing algorithm instance based on name
        """
        n = self.alg_name

        try:
            algo = getattr(hashing, n.upper())\
                    (is_verbose=self.flags.is_verbose)
        except AttributeError as e:
            if self.alg_name == "all":
                pass
            raise ValueError(f"Unsupported algo: {self.alg_name}") from e

        return algo

    def main(self, length:int, iteration: int = 0):
        """
        Main entry point for hash generation and validation
        """
        metadata = Metadata(hash_alg=self.alg_name, is_message=self.flags.is_message)
        metadata.setter(input_length=length, \
                            exec_start=self.start_time)
        baselogs = BaseLogs()

        if iteration == 0:
            sys.exit()

        assert length > 0, "Length must be positive."
        assert length % 8 == 0, "Length must be multiple of 8."

        assert iteration > 0, "Iteration count must be non-negative integer."
        if self.flags.is_verbose:
            print(f"Running {self.alg_name.upper()} Hash with length {length} "
                f"for {iteration} iterations.")

        algo = self.get_hasher()

        for _i in range(iteration):
            perf_timer_start = Logs.perftimer_start()
            print(f"--- Iteration {_i + 1}/{iteration} ---")
            algo.reset()

            json_file_name = f"{self.alg_name}_{length}_{self.start_time[:19]}_{_i}.json"

            input_msg = self.message_generator(length, algo.byteorder)
            generated_hash = algo.digest(input_msg)
            valid, correct_hash = \
                validate(generated_hash, input_msg, self.alg_name, self.flags.is_verbose)

            if not valid:
                print(f"Iteration {_i + 1}/{iteration} Hash validation failed. Exiting.")
                raise RuntimeError(f"Hash va!lidation failed at iteration {_i + 1}.")

            print(f"Iteration {_i + 1}/{iteration} completed.\n")
            # Logs.stdout_writer(generated_hash, valid, correct_hash, iteration_index=_i)

            metadata.update(entropy=Metadata.calc_entropy(length, input_msg), \
                            elapsed_time=Logs.perftimer_end(perf_timer_start))

            baselogs.update(message=input_msg, \
                            generated_hash=generated_hash, \
                            correct_hash=correct_hash, \
                            is_message=self.flags.is_message)

            if self.flags.is_debug:
                if _i == 0:
                    breakpoint()

            self.file_io.file_writer(filename=json_file_name, content={"metadata": metadata, \
                    "baselogs": baselogs, "steplogs": algo.step_logs}, length=length)

    def run(self, length:int, iteration: int = 0):
        """
        Run the main process
        """
        json_to_xlsx_converter = JSONToXLSXConverter(verbose_flag=self.flags.is_verbose, \
                                                    length=length)
        _start_total = Logs.perftimer_start()
        self.main(length, iteration)
        _end_total = Logs.perftimer_end(_start_total)
        elapsed_time = Logs.perftimer(_end_total)
        print(f"Total execution time: {elapsed_time}")

        if self.flags.make_xlsx:
            _start = Logs.perftimer_start()
            json_to_xlsx_converter.convert_to_xlsx(self.alg_name.lower())
            print(f"JSON to XLSX conversion completed in "
                f"{Logs.perftimer_end(_start)} ns.")



if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="SHA-256 Hash Generator")
    parser.add_argument('-l', '--length', type=int, default=argparse.SUPPRESS,
                        help='Length of random bits to generate (default: 512)')
    parser.add_argument('-e', '--exponentiation', type=int, default=argparse.SUPPRESS,
                        help='2 to the power of <exponentiation> (default: 9)')

    parser.add_argument('-i', '--iteration', type=int, default=0,
                        help='Running iterations (default: 0)')

    parser.add_argument('--hash', type=str, default='md5',
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
    gc.add_argument('-C', '--no-clear', action='store_true', dest='clear',
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
    MAKE_XLSX = False
    if hasattr(_args, 'make_xlsx'):
        MAKE_XLSX = _args.make_xlsx

    Main(_args.message, _args.verbose, _args.clear, DEBUG, MAKE_XLSX, \
        hash_alg=_args.hash).run(length=LENGTH, iteration=_args.iteration)
