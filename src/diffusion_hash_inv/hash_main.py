"""
Hash algorithm main module
"""

import argparse
import sys
from dataclasses import dataclass

try:
    from diffusion_hash_inv.generator import GenerateRandom
except ImportError as e:
    print(f"Error importing GenerateRandom: {e}")

try:
    from diffusion_hash_inv.generator import GenerateRandomNChar
except ImportError as e:
    print(f"Error importing GenerateRandomNChar: {e}")

try:
    from diffusion_hash_inv.utils import FileIO
except ImportError as e:
    print(f"Error importing FileIO: {e}")

try:
    from diffusion_hash_inv.utils import OutputFormat
except ImportError as e:
    print(f"Error importing OutputFormat: {e}")

try:
    from diffusion_hash_inv.utils import CSVFormat
except ImportError as e:
    print(f"Error importing CSVFormat: {e}")

try:
    from diffusion_hash_inv.validation.hash_validation import validate
except ImportError as e:
    print(f"Error importing validate: {e}")

try:
    from diffusion_hash_inv import hashing
except ImportError as e:
    print(f"Error importing hashing module: {e}")

@dataclass
class Flags:
    """
    Command line flags
    """
    is_message: bool
    is_verbose: bool
    is_clean: bool
    is_main: bool

class Main:
    """
    Entry point for hash generation and validation
    """
    def __init__(self, *flags, length: int = 512, hash_alg: str = "sha256"):
        print(len(flags))
        _is_m, _is_v, _is_c = flags
        self.flags = Flags(is_message=_is_m, is_verbose=_is_v, is_clean=_is_c, is_main=True)
        assert length > 0, "Length must be positive."

        assert length % 8 == 0, "Length must be multiple of 8."
        self.length = length
        self.alg_name = hash_alg

        self.file_io = FileIO(init_flag=True, clear_flag=self.flags.is_clean,
                            verbose_flag=self.flags.is_verbose, length=self.length)
        self.flags.is_clean = False
        self.json_formatter = OutputFormat()
        self.csv_formatter = CSVFormat()
        self.start_time = self.file_io.encode_timestamp().decode("UTF-8")

    def message_generator(self):
        """
        Generate random message for hashing
        """
        if self.flags.is_message:
            generator = GenerateRandomNChar(verbose_flag=self.flags.is_verbose,
                                            main_flag=self.flags.is_main)
            _msg = generator.main(self.start_time, self.length // 8)
        else:
            generator = GenerateRandom(verbose_flag=self.flags.is_verbose,
                                        main_flag=self.flags.is_main)
            _msg = generator.main(self.start_time, self.length)

        self.flags.is_main = False
        _entropy = generator.calc_entropy(len(_msg.decode("UTF-8")), _msg)
        _strength = self.json_formatter.set_metadata(self.alg_name,
                                                    len(_msg)*8, self.start_time, 0, _entropy)
        self.json_formatter.set_message(_msg, self.flags.is_message)

        return _msg, _entropy, _strength

    def file_writer(self, result_df: dict, steps_logs: dict, iteration_index:int, total_iter: int):
        """
        Write output to files
        """
        _result_df = self.csv_formatter.df_accumulate(result_df, steps_logs, iteration_index)

        xlsx_file_name = f"{self.alg_name}_{self.length}_{self.start_time[:19]}.xlsx"
        xlsx_writer, _ = self.file_io.file_io(xlsx_file_name)
        if _result_df is None:
            pass
        if iteration_index + 1 == total_iter:
            xlsx_writer(_result_df)

        return _result_df

    def stdout_writer(self, *hash_info, msg: str, entropy: float, strength: str,
                    iteration_index: int):
        """
        Write output to stdout
        """
        generated_hash, valid, correct_hash = hash_info
        if self.flags.is_message:
            decoded_msg = msg.decode("UTF-8")
            str_len_bits = f"{len(decoded_msg) * 8} bits"

        else:
            decoded_msg = msg
            str_len_bits = f"{len(decoded_msg)} bits"

        str_entropy = f"Entropy = {entropy}"
        str_strength = f"Strength = {strength}"

        print(f"Input ({str_len_bits}, {str_entropy}, {str_strength}):")
        print(f"{decoded_msg}\n")
        print(f"----------------Result for iteration ({iteration_index + 1})----------------")
        print(f"Generated {self.alg_name.upper()} Hash: ")
        hex_str = ''.join(f"{x:08x}" for x in generated_hash)
        print(f"{hex_str}\n")

        print(f"--------------Validation for iteration ({iteration_index + 1})--------------")
        print(f"Correct {self.alg_name.upper()} Hash: ")
        print(f"{correct_hash}\n")

        print("--------------Validation result--------------")
        valid_message = "Fail" if not valid else "Success"
        print(f"Validation: {valid_message}")

    def get_hasher(self):
        """
        Get the hashing algorithm instance based on name
        """
        n = self.alg_name

        try:
            algo = getattr(hashing, n.upper())\
                    (is_verbose=self.flags.is_verbose, output_format=self.json_formatter)
        except AttributeError as e:
            if self.alg_name == "all":
                pass
            raise ValueError(f"Unsupported algo: {self.alg_name}") from e

        return algo

    def main(self, iteration: int = 0):
        """
        Main entry point for hash generation and validation
        """
        if iteration == 0:
            sys.exit()

        assert iteration > 0, "Iteration count must be non-negative integer."
        if self.flags.is_verbose:
            print(f"Running {self.alg_name.upper()} Hash with length {self.length} "
                f"for {iteration} iterations.")

        algo = self.get_hasher()

        result_df = None

        for _i in range(iteration):
            print(f"--- Iteration {_i + 1}/{iteration} ---")
            algo.reset()

            json_file_name = f"{self.alg_name}_{self.length}_{self.start_time[:19]}_{_i}.json"
            json_writer, _ = self.file_io.file_io(json_file_name)

            input_msg, entropy, strength = self.message_generator()
            generated_hash = algo.digest(input_msg)
            valid, valid_hash, correct_hash = \
                validate(generated_hash, input_msg, self.alg_name, self.flags.is_verbose)

            self.stdout_writer(generated_hash, valid, correct_hash, msg=input_msg,
                            entropy=entropy, strength=strength, iteration_index=_i)

            if not valid:
                print(f"Iteration {_i + 1}/{iteration} Hash validation failed. Exiting.")
                raise RuntimeError(f"Hash va!lidation failed at iteration {_i + 1}.")

            print(f"Iteration {_i + 1}/{iteration} completed.\n")

            self.json_formatter.set_hashes(valid_hash, correct_hash)

            steps_logs = self.json_formatter.to_dict()
            result_df = self.file_writer(result_df, steps_logs, _i, iteration)

            if self.flags.is_verbose:
                print(result_df)
            json_writer(self.json_formatter.dumps(indent=4, data=steps_logs))


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
    parser.set_defaults(clear=False)
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

    Main(_args.message, _args.verbose, _args.clear,
        length=LENGTH, hash_alg=_args.hash).main(_args.iteration)
