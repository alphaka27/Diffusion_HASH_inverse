"""
Hash algorithm main module
"""

import argparse

from diffusion_hash_inv.config import MainConfig
from diffusion_hash_inv.config import MessageConfig
from diffusion_hash_inv.config import HashConfig
from diffusion_hash_inv.config import OutputConfig
from diffusion_hash_inv.config import Byte2RGBConfig
from diffusion_hash_inv.main import MainEP
from diffusion_hash_inv.main import RuntimeState, RuntimeConfig

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Hash Generation and Image Creation Script")
    parser.add_argument('--hash_alg', type=str, default='md5',
                        help='Hash algorithm to use (default: md5)')

    gt = parser.add_mutually_exclusive_group()
    gt.add_argument('-l', '--length', type=int, default=argparse.SUPPRESS,
                        help='Length of random bits to generate (default: 512)')
    gt.add_argument('-e', '--exponentiation', type=int, default=argparse.SUPPRESS,
                        help='2 to the power of <exponentiation> (default: 9)')

    parser.add_argument('-i', '--iteration', type=int, default=0,
                        help='Running iterations (default: 0)')

    gm = parser.add_mutually_exclusive_group()
    gm.add_argument('-m', '--message', action="store_true",
                    dest='message', help='Message input mode')
    gm.add_argument('-b', '--bit', action="store_false",
                    dest='message', help='Bit string input mode')
    parser.set_defaults(message=False)

    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                    help='Enable verbose output (default: False)')
    parser.set_defaults(verbose=False)

    parser.add_argument('-c', '--clear', action='store_true', default=False,
                    help='Do not clear generated files (default: False)')

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

    # Initialize configurations
    main_config = MainConfig(verbose_flag=_args.verbose,
                            clean_flag=_args.clear,
                            debug_flag=DEBUG,
                            make_image_flag=False)
    msg_config = MessageConfig(
                            message_flag=_args.message,
                            length=LENGTH,
                            random_flag=True,
                            )
    hash_config = HashConfig(hash_alg=_args.hash_alg,
                            length=LENGTH,)
    output_config = OutputConfig()
    byte2rgb_config = Byte2RGBConfig()
    runtime_config = RuntimeConfig(main=main_config, message=msg_config, hash=hash_config,
                                output=output_config, rgb=byte2rgb_config)
    # Run main process
    main_ep = MainEP(runtime_config)
    main_ep.run(_args.iteration)
