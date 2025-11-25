"""
Analyze Hashing intermediate process
"""

import hashlib
import math
import argparse
import json
from pathlib import Path
import os

import numpy as np
from diffusion_hash_inv.utils.project_root import get_project_root

class Analyze:
    """
    SHA256 intermediate process analyzer
    """
    def __init__(self, data_path, is_verbose):
        self.data = []
        for _fp in os.listdir(data_path):
            _file = os.path.join(data_path, _fp)
            with open(_file, "r", encoding="utf-8") as rf:
                self.data.append(json.load(rf))
        if is_verbose:
            print(self.data)
            print(len(self.data))
            print(self.data[0])

    def __getattribute__(self, name=None, iteration=0):
        if name is None:
            ret = self.data
        else:
            ret = self.data[iteration][name]
        return ret

    def avalanche(self):
        """
        avalanche rate analyze
        """

    def haming_distnace(self):
        """
        haming distance analyze
        """

    def subtract(self):
        """
        subtract each step
        """

def main(file_path, is_verbose):
    """
    Main algorithm
    """
    analyzer = Analyze(file_path, is_verbose)
    __data = analyzer.data


if __name__=="__main__":
    project_root = get_project_root()
    _data_path = project_root / "output" / "json"
    # Argument parsing
    parser = argparse.ArgumentParser(description="SHA-256 Hash Analyzer")
    parser.add_argument('-f', '--file_path', type=str, default=str(_data_path / "256"),
                        help='Set output directory')

    gv = parser.add_mutually_exclusive_group()
    gv.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                    help='Enable verbose output')
    gv.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                    help='Suppress output')
    parser.set_defaults(verbose=True)

    _args = parser.parse_args()

    main(_args.file_path, _args.verbose)
