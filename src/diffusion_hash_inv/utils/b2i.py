"""
Byte string to image conversion utilities with EMNIST dataset.
Input: Byte string in hex representation
Output: Image
"""
from pathlib import Path
import argparse
from functools import partial

from torchvision import datasets, transforms
import torchvision.transforms.functional as F

from diffusion_hash_inv.utils import FileIO
from diffusion_hash_inv.utils import add_root_to_path
ROOT_DIR = add_root_to_path()

class ByteToImageConverter:
    """
    Converts byte strings to images using the EMNIST dataset.
    """
    def __init__(self, standalone: bool = False, **kwargs):
        self.standalone = standalone

        self.is_verbose: bool = kwargs.pop("is_verbose", not self.standalone)

        self.length = kwargs.pop("length", 256)
        print(f"Length set to: {self.length}")

        self.file_io = FileIO(verbose_flag=self.is_verbose)

        self.img_path_arg = kwargs.pop("img_path", None)
        self.json_path_arg = kwargs.pop("json_path", None)

        self.hash_alg = kwargs.pop("hash_alg", None)


    def emnist_save(self, img_path: Path) -> None:
        """
        EMNIST dataset loading function
        
        :param self: Description
        """
        transform = transform = transforms.Compose([
            partial(F.rotate, angle=-90),
            F.hflip,
        ])
        train_dataset = datasets.EMNIST(
            root=ROOT_DIR / "data",
            split="byclass",
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.EMNIST(
            root=ROOT_DIR / "data",
            split="byclass",
            train=False,
            download=True,
            transform=transform,
        )
        all_dataset = train_dataset + test_dataset
        self.file_io.file_writer(filename=img_path, content=all_dataset, length=self.length)


    def get_json_list(self, hash_alg: str):
        """
        Loads JSON logs from a specified file path.
        
        :param self: Description
        :param json_file_path: Description
        :type json_file_path: str
        """
        _hash_alg = hash_alg if self.hash_alg is None else self.hash_alg
        default_json_dir = self.file_io.select_dir(filetype="json", length=self.length)
        json_path: Path = Path(self.json_path_arg) if self.json_path_arg else default_json_dir

        json_list = self.file_io.get_latest_files_by_date(json_path, _hash_alg, self.length)
        json_list.sort()
        return json_list


    def json_data_loader(self, json_file_path: str):
        """
        Parses JSON data from a specified file path.
        
        :param self: Description
        :param json_file_path: Description
        :type json_file_path: str
        """
        json_path_list = self.get_json_list(json_file_path)
        for json_file in json_path_list:
            json_data = self.file_io.file_reader(json_file, self.length)

        return json_data


    def log_parser(self, json_data: dict):
        """
        Parses log data from JSON content.
        
        :param self: Description
        :param json_data: Description
        :type json_data: dict
        """
        byte_string = json_data.get("BaseLogs", {}).get("message", "")
        return byte_string


    def byte_string_to_image(self, byte_string: str):
        """
        Converts a byte string to an image using the EMNIST dataset.
        
        :param self: Description
        :param byte_string: Description
        :type byte_string: str
        """


    def main(self, test: bool = False):
        """
        Main function to convert byte string to image.
        
        :param self: Description
        """
        if not test:
            default_outuput_dir = self.file_io.select_dir(filetype="image", \
                                                        length=self.length, data_mode="data")
            img_path: Path = Path(self.img_path_arg) if self.img_path_arg else default_outuput_dir
            print(img_path)
            self.emnist_save(img_path)

        latest_json_list = self.get_json_list(self.hash_alg)

        print(f"Found {len(latest_json_list)} JSON files.")

        for json_file in latest_json_list:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to XLSX")
    parser.add_argument("--json_path", type=str, help="Path to the input JSON file")
    parser.add_argument("--img_path", type=str, help="Path to the output image file")
    parser.add_argument("-v", "--is_verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--length", type=int, default=256, help="Length of the output text")
    parser.add_argument("--standalone", action="store_true", help="Run as standalone script")
    parser.add_argument("--hash_alg", type=str, required=True, \
                        help="Hash algorithm to filter JSON files")
    parser.add_argument("-t", "--test", action="store_true", help="Run in test mode")

    args = parser.parse_args()

    converter = ByteToImageConverter(
        standalone=args.standalone,
        json_path=args.json_path,
        img_path=args.img_path,
        length=args.length,
        is_verbose=args.is_verbose,
        hash_alg=args.hash_alg,
    )
    converter.main(args.test)
