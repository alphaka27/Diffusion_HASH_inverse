"""
Byte string to image conversion utilities with EMNIST dataset.
Input: Byte string in hex representation
Output: Image
"""
from pathlib import Path
import argparse
from functools import partial
from typing import List, Dict

from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torchvision
import torch

from diffusion_hash_inv.common import Logs
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

        self.hash_alg = kwargs.pop("hash_alg", None)

        transform = transform = transforms.Compose([
            partial(F.rotate, angle=-90),
            F.hflip,
            transforms.ToTensor(),
        ])
        self.dataset = datasets.EMNIST(
            root=ROOT_DIR / "data",
            split="byclass",
            train=True,
            download=True,
            transform=transform,
        )

    def get_json_list(self, hash_alg: str, json_path: Path = None) -> List[Path]:
        """
        Get list of JSON files for a specific hash algorithm.

        :param hash_alg: Specified hash algorithm
        :type hash_alg: str
        :return: Returns list of JSON file paths (full paths)
        :rtype: List[Path]
        """
        _hash_alg = hash_alg if self.hash_alg is None else self.hash_alg

        json_list = self.file_io.get_latest_files_by_date(json_path, _hash_alg, self.length)
        json_list.sort()
        return json_list

    def subdataset_by_labels(self, target_labels: int | str):
        """
        Creates a sub-dataset filtered by target labels.

        :param target_labels: Target label ids to include
        :type target_labels: int | str
        :return: Filtered sub-dataset
        :rtype: Dataset
        """
        if isinstance(target_labels, int):
            target_labels = [target_labels]
        for img, label in self.dataset:
            if label in target_labels:
                return img
        return None


    def byte_string_to_image(self, byte_string: str):
        """
        Converts a byte string to an image using the EMNIST dataset.
        
        :param self: Description
        :param byte_string: Description
        :type byte_string: str
        """
        #['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        # 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        # 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        # 'U', 'V', 'W', 'X', 'Y', 'Z',
        # 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        # 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        # 'u', 'v', 'w', 'x', 'y', 'z']

        byte_values = [int(byte, 16) for byte in byte_string]
        print(f"Byte values: {byte_values}")

        images = []
        for byte in byte_values:
            if '0' <= byte_string <= '9':
                label = byte
            elif 'A' <= byte_string <= 'Z':
                label = byte
            elif 'a' <= byte_string <= 'z':
                label = byte.upper()
            else:
                raise ValueError(f"Invalid byte character: {byte_string}")
            image = self.subdataset_by_labels(label)
            assert image is not None, f"No image found for label: {label}"
            images.append(image)

        grid_image = F.to_pil_image(torchvision.utils.make_grid(images, nrow=16))
        return grid_image


    def save_image(self, image, img_path: Path):
        """
        Saves the image to the specified path.
        
        :param self: Description
        :param image: Description
        :param img_path: Description
        :type img_path: Path
        """
        self.file_io.file_writer(img_path, image, length=self.length)



    def json_loader(self, json_file: Path) -> dict:
        """
        Loads JSON logs from a specified file path.

        :param json_file: Description
        :type json_file: Path
        """
        json_data = self.file_io.file_reader(json_file, length=self.length)
        return json_data

    def _dict_parser(self, data: Dict):
        """
        Parses dictionary data.
        
        :param self: Description
        :param data: Description
        :type data: Dict
        """
        ret = []
        for entry in data:

            if isinstance(data[entry], str):
                print(f"String data: {data[entry]}")
                ret.append(Logs.str_strip(data[entry]))

            elif isinstance(data[entry], Dict):
                print(f"Dict data: {data[entry]}")
                ret.append(self._dict_parser(data[entry]))

            elif isinstance(data[entry], List):
                print(f"List data: {data[entry]}")
                for item in data[entry]:
                    _item = Logs.str_strip(item)
                    ret.append(_item)

            else:
                raise TypeError(f"Unsupported data type: {type(data[entry])}")

        assert ret is not None, "Parsed data is None"
        return ret

    def list_to_string(self, data: List) -> str:
        """
        Converts a list of items to a concatenated string.
        
        :param self: Description
        :param data: Description
        :type data: List
        """

        _list_data = [str(item) for item in data]
        _str_data = "".join(_list_data)
        _str_data = _str_data.replace(" ", "").replace("[", "")\
            .replace("]", "").replace(",", "").replace("'", "")
        return _str_data

    def log_parser(self, json_data: dict, img_path: Path):
        """
        Parses log data from JSON content.
        
        :param self: Description
        :param json_data: Description
        :type json_data: dict
        """
        _json_data = json_data.get("Logs")
        temp_data = self._dict_parser(_json_data)

        for entry in _json_data:
            print(f"Entry: {entry}")
            temp_data = []

            if isinstance(_json_data[entry], Dict):
                temp_data += self._dict_parser(_json_data[entry])
                _list_data = [str(item) for item in temp_data]
                print(f"Convert data: {_list_data}")
                _str_data = self.list_to_string(_list_data)
                print(f"String data: {_str_data}")
                byte_image = self.byte_string_to_image(_str_data)
                self.save_image(byte_image, img_path / f"{entry}.png")

            else:
                print(f"Data: {_json_data[entry]}\n")

            print(f"Temp data: {temp_data}")

    def main(self, img_path: Path = None, json_path: Path = None):
        """
        Main function to convert byte string to image.
        
        :param self: Description
        """

        _img_path: Path = Path(img_path) if img_path \
            else self.file_io.select_dir(filetype="image",\
                                            length=self.length)
        print(_img_path)

        json_path: Path = Path(json_path) if json_path \
            else self.file_io.select_dir(filetype="json",\
                                            length=self.length)

        latest_json_list = self.get_json_list(self.hash_alg, json_path=json_path)

        print(f"Found {len(latest_json_list)} JSON files.")

        for file_path in latest_json_list:
            print(f"filename: {file_path.stem}")
            json_data = self.json_loader(file_path)
            self.log_parser(json_data, _img_path / file_path.stem)
            print(f"Processing file: {file_path}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to XLSX")
    parser.add_argument("--json_path", type=str, help="Path to the input JSON file")
    parser.add_argument("--img_path", type=str, help="Path to the output image file")
    parser.add_argument("-v", "--is_verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--length", type=int, default=256, help="Length of the output text")
    parser.add_argument("--standalone", action="store_true", help="Run as standalone script")
    parser.add_argument("--hash_alg", type=str, required=True, \
                        help="Hash algorithm to filter JSON files")
    parser.add_argument("--target_labels", type=int, nargs="+", \
                        help="Explicit label ids to include (e.g., 0 1 2)")
    parser.add_argument("--target_label_range", type=int, nargs=2, metavar=("START", "END"), \
                        help="Inclusive label id range to include (e.g., 10 20)")
    parser.add_argument("-t", "--test", action="store_true", help="Run in test mode")

    args = parser.parse_args()

    converter = ByteToImageConverter(
        standalone=args.standalone,
        length=args.length,
        is_verbose=args.is_verbose,
        hash_alg=args.hash_alg,
        target_labels=args.target_labels,
        target_label_range=args.target_label_range,
    )
    converter.main(img_path=args.img_path, json_path=args.json_path)
