import sys
from typing import Callable, Tuple
import argparse


def get_morph_driver(morph: str) -> Callable[[Tuple[int, str, str, str]], None]:
    if morph == "mipgan1":
        from morphs.mipgan1 import _driver

        sys.path.append("./morphs/mipgan1/")

        return _driver
    if morph == "mipgan2":
        from morphs.mipgan2 import _driver

        sys.path.append("./morphs/mipgan2/")

        return _driver


def main(args) -> None:
    process_num, src_dir, morph_list_csv, output_dir = (
        args.process_num,
        args.src_dir,
        args.morph_list_csv,
        args.output_dir,
    )
    _driver = get_morph_driver(args.morph)
    _driver(process_num, src_dir, morph_list_csv, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("process_num", type=int)
    parser.add_argument("src_dir", type=str)
    parser.add_argument("morph_list_csv", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("morph", type=str)
    args = parser.parse_args()
    main(args)
