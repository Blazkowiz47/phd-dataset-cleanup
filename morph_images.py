from typing import Callable, Tuple


def get_morph_driver(morph: str) -> Callable[[Tuple[int, str, str, str]], None]:
    if morph == "ubo":
        from morphs.lmaubo import driver

        return driver
    if morph == "mipgan1":
        from morphs.mipgan1 import driver

        return driver
    if morph == "mipgan2":
        from morphs.mipgan2 import driver

        return driver
    if morph == "pipe":
        from morphs.pipe import driver

        return driver
    if morph == "mordiff":
        from morphs.mordiff import driver

        return driver
    if morph == "greedy":
        from morphs.greedy import driver

        return driver

    raise NotImplementedError(f"Morpher {morph} not implemented")


def perform_morphing(
    morph: str, src_dir: str, morph_list_csv: str, output_dir: str
) -> None:
    morpher = get_morph_driver(morph)
    morpher((0, src_dir, morph_list_csv, output_dir))


def main() -> None:
    perform_morphing("mipgan1", "", "", "")
