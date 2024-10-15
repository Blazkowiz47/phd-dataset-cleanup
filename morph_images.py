import sys
from typing import Callable, Tuple


def get_morph_driver(morph: str) -> Callable[[Tuple[int, str, str, str]], None]:
    if morph == "ubo":
        from morphs.lmaubo import driver

        sys.path.append("./morphs/lmaubo/")

        return driver
    if morph == "mipgan1":
        from morphs.mipgan1 import driver

        sys.path.append("./morphs/mipgan1/")

        return driver
    if morph == "mipgan2":
        from morphs.mipgan2 import driver

        sys.path.append("./morphs/mipgan2/")

        return driver
    if morph == "pipe":
        from morphs.pipe import driver

        sys.path.append("./morphs/pipe/")

        return driver
    if morph == "mordiff":
        from morphs.mordiff import driver

        sys.path.append("./morphs/mordiff/")

        return driver
    if morph == "greedy":
        from morphs.greedy import driver

        sys.path.append("./morphs/greedy/")

        return driver

    raise NotImplementedError(f"Morpher {morph} not implemented")


def perform_morphing(
    morph: str, src_dir: str, morph_list_csv: str, output_dir: str
) -> None:
    morpher = get_morph_driver(morph)
    morpher((0, src_dir, morph_list_csv, output_dir))


def main() -> None:
    perform_morphing(
        "mipgan1",
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/aligned/test/",
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/test_index.csv",
        "./test_morphs/mipgan1",
    )


#     perform_morphing(
#         "mipgan2",
#         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/aligned/test/",
#         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/test_index.csv",
#         "./test_morphs/mipgan2",
#     )
#     perform_morphing(
#         "pipe",
#         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/aligned/test/",
#         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/test_index.csv",
#         "./test_morphs/pipe",
#     )
if __name__ == "__main__":
    main()
