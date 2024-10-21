from typing import Callable, Optional, Tuple

from numpy.typing import NDArray
from torch.nn import Module


def get_frs_initializers(
    backbone: str,
) -> Tuple[
    Callable[
        [Optional[str]],
        Module,
    ],
    Callable[[str, Module], NDArray],
]:
    if backbone == "arcface":
        from .arcface import get_features, get_model

        return get_model, get_features

    if backbone == "magface":
        from .magface import get_features, get_model

        return get_model, get_features

    if backbone == "adaface":
        from .adaface import get_features, get_model

        return get_model, get_features
    raise NotImplementedError(f"FRS: {backbone} NOT IMPLEMENTED")
