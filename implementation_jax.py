from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class WeightDefinition:
    components: int
    fourier_kernel: Callable[[float, "Grid"], Array]
