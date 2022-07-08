from typing import Optional

import numpy as np


class Slices:
    """
    Slices class containing the slices functions. Each slice is assigned to a
    basestation.
    """

    def __init__(
        self,
        max_number_slices: int,
        max_number_ues: int,
        ue_assoc: np.ndarray,
        requirements: Optional[dict] = None,
    ) -> None:
        self.max_number_slices = max_number_slices
        self.max_number_ues = max_number_ues
        self.ue_assoc = ue_assoc  # Matrix of |Slices|x|UEs|
        self.requirements = requirements

    def update_assoc(
        self,
        ue_assoc: Optional[np.ndarray] = None,
    ) -> None:
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def update_slice_req(self, requirements: dict) -> None:
        self.requirements = requirements

    def get_number_ue_per_slice(self) -> np.ndarray:
        return np.sum(self.ue_assoc, axis=1)
