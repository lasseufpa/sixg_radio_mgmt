from typing import Optional

import numpy as np


class Basestations:
    def __init__(
        self,
        max_number_basestations: int,
        max_number_slices: int,
        slice_assoc: np.ndarray,
        ue_assoc: np.ndarray,
        bandwidths: np.ndarray,
        carrier_frequencies: np.ndarray,
        num_available_rbs: np.ndarray,
    ) -> None:
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.slice_assoc = slice_assoc  # Matrix with dimensions |basestations|x|slices|
        self.ue_assoc = ue_assoc
        self.bandwidths = bandwidths
        self.carrier_frequencies = carrier_frequencies
        self.num_available_rbs = num_available_rbs

    def get_assoc(self) -> np.ndarray:
        return self.slice_assoc

    def update_assoc(
        self,
        slice_assoc: Optional[np.ndarray] = None,
        ue_assoc: Optional[np.ndarray] = None,
    ) -> None:
        self.slice_assoc = slice_assoc if slice_assoc is not None else self.slice_assoc
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def get_number_slices_per_basestation(self) -> np.ndarray:
        return np.sum(self.slice_assoc, axis=1)
