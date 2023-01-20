from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from .ues import UEs


class Association(ABC):
    """
    Associations abstract class to implement dynamic basestations, slices and UEs associations.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation
    max_number_basestations : int
        Maximum number of basestations in the simulation
    max_number_slices: int
        Maximum number of supported slices

    Methods
    -------
    step(self, step_number: int, episode_number: int)
        Generate 2D positions for each UE in the simulation
    """

    def __init__(
        self,
        ues: UEs,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        """
        Parameters
        ----------
        max_number_ues : int
            Maximum number of UEs in the simulation
        max_number_basestations : int
            Maximum number of basestations in the simulation
        max_number_slices: int
            Maximum number of supported slices
        """
        self.ues = ues
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.rng = rng

    @abstractmethod
    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        """Update Basestations, Slices and UEs associations

        Parameters
        ----------
        basestation_ue_assoc: np.ndarray
            Numpy array associating UEs to basestations with a form BxU,
            where represents the maximum number of UEs
        basestation_slice_assoc: np.ndarray
            Numpy array associating basestations to slices with a form BxS,
            where B is the maximum number of basestations and S is the
            maximum number of slices
        slice_ue_assoc: Optional[np.ndarray]
            UE association to slices
        slice_req: dict
            Dictionary contaning the slice requirements defined for each slice
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation

        Returns
        -------
        basestation_ue_assoc: np.ndarray
            New Numpy array associating UEs to basestations with a form BxU,
            where represents the maximum number of UEs
        basestation_slice_assoc: np.ndarray
            New numpy array associating basestations to slices with a form BxS,
            where B is the maximum number of basestations and S is the
            maximum number of slices
        slice_ue_assoc: Optional[np.ndarray]
            New UE association to slices to update the existent one
        slice_req: dict
            New dictionary contaning the slice requirements defined for each slice
        """
        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )
