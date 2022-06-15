from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def step(self, obs_space: np.array) -> np.array:
        pass
