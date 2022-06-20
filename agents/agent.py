from abc import ABC, abstractmethod
from os import stat

import numpy as np


class Agent(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def step(self, obs_space: list) -> list:
        pass

    @staticmethod
    @abstractmethod
    def obs_space_format(obs_space: dict) -> list:
        pass

    @staticmethod
    @abstractmethod
    def calculate_reward(obs_space: dict) -> np.float:
        pass
