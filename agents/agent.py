from abc import ABC, abstractmethod
from os import stat

import numpy as np


class Agent(ABC):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: list,
    ) -> None:
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.num_available_rbs = num_available_rbs

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
