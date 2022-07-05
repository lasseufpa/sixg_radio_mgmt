from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.array,
    ) -> None:
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.num_available_rbs = num_available_rbs

    @abstractmethod
    def step(self, obs_space: np.array) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def obs_space_format(obs_space: dict) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def calculate_reward(obs_space: dict) -> np.float:
        pass

    @staticmethod
    def action_format(
        action: np.array,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.array,
    ) -> list:
        return action.tolist()
