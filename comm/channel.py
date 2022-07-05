from abc import ABC, abstractmethod

import numpy as np


class Channel(ABC):
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
    def step(self, step_number: int, episode_number: int, mobilities: np.array) -> list:
        pass


def main():
    pass


if __name__ == "__main__":
    main()
