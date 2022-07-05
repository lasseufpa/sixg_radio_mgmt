from abc import ABC, abstractmethod

import numpy as np


class Traffic(ABC):
    def __init__(self, max_number_ues) -> None:
        self.max_number_ues = max_number_ues

    @abstractmethod
    def step(self, step_number: int, episode_number: int) -> np.array:
        pass


def main():
    pass


if __name__ == "__main__":
    main()
