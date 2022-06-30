import numpy as np

from comm import Traffic


class SimpleTraffic(Traffic):
    def __init__(self, max_number_ues) -> None:
        super().__init__(max_number_ues)

    def step(self, step_number: int, episode_number: int) -> list:
        return np.ones(self.max_number_ues) * 2
