import numpy as np

from comm.mobility import Mobility


class SimpleMobility(Mobility):
    def __init__(self, max_number_ues: int) -> None:
        super().__init__(max_number_ues)

    def step(self, step_number: int, episode_number: int) -> np.array:
        return np.ones((self.max_number_ues, 2))
