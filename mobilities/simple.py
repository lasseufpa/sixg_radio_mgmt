import numpy as np

from comm.mobility import Mobility


class SimpleMobility(Mobility):
    def __init__(self, max_number_ues: int) -> None:
        super().__init__(max_number_ues)

    def step(self) -> list:
        return np.ones(self.max_number_ues)
