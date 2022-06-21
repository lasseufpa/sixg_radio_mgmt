import numpy as np

from comm import Traffic


class SimpleTraffic(Traffic):
    def __init__(self, max_number_ues) -> None:
        super().__init__(max_number_ues)

    def step(self) -> list:
        return np.ones(self.max_number_ues) * 4
