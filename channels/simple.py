import numpy as np

from comm import Channel


class SimpleChannel(Channel):
    def __init__(
        self, max_number_ues: int, max_number_basestations: int, num_available_rbs: list
    ) -> None:
        super().__init__(max_number_ues, max_number_basestations, num_available_rbs)

    def step(self, mobilities: list) -> list:
        spectral_efficiencies = [
            np.ones((self.max_number_ues, self.num_available_rbs[i]))
            for i in np.arange(self.max_number_basestations)
        ]
        return spectral_efficiencies
