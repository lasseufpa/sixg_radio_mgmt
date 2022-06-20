import numpy as np


class Channel:
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: list,
    ) -> None:
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.num_available_rbs = num_available_rbs

    def step(self, mobilities: list) -> list:
        spectral_efficiencies = [
            np.ones((self.max_number_ues, self.num_available_rbs[i]))
            for i in np.arange(self.max_number_basestations)
        ]
        return spectral_efficiencies


def main():
    pass


if __name__ == "__main__":
    main()
