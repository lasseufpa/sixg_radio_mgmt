import numpy as np
from agent import Agent
from matplotlib.style import available


class RoundRobin(Agent):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.array,
    ) -> None:
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.num_available_rbs = num_available_rbs

    def step(self, obs_space: np.array, basestation_ue_assoc: np.array) -> list:
        allocation_rbs = [
            np.zeros((self.max_number_ues, self.num_available_rbs[basestation]))
            for basestation in np.arange(self.max_number_basestations)
        ]
        for basestation in np.arange(self.max_number_basestations):
            ue_idx = 0
            rb_idx = 0
            while rb_idx < self.num_available_rbs[basestation]:
                if basestation_ue_assoc[basestation, ue_idx] == 1:
                    allocation_rbs[basestation][ue_idx][rb_idx] += 1
                    rb_idx += 1
                    ue_idx += 1 if ue_idx + 1 != self.max_number_ues else -ue_idx

        return allocation_rbs


def main():
    rr = RoundRobin(2, 2, [3, 2])
    basestation_ue_assoc = np.array([[1, 1], [1, 1]])
    for i in range(1):
        print(rr.step([], basestation_ue_assoc))


if __name__ == "__main__":
    main()
