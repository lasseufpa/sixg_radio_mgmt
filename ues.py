import os
from typing import Tuple

import numpy as np

from buffer import Buffer


class UES:
    """
    Class containing the UEs functions. Each UE have a buffer and Channel values
    for specific trials. Each UE will be assigned to a slice.
    """

    def __init__(
        self,
        max_number_ues: int,
        max_buffer_latencies: np.array(dtype=int),
        max_buffer_pkts: np.array(dtype=int),
        pkt_sizes: np.array(dtype=int),
    ) -> None:
        self.max_number_ues = max_number_ues
        self.max_buffer_latencies = max_buffer_latencies
        self.max_buffer_pkts = max_buffer_pkts
        self.pkt_sizes = pkt_sizes
        buffers = [
            Buffer(max_buffer_pkts[i], max_buffer_latencies[i])
            for i in np.arange(max_number_ues)
        ]

    @staticmethod
    def get_pkt_throughputs(sched_decision, spectral_efficiencies, pkt_sizes):
        return np.floor(
            np.sum(sched_decision * spectral_efficiencies, axis=2) / pkt_sizes
        )  # TODO Check because of 3D array

    def step(self, sched_decision, traffics, spectral_efficiencies) -> None:
        pkt_throughputs = self.get_pkt_throughputs(
            sched_decision, spectral_efficiencies, self.pkt_sizes
        )
        pkts_incoming = traffics

        for i in np.arange(self.max_number_ues):
            pass


def main():
    pass


if __name__ == "__main__":
    main()
