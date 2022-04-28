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
        max_buffer_latencies: np.array,
        max_buffer_pkts: np.array,
        pkt_sizes: np.array,
    ) -> None:
        self.max_number_ues = max_number_ues
        self.max_buffer_latencies = max_buffer_latencies
        self.max_buffer_pkts = max_buffer_pkts
        self.pkt_sizes = pkt_sizes
        self.buffers = [
            Buffer(max_buffer_pkts[i], max_buffer_latencies[i])
            for i in np.arange(max_number_ues)
        ]

    @staticmethod
    def get_pkt_throughputs(
        basestation_decision: np.array,
        spectral_efficiencies: np.array,
        pkt_sizes: np.array,
    ) -> np.array:
        return np.floor(
            np.sum(basestation_decision * spectral_efficiencies, axis=1) / pkt_sizes
        )

    def add_ue(ue_indexes: np.array) -> None:
        pass

    def remove_ue():
        pass

    def step(
        self,
        sched_decision: np.array,
        traffics: np.array,
        spectral_efficiencies: np.array,
    ) -> dict:
        for i, basestation_decision in enumerate(sched_decision):
            pkt_throughputs = self.get_pkt_throughputs(
                basestation_decision, spectral_efficiencies[i], self.pkt_sizes
            )
            pkt_incomings = np.floor(traffics / self.pkt_sizes)

            for j in np.arange(self.max_number_ues):
                self.buffers[j].receive_packets(pkt_incomings[j])
                self.buffers[j].send_packets(pkt_throughputs[j])
        return {
            "pkt_incoming": pkt_incomings,
            "pkt_throughputs": pkt_throughputs,
            "pkt_effective_thr": [buffer.sent_packets for buffer in self.buffers],
            "buffer_occupancies": [
                buffer.get_buffer_occupancy() for buffer in self.buffers
            ],
            "buffer_latencies": [buffer.get_avg_delay() for buffer in self.buffers],
            "dropped_pkts": [buffer.dropped_packets() for buffer in self.buffers],
        }


def main():
    ues = UES(2, [10, 10], [20, 20], [1, 1])
    sched_decision = np.array([[[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]]])
    traffics = np.array([2, 2])
    sprectral_efficiences = np.array([[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]])
    ues.step(sched_decision, traffics, sprectral_efficiences)


if __name__ == "__main__":
    main()
