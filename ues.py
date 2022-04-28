import os
from typing import Tuple

import numpy as np

from buffer import Buffer


class UEs:
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

    def update_ues(
        self,
        ue_indexes: np.array,
        max_buffer_latencies: np.array,
        max_buffer_pkts: np.array,
        pkt_sizes: np.array,
    ) -> None:
        self.max_buffer_latencies[ue_indexes] = max_buffer_latencies
        self.max_buffer_pkts[ue_indexes] = max_buffer_pkts
        self.pkt_sizes[ue_indexes] = pkt_sizes
        for ue_index in ue_indexes:
            self.buffers[ue_index] = Buffer(
                max_buffer_pkts[ue_index], max_buffer_latencies[ue_index]
            )

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
            "dropped_pkts": [buffer.dropped_packets for buffer in self.buffers],
        }


def main():
    ues = UEs(
        max_number_ues=2,
        max_buffer_latencies=[10, 10],
        max_buffer_pkts=[20, 10],
        pkt_sizes=[1, 1],
    )
    sched_decision = np.array([[[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]]])
    traffics = np.array([4, 4])
    spectral_efficiences = np.array([[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]])
    steps = 10
    for i in np.arange(steps):
        info = ues.step(sched_decision, traffics, spectral_efficiences)
        print("Step {}:\n{}".format(i, info))


if __name__ == "__main__":
    main()
