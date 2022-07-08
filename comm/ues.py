import numpy as np

from comm import Buffer


class UEs:
    """
    Class containing the UEs functions. Each UE have a buffer and Channel values
    for specific trials. Each UE will be assigned to a slice.
    """

    def __init__(
        self,
        max_number_ues: int,
        max_buffer_latencies: np.ndarray,
        max_buffer_pkts: np.ndarray,
        pkt_sizes: np.ndarray,
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
        sched_decision: np.ndarray,
        spectral_efficiencies: np.ndarray,
        bandwidth: float,
        num_available_rbs: int,
        pkt_sizes: np.ndarray,
    ) -> list:
        return np.floor(
            np.sum(
                (bandwidth / num_available_rbs)
                * np.array(sched_decision)
                * np.array(spectral_efficiencies),
                axis=1,
            )
            / pkt_sizes
        )

    def update_ues(
        self,
        ue_indexes: np.ndarray,
        max_buffer_latencies: np.ndarray,
        max_buffer_pkts: np.ndarray,
        pkt_sizes: np.ndarray,
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
        sched_decision: np.ndarray,
        traffics: np.ndarray,
        spectral_efficiencies: np.ndarray,
        bandwidths: np.ndarray,
        num_available_rbs: np.ndarray,
    ) -> dict:
        pkt_throughputs = np.zeros(self.max_number_ues)
        for basestation in np.arange(len(sched_decision)):
            pkt_throughputs += self.get_pkt_throughputs(
                sched_decision[basestation],
                spectral_efficiencies[basestation],
                bandwidths[basestation],
                num_available_rbs[basestation],
                self.pkt_sizes,
            )
        pkt_incomings = np.floor(traffics / self.pkt_sizes)

        for i in np.arange(self.max_number_ues):
            self.buffers[i].receive_packets(pkt_incomings[i])
            self.buffers[i].send_packets(pkt_throughputs[i])
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
