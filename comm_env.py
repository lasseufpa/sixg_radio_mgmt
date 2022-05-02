import numpy as np

from basestation import Basestation
from basestations import Basestations
from channel import Channel
from mobility import Mobility
from slices import Slices
from traffic import Traffic
from ues import UEs


class CommunicationEnv:
    def __init__(self) -> None:
        self.max_number_basestations = 1
        self.max_number_slices = 1
        self.max_number_ues = 2
        self.max_buffer_latencies = [10, 10]
        self.max_buffer_pkts = [20, 10]
        self.bandwidths = [100]  # In MHz
        self.carrier_frequencies = [28]  # In GHz
        self.pkt_sizes = [1]  # In bits
        self.basestation_slice_assoc = np.array([[1]])
        self.slice_ue_association = np.array([[1, 1]])
        self.num_rbs_available = np.array([10])

        self.step = 0  # Initial simulation step
        self.episode = 1  # Initial episode
        self.max_number_steps = 2000  # Maximum number of steps per simulated episode
        self.max_number_episodes = 1  # Maximum number of simulated episodes

        self.ues = UEs(
            self.max_number_ues,
            self.max_buffer_latencies,
            self.max_buffer_pkts,
            self.pkt_sizes,
        )
        self.slices = Slices(
            self.max_number_slices, self.max_number_ues, self.slice_ue_association
        )
        self.basestations = Basestations(
            self.max_number_basestations,
            self.max_number_slices,
            self.basestation_slice_assoc,
            self.bandwidths,
            self.carrier_frequencies,
            self.num_rbs_available,
        )
        self.mobilities = Mobility()
        self.channels = Channel()
        self.traffics = Traffic()

    def step(self, sched_decision):
        # sched_decisions is a matrix with dimensions BxNxM where B represents the number of basestations, N represents the maximum number of UEs and M the maximum number of RBs. For instance [[[1,1,0], [0,0,1]], [[0,0,1], [1,1,0]]] means that in the basestation 1, the UE 1 received the RBs 1 and 2 allocated while the second UE received the RB 3. For basestation 2, the UE 1 received the RB 3, and UE 2 got RBs 1 and 2. Remember that N and M value varies in according to the basestation configuration.

        # if add_new_slice or add_new_ue:
        #     pass  # TODO

        mobilities = self.mobility.step()
        spectral_efficiencies = self.channel.step(mobilities)
        traffics = self.traffic.step()

        if self.verify_sched_decision():
            self.ues.step(sched_decision, traffics, spectral_efficiencies)


def main():
    pass


if __name__ == "__main__":
    main()
