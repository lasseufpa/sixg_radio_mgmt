class Simulation:
    def __init__(self) -> None:
        number_basestations = 1
        max_number_slices_per_bs = [3]
        max_number_ues = 3
        max_number_ues_per_slice = [1]
        max_buffer_latencies = [1, 2, 3]
        max_buffer_pkts = [1, 2, 3]
        bandwidths = [100]  # In MHz
        carrier_frequencies = [28]  # In GHz
        pkt_sizes = [10]  # In bits
        step = 0  # Initial simulation step
        episode = 1  # Initial episode
        max_number_steps = 2000  # Maximum number of steps per simulated episode
        max_number_episodes = 1  # Maximum number of simulated episodes

        ues = UEs(max_buffer_latencies, max_buffer_pkts, pkt_sizes)
        slices = Slices(max_number_ues_per_slice, ues.ues_indexes)
        basestations = Basestations(
            number_basestations,
            bandwidths,
            carrier_frequencies,
            max_number_slices_per_bs,
            max_number_ues,
            slices.slices_indexes,
        )
        mobility = Mobility()
        channel = Channel()
        traffic = Traffic()

    def step(self, sched_decision):
        # allocation decisions is a matrix with dimensions BxNxM where B represents the number of basestations, N represents the maximum number of UEs and M the maximum number of RBs. For instance [[[1,1,0], [0,0,1]], [[0,0,1], [1,1,0]]] means that in the basestation 1, the UE 1 received the RBs 1 and 2 allocated while the second UE received the RB 3. For basestation 2, the UE 1 received the RB 3, and UE 2 got RBs 1 and 2. Remember that N and M value varies in according to the basestation configuration.
        if add_new_slice or add_new_ue:
            pass  # TODO

        mobilities = self.mobility.step()
        spectral_efficiencies = self.channel.step(mobilities)
        traffics = self.traffic.step()

        if self.verify_sched_decision():
            self.ues.step(sched_decision, traffics, spectral_efficiencies)
