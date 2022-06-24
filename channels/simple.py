import numpy as np
from numpy import linalg as la

from comm import Channel


class SimpleChannel(Channel):
    def __init__(
        self, max_number_ues: int, max_number_basestations: int, num_available_rbs: list
    ) -> None:
        pass
        super().__init__(max_number_ues, max_number_basestations, num_available_rbs)
        self.ula = SimpleChannel.ULA()
        self.target_spectral_efficiency = 5  # in bits/s/Hz
        self.power_tx = 10
        self.power_noise = 0.00929032  # Calculated for SE=5

    def step(self, step_number: int, episode_number: int, mobilities: list) -> list:
        capacities = []
        for ue in np.arange(2):
            h_channel = self.my_channel(
                mobilities[ue][0], mobilities[ue][0], self.ULA()
            )
            precoder_weight = h_channel
            capacity = self.capacity_miso_beamforming(
                h_channel, precoder_weight, self.power_tx, self.power_noise
            )
            capacities.append(capacity)
        # Considering only one basestation in self.num_available_rbs[0]
        spectral_efficiencies = [
            np.repeat(capacities[ue], self.num_available_rbs[0])
            for ue in np.arange(self.max_number_ues)
        ]

        # spectral_efficiencies = [
        #     np.ones((self.max_number_ues, self.num_available_rbs[i]))
        #     for i in np.arange(self.max_number_basestations)
        # ]
        return spectral_efficiencies

    """
    Very rough estimate of path loss (in this case, the gain)
    """

    def get_path_gain(self, distance):
        # Inspired on https://en.wikipedia.org/wiki/Free-space_path_loss
        return 1.0 / (distance**2)

    class ULA:  # starts at (0,0) and is aligned with y-axis
        def __init__(self, Na=18, normDistance=0.5, carrier_frequency=28e9):
            self.Na = Na
            c = 3e8  # light speed m/s
            self.lambda_c = c / carrier_frequency  # v=e/t
            self.normDistance = normDistance

    """
    rx_angle - angle with x-axis
    """

    def my_channel(self, rx_mag, rx_angle, ULA):
        d = rx_mag  # distance assuming ULA is in origin (0,0)
        a = self.get_path_gain(d)
        # assume we are in the first quadrant
        phi = 0.5 * np.pi - rx_angle  # angle with y-axis
        directional_cosine = np.cos(phi)  # Omega
        h_first_factor = a * np.exp(-1j * 2 * np.pi * d / ULA.lambda_c)
        exp_argument = -1j * 2 * np.pi * ULA.normDistance * directional_cosine
        h = h_first_factor * np.exp(np.kron(np.arange(0, ULA.Na), exp_argument))
        return h

    def estimate_noise_power(self, target_spectral_efficiency, h_channel, power_tx):
        channel_norm_squared = la.norm(h_channel) ** 2
        power_noise = (
            power_tx * channel_norm_squared / (2**target_spectral_efficiency - 1)
        )
        return power_noise

    """
    Spectral efficiency based on capacity for a given precoder weights vector.
    """

    def capacity_miso_beamforming(
        self, h_channel, precoder_weight, power_tx, power_noise
    ):
        inner_product = np.vdot(h_channel, precoder_weight)
        inner_product_magnitude = la.norm(inner_product)
        c = np.log2(1.0 + power_tx * inner_product_magnitude / power_noise)
        return c

    """
    Signal to noise ratio for a given precoder weights vector.
    """

    def snr_miso_beamforming(self, h_channel, precoder_weight, power_tx, power_noise):
        inner_product = np.vdot(h_channel, precoder_weight)
        inner_product_magnitude = la.norm(inner_product)
        snr = power_tx * inner_product_magnitude / power_noise
        return snr


def main():
    channel = SimpleChannel(2, 1, [2])
    mobilities = [[5, np.pi / 4], [5, np.pi / 4]]
    spectral_efficiences = channel.step(1, 1, mobilities)
    print(spectral_efficiences)
    # rx_mag = 5
    # rx_angle = np.pi / 4

    # Na = 18
    # normDistance = 0.5
    # carrier_frequency = 28e9  # 28 GHz
    # channel = SimpleChannel(2, 1, [2])
    # myULA = channel.ULA(Na, normDistance, carrier_frequency)

    # h_channel = channel.my_channel(rx_mag, rx_angle, myULA)
    # print("channel=", h_channel)
    # print("channel norm=", la.norm(h_channel))

    # power_tx = 10
    # power_noise = 1
    # precoder_weight = h_channel

    # capacity = channel.capacity_miso_beamforming(
    #     h_channel, precoder_weight, power_tx, power_noise
    # )
    # print("capacity=", capacity, "bits/s/Hz for noise power=", power_noise, "Watts")

    # target_spectral_efficiency = 5  # in bits/s/Hz
    # print("target_spectral_efficiency=", target_spectral_efficiency)
    # power_noise = channel.estimate_noise_power(
    #     target_spectral_efficiency, h_channel, power_tx
    # )
    # print("power_noise=", power_noise, "Watts")

    # capacity = channel.capacity_miso_beamforming(
    #     h_channel, precoder_weight, power_tx, power_noise
    # )
    # print(
    #     "spectra efficiency=",
    #     capacity,
    #     "bits/s/Hz for noise power=",
    #     power_noise,
    #     "Watts",
    # )

    # # Calibrate noise power to get given effiency
    # for d in range(1, 10, 1):
    #     rx_mag = d
    #     h_channel = channel.my_channel(rx_mag, rx_angle, myULA)
    #     se = channel.capacity_miso_beamforming(
    #         h_channel, precoder_weight, power_tx, power_noise
    #     )
    #     snr = channel.snr_miso_beamforming(
    #         h_channel, precoder_weight, power_tx, power_noise
    #     )
    #     print(
    #         "distance=",
    #         d,
    #         "spectra efficiency=",
    #         se,
    #         "bits/s/Hz, SNR=",
    #         10.0 * np.log10(snr),
    #         "dB",
    #     )


if __name__ == "__main__":
    main()
