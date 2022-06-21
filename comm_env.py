from typing import Callable

import gym
import numpy as np
import yaml
from tqdm import tqdm

from comm import Basestations, Channel, Metrics, Mobility, Slices, Traffic, UEs


class CommunicationEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ChannelClass: Channel,
        TrafficClass: Traffic,
        config_file: str = "simple",
        obs_space_format: Callable[[dict], list] = None,
        calculate_reward: Callable[[dict], float] = None,
    ) -> None:

        with open("./env_config/{}.yml".format(config_file)) as file:
            data = yaml.safe_load(file)

        self.max_number_basestations = data["basestations"]["max_number_basestations"]
        self.bandwidths = data["basestations"]["bandwidths"]  # In MHz
        self.carrier_frequencies = data["basestations"]["carrier_frequencies"]  # In GHz
        self.num_available_rbs = data["basestations"]["num_available_rbs"]
        self.basestation_ue_assoc = data["basestations"]["basestation_ue_assoc"]
        self.basestation_slice_assoc = data["basestations"]["basestation_slice_assoc"]

        self.max_number_slices = data["slices"]["max_number_slices"]
        self.slice_ue_assoc = data["slices"]["slice_ue_assoc"]

        self.max_number_ues = data["ues"]["max_number_ues"]
        self.max_buffer_latencies = data["ues"]["max_buffer_latencies"]
        self.max_buffer_pkts = data["ues"]["max_buffer_pkts"]
        self.pkt_sizes = data["ues"]["pkt_sizes"]  # In bits

        self.step_number = 0  # Initial simulation step
        self.episode_number = 1  # Initial episode
        self.max_number_steps = data["simulation"][
            "max_number_steps"
        ]  # Maximum number of steps per simulated episode
        self.max_number_episodes = data["simulation"][
            "max_number_episodes"
        ]  # Maximum number of simulated episodes
        self.hist_root_path = data["simulation"]["hist_root_path"]
        self.simu_name = data["simulation"]["simu_name"]

        self.obs_space_format = (
            obs_space_format
            if obs_space_format is not None
            else self.obs_space_format_default
        )
        self.calculate_reward = (
            calculate_reward
            if calculate_reward is not None
            else self.calculate_reward_default
        )
        self.ChannelClass = ChannelClass
        self.TrafficClass = TrafficClass

        self.create_scenario()

    def step(self, sched_decision: list) -> None:
        """
        sched_decisions is a matrix with dimensions BxNxM where B represents
        the number of basestations, N represents the maximum number of UEs
        and M the maximum number of RBs. For instance
        [[[1,1,0], [0,0,1]], [[0,0,1], [1,1,0]]] means that in the basestation
        1, the UE 1 received the RBs 1 and 2 allocated while the second UE
        received the RB 3. For basestation 2, the UE 1 received the RB 3, and
        UE 2 got RBs 1 and 2. Remember that N and M value varies in according
        to the basestation configuration.
        """

        mobilities = self.mobility.step()
        spectral_efficiencies = self.channel.step(mobilities)
        traffics = self.traffic.step()

        if self.verify_sched_decision(sched_decision):
            step_hist = self.ues.step(sched_decision, traffics, spectral_efficiencies)
        step_hist.update(
            {
                "mobility": mobilities,
                "spectral_efficiencies": spectral_efficiencies,
                "basestation_ue_assoc": self.basestation_ue_assoc,
                "basestation_slice_assoc": self.basestation_slice_assoc,
                "slice_ue_assoc": self.slice_ue_assoc,
                "sched_decision": sched_decision,
            }
        )
        self.step_number += 1
        obs = self.obs_space_format(step_hist)
        reward = self.calculate_reward(obs)

        step_hist.update({"reward": reward})
        self.metrics_hist.step(step_hist)

        return (
            obs,
            reward,
            self.step_number == self.max_number_steps,
            {},
        )

    def reset(self) -> None:
        self.create_scenario()
        obs = {
            "basestation_ue_assoc": self.basestation_ue_assoc,
            "basestation_slice_assoc": self.basestation_slice_assoc,
            "slice_ue_assoc": self.slice_ue_assoc,
        }

        return self.obs_space_format(obs)

    @staticmethod
    def calculate_reward_default(obs_space: dict) -> float:
        return 0

    @staticmethod
    def obs_space_format_default(obs_space: dict) -> list:
        return obs_space

    @staticmethod
    def verify_sched_decision(sched_decision: list) -> bool:
        for basestation_sched in sched_decision:
            if np.sum(np.sum(basestation_sched, axis=0) > 1) > 0:
                raise Exception(
                    "Scheduling decision allocated the same RB for more than one UE"
                )
        return True

    def create_scenario(self) -> None:
        self.ues = UEs(
            self.max_number_ues,
            self.max_buffer_latencies,
            self.max_buffer_pkts,
            self.pkt_sizes,
        )
        self.slices = Slices(
            self.max_number_slices, self.max_number_ues, self.slice_ue_assoc
        )
        self.basestations = Basestations(
            self.max_number_basestations,
            self.max_number_slices,
            self.basestation_slice_assoc,
            self.basestation_ue_assoc,
            self.bandwidths,
            self.carrier_frequencies,
            self.num_available_rbs,
        )
        self.mobility = Mobility(self.max_number_ues)
        self.channel = self.ChannelClass(
            self.max_number_ues,
            self.max_number_basestations,
            self.num_available_rbs,
        )
        self.traffic = self.TrafficClass(self.max_number_ues)
        self.metrics_hist = Metrics(self.hist_root_path)


def main():
    comm_env = CommunicationEnv()
    sched_decision = [[[1, 0], [0, 0]], [[0, 0], [1, 0]]]
    comm_env.reset()
    for episode in np.arange(1, comm_env.max_number_episodes + 1):
        print("Episode ", episode)
        for step_number in tqdm(np.arange(comm_env.max_number_steps)):
            comm_env.step(sched_decision)
            if step_number == comm_env.max_number_steps - 1:
                comm_env.metrics_hist.save(comm_env.simu_name, comm_env.episode_number)
                comm_env.reset()


if __name__ == "__main__":
    main()
