from typing import Any, Callable

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
        MobilityClass: Mobility,
        config_file: str,
        action_format: Callable,
        obs_space_format: Callable[[dict], list] = None,
        calculate_reward: Callable[[dict], float] = None,
        obs_space: Callable = None,
        action_space: Callable = None,
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
        self.action_format = action_format
        self.ChannelClass = ChannelClass
        self.TrafficClass = TrafficClass
        self.MobilityClass = MobilityClass

        self.observation_space = obs_space() if obs_space is not None else obs_space
        self.action_space = action_space() if obs_space is not None else obs_space

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

        sched_decision = self.action_format(
            sched_decision,
            self.max_number_ues,
            self.max_number_basestations,
            self.num_available_rbs,
        )

        mobilities = self.mobility.step(self.step_number, self.episode_number)
        spectral_efficiencies = self.channel.step(
            self.step_number, self.episode_number, mobilities
        )
        traffics = self.traffic.step(self.step_number, self.episode_number)

        if self.verify_sched_decision(sched_decision):
            step_hist = self.ues.step(
                sched_decision,
                traffics,
                spectral_efficiencies,
                self.bandwidths,
                self.num_available_rbs,
            )
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
        reward = self.calculate_reward(step_hist)

        step_hist.update({"reward": reward})
        self.metrics_hist.step(step_hist)

        if self.step_number == self.max_number_steps:
            self.metrics_hist.save(self.simu_name, self.episode_number)

        return (
            obs,
            reward,
            self.step_number == self.max_number_steps,
            {},
        )

    def reset(self, initial_episode: int = -1) -> None:
        if (self.step_number == 0 and self.episode_number == 1) or (
            self.episode_number == self.max_number_episodes
        ):
            self.episode_number = 1 if initial_episode == -1 else initial_episode
        elif self.episode_number < self.max_number_episodes:
            self.episode_number += 1
        else:
            raise Exception(
                "Episode number received a non expected value equals to {}.".format(
                    self.episode_number
                )
            )
        self.step_number = 0

        self.create_scenario()
        initial_positions = self.mobility.step(self.step_number, self.episode_number)
        obs = {
            "mobility": initial_positions,
            "spectral_efficiencies": self.channel.step(
                self.step_number, self.episode_number, initial_positions
            ),
            "basestation_ue_assoc": self.basestation_ue_assoc,
            "basestation_slice_assoc": self.basestation_slice_assoc,
            "slice_ue_assoc": self.slice_ue_assoc,
            "sched_decision": [],
            "pkt_incoming": self.traffic.step(self.step_number, self.episode_number),
            "pkt_throughputs": np.zeros(self.max_number_ues),
            "pkt_effective_thr": np.zeros(self.max_number_ues),
            "buffer_occupancies": np.zeros(self.max_number_ues),
            "buffer_latencies": np.zeros(self.max_number_ues),
            "dropped_pkts": np.zeros(self.max_number_ues),
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
        self.mobility = self.MobilityClass(self.max_number_ues, self.episode_number)
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
                comm_env.reset()


if __name__ == "__main__":
    main()
