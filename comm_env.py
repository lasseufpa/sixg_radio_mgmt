from typing import Any, Callable, Optional, Tuple, Type

import gym
import numpy as np
import yaml

from comm import Basestations, Channel, Metrics, Mobility, Slices, Traffic, UEs


class CommunicationEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ChannelClass: Type[Channel],
        TrafficClass: Type[Traffic],
        MobilityClass: Type[Mobility],
        config_file: str,
        action_format: Callable[[np.ndarray, int, int, np.ndarray], np.ndarray],
        obs_space_format: Optional[Callable[[dict], Any]] = None,
        calculate_reward: Optional[Callable[[dict], float]] = None,
        obs_space: Optional[Callable] = None,
        action_space: Optional[Callable] = None,
        debug: bool = True,
    ) -> None:

        with open("./env_config/{}.yml".format(config_file)) as file:
            data = yaml.safe_load(file)

        self.max_number_basestations = data["basestations"]["max_number_basestations"]
        self.bandwidths = np.array(data["basestations"]["bandwidths"])  # In MHz
        self.carrier_frequencies = np.array(
            data["basestations"]["carrier_frequencies"]
        )  # In GHz
        self.num_available_rbs = np.array(data["basestations"]["num_available_rbs"])
        self.init_basestation_ue_assoc = np.array(
            data["basestations"]["basestation_ue_assoc"]
        )
        self.init_basestation_slice_assoc = np.array(
            data["basestations"]["basestation_slice_assoc"]
        )

        self.max_number_slices = data["slices"]["max_number_slices"]
        self.init_slice_ue_assoc = np.array(data["slices"]["slice_ue_assoc"])

        self.max_number_ues = data["ues"]["max_number_ues"]
        self.max_buffer_latencies = np.array(data["ues"]["max_buffer_latencies"])
        self.max_buffer_pkts = np.array(data["ues"]["max_buffer_pkts"])
        self.pkt_sizes = np.array(data["ues"]["pkt_sizes"])  # In bits

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
        self.associations = data["associations"]
        self.mobility_size = 2  # X and Y axis
        self.debug = debug

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

        if obs_space is not None:
            self.observation_space = obs_space()
        if action_space is not None:
            self.action_space = action_space()

        self.create_scenario()

    def step(self, sched_decision: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
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

        if self.debug:
            self.check_env_agent(
                sched_decision,
                spectral_efficiencies,
                mobilities,
                traffics,
            )
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
                "basestation_ue_assoc": self.basestations.ue_assoc,
                "basestation_slice_assoc": self.basestations.slice_assoc,
                "slice_ue_assoc": self.slices.ue_assoc,
                "sched_decision": sched_decision,
                "slice_req": self.slices.requirements,
            }
        )
        self.step_number += 1
        obs = self.obs_space_format(step_hist)
        reward = self.calculate_reward(step_hist)

        step_hist.update({"reward": reward})
        self.metrics_hist.step(step_hist)

        if self.step_number in self.associations["timeline"]:
            idx_timeline = np.equal(
                self.step_number, self.associations["timeline"]
            ).nonzero()[0][0]
            self.slices.update_assoc(self.associations["slice_ue_assoc"][idx_timeline])
            self.basestations.update_assoc(
                slice_assoc=self.associations["basestation_slice_assoc"][idx_timeline],
                ue_assoc=self.associations["basestation_ue_assoc"][idx_timeline],
            )

        if self.step_number == self.max_number_steps:
            self.metrics_hist.save(self.simu_name, self.episode_number)

        return (
            obs,
            reward,
            self.step_number == self.max_number_steps,
            {},
        )

    def reset(self, initial_episode: int = -1) -> Any:
        if (
            (self.step_number == 0 and self.episode_number == 1)
            or (self.episode_number == self.max_number_episodes)
            or initial_episode != -1
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
            "basestation_ue_assoc": self.basestations.ue_assoc,
            "basestation_slice_assoc": self.basestations.slice_assoc,
            "slice_ue_assoc": self.slices.ue_assoc,
            "sched_decision": [],
            "pkt_incoming": self.traffic.step(self.step_number, self.episode_number),
            "pkt_throughputs": np.zeros(self.max_number_ues),
            "pkt_effective_thr": np.zeros(self.max_number_ues),
            "buffer_occupancies": np.zeros(self.max_number_ues),
            "buffer_latencies": np.zeros(self.max_number_ues),
            "dropped_pkts": np.zeros(self.max_number_ues),
            "slice_req": self.slices.requirements,
        }

        return self.obs_space_format(obs)

    @staticmethod
    def calculate_reward_default(obs_space: dict) -> float:
        return 0

    @staticmethod
    def obs_space_format_default(obs_space: dict) -> Any:
        return np.array(list(obs_space.items()), dtype=object)

    def check_env_agent(
        self,
        sched_decision: np.ndarray,
        spectral_efficiencies: np.ndarray,
        mobilities: np.ndarray,
        traffics: np.ndarray,
    ) -> None:
        # Scheduling decision check
        assert len(sched_decision) == self.max_number_basestations and isinstance(
            sched_decision, np.ndarray
        ), "Sched decision shape does not match the number of basestations or is not of type list"
        for i, basestation_sched in enumerate(sched_decision):
            basestation_sched = np.array(basestation_sched)
            assert basestation_sched.shape == (
                self.max_number_ues,
                self.num_available_rbs[i],
            ), "Scheduling decision does not present the correct shape"
            if np.sum(np.sum(basestation_sched, axis=0) > 1) > 0:
                raise Exception(
                    "Scheduling decision allocated the same RB for more than one UE"
                )
        # Spectral efficiency check
        assert isinstance(
            spectral_efficiencies, np.ndarray
        ), "Spectral efficiencies are not list type"
        for i, basestation_spec in enumerate(spectral_efficiencies):
            assert basestation_spec.shape == (
                self.max_number_ues,
                self.num_available_rbs[i],
            ), "Spectral efficiences have wrong shape."
        # Mobility check
        assert isinstance(mobilities, np.ndarray) and mobilities.shape == (
            self.max_number_ues,
            self.mobility_size,
        ), "Mobility values are not numpy arrays or have wrong shape."
        # Traffics check
        assert isinstance(traffics, np.ndarray) and traffics.shape == (
            self.max_number_ues,
        ), "Traffics values are not numpy arrays or have wrong shape."

    def create_scenario(self) -> None:
        self.ues = UEs(
            self.max_number_ues,
            self.max_buffer_latencies,
            self.max_buffer_pkts,
            self.pkt_sizes,
        )
        self.slices = Slices(
            self.max_number_slices, self.max_number_ues, self.init_slice_ue_assoc
        )
        self.basestations = Basestations(
            self.max_number_basestations,
            self.max_number_slices,
            self.init_basestation_slice_assoc,
            self.init_basestation_ue_assoc,
            self.bandwidths,
            self.carrier_frequencies,
            self.num_available_rbs,
        )
        self.mobility = self.MobilityClass(self.max_number_ues)
        self.channel = self.ChannelClass(
            self.max_number_ues,
            self.max_number_basestations,
            self.num_available_rbs,
        )
        self.traffic = self.TrafficClass(self.max_number_ues)
        self.metrics_hist = Metrics(self.hist_root_path)
