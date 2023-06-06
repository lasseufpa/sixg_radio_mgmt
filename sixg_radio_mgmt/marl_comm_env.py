from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from .association import Association
from .basestations import Basestations
from .channel import Channel
from .comm_env import CommunicationEnv
from .metrics import Metrics
from .mobility import Mobility
from .slices import Slices
from .traffic import Traffic
from .ues import UEs


class MARLCommEnv(AECEnv, CommunicationEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        CommunicationEnv.__init__(self, *args)

        self.possible_agents = [
            "player_" + str(r) for r in range(kwargs["number_agents"])
        ]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.observations = {}
        self.state = {agent: None for agent in self.agents}

    def observe(self, agent: str) -> Optional[np.ndarray]:
        return (
            self.observations[agent] if self.observations is not None else None
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: dict = {"initial_episode": -1},
    ):
        super().reset(seed=seed)

        if (
            (self.step_number == 0 and self.episode_number == 0)
            or (self.episode_number == (self.max_number_episodes - 1))
            or options["initial_episode"] != -1
        ):
            self.episode_number = (
                0
                if options["initial_episode"] == -1
                else options["initial_episode"]
            )
        elif self.episode_number < (self.max_number_episodes - 1):
            self.episode_number += 1
        else:
            raise Exception(
                "Episode number received a non expected value equals to {}.".format(
                    self.episode_number
                )
            )
        self.step_number = 0
        self.create_scenario()

        # MARL variables
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, sched_decision):
        # Multi-agent specific
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = sched_decision

        # Multi-agent specific
        if self._agent_selector.is_last():
            # Single-agent specific
            sched_decision = self.action_format(sched_decision)

            mobilities = self.mobility.step(
                self.step_number, self.episode_number
            )
            spectral_efficiencies = self.channel.step(
                self.step_number,
                self.episode_number,
                mobilities,
                sched_decision,
            )
            traffics = self.traffic.step(
                self.slices.ue_assoc,
                self.slices.requirements,
                self.step_number,
                self.episode_number,
            )

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
                self.basestations.bandwidths,
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

            if self.step_number == self.max_number_steps:
                self.metrics_hist.save(
                    self.simu_name, self.agent_name, self.episode_number
                )
            else:
                # Update associations
                (
                    self.basestations.ue_assoc,
                    self.basestations.slice_assoc,
                    self.slices.ue_assoc,
                    self.slices.requirements,
                ) = self.associations.step(
                    self.basestations.ue_assoc,
                    self.basestations.slice_assoc,
                    self.slices.ue_assoc,
                    self.slices.requirements,
                    self.step_number,
                    self.episode_number,
                )
            #### End single agent specific

            # rewards for all agents are placed in the .rewards dictionary
            (
                self.rewards[self.agents[0]],
                self.rewards[self.agents[1]],
            ) = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()
