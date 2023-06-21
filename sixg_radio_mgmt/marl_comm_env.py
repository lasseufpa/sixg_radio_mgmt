from typing import Callable, Optional, Tuple, Type, TypeVar, Union

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


class MARLCommEnv(AECEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.possible_agents = [
            "player_" + str(r) for r in range(kwargs["number_agents"])
        ]
        kwargs.pop("number_agents")
        self.comm_env = CommunicationEnv(*args, **kwargs)
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_actions = {agent: None for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        try:
            self.observation_spaces = (
                self.comm_env.observation_space
                if isinstance(self.comm_env.observation_space, dict | None)
                else {}
            )
            self.action_spaces = (
                self.comm_env.action_space
                if isinstance(self.comm_env.action_space, dict | None)
                else {}
            )
        except AttributeError:
            self.observation_spaces = {}
            self.action_spaces = {}

    def observe(self, agent: str) -> Optional[np.ndarray]:
        return np.array(self.observations[agent])

    def reset(
        self,
        seed: Optional[int] = None,
        options: dict = {"initial_episode": -1},
        return_info: bool = True,
    ):
        obs, _ = self.comm_env.reset(seed=seed)

        # MARL variables
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_actions = {agent: None for agent in self.agents}
        self.observations = obs
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: np.ndarray):
        # Multi-agent specific
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.agent_actions[self.agent_selection] = action  # type: ignore

        # Multi-agent specific
        if self._agent_selector.is_last() and action is not None:
            # Single-agent specific
            (
                self.observations,
                rewards,
                termination,
                truncation,
                info,
            ) = self.comm_env.step(self.agent_actions)
            self.rewards = rewards if isinstance(rewards, dict) else {}
            if termination:
                self.terminations = {agent: True for agent in self.agents}
                self.agents = []  # All agents are terminated together
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.agent_actions[
                self.agents[1 - self.agent_name_mapping[agent]]
            ] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
