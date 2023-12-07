from typing import Callable, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .comm_env import CommunicationEnv


class MARLCommEnv(MultiAgentEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.agents = {
            "player_" + str(r) for r in range(kwargs["number_agents"])
        }
        self._agent_ids = set(self.agents)
        kwargs.pop("number_agents")
        self.comm_env = CommunicationEnv(*args, **kwargs)
        self._obs_space_in_preferred_format = True
        self.observation_space = self.comm_env.observation_space
        self._action_space_in_preferred_format = True
        self.action_space = self.comm_env.action_space

        super().__init__()

    def reset(
        self,
        seed: Optional[int] = None,
        options: dict = {"initial_episode": -1},
        return_info: bool = True,
    ):
        if options is None:
            options = {"initial_episode": -1}
        obs, _ = self.comm_env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action_dict):
        obs, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        (
            obs,
            rewards,
            termination,
            truncation,
            info,
        ) = self.comm_env.step(action_dict)
        if termination:
            if isinstance(action_dict, dict):
                terminated = {agent: True for agent in self.agents}
                truncated = {agent: True for agent in self.agents}
                terminated["__all__"], truncated["__all__"] = True, True
            else:
                terminated, truncated = True, True
        else:
            if isinstance(action_dict, dict):
                terminated = {agent: False for agent in self.agents}
                truncated = {agent: False for agent in self.agents}
                terminated["__all__"], truncated["__all__"] = False, False
            else:
                terminated = False
                truncated = False
        return obs, rewards, terminated, truncated, info
