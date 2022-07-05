import numpy as np
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from agents.rl_simple import RLSimple
from channels.simple import SimpleChannel
from comm_env import CommunicationEnv
from mobilities.simple import SimpleMobility
from traffics.simple import SimpleTraffic

comm_env = CommunicationEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    "simple",
    RLSimple.action_format,
    RLSimple.obs_space_format,
    RLSimple.calculate_reward,
    RLSimple.get_obs_space,
    RLSimple.get_action_space,
)
check_env(comm_env)
rl_agent = RLSimple(2, 2, [2, 2], comm_env)
total_number_steps = 10000
rl_agent.train(total_number_steps)

obs = comm_env.reset()
for step_number in tqdm(np.arange(total_number_steps)):
    sched_decision = rl_agent.step(obs)
    obs, _, end_ep, _ = comm_env.step(sched_decision)
    if end_ep:
        comm_env.reset()
