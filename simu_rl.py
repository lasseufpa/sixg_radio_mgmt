import numpy as np
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from agents.rl_simple import RLSimple
from agents.round_robin import RoundRobin
from channels.simple import SimpleChannel
from comm_env import CommunicationEnv
from mobilities.simple import SimpleMobility
from traffics.simple import SimpleTraffic

comm_env = CommunicationEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    "simple_rl",
    RLSimple.action_format,
    RLSimple.obs_space_format,
    RLSimple.calculate_reward,
    RLSimple.get_obs_space,
    RLSimple.get_action_space,
)
check_env(comm_env)
rl_agent = RLSimple(2, 1, [3], comm_env)
total_number_steps = 2000
rl_agent.train(total_number_steps)

obs = comm_env.reset()
for episode in np.arange(1, comm_env.max_number_episodes + 1):
    print("Episode ", episode)
    for step_number in tqdm(np.arange(comm_env.max_number_steps)):
        sched_decision = rl_agent.step(obs)
        obs, _, _, _ = comm_env.step(sched_decision)
        if step_number == comm_env.max_number_steps - 1:
            comm_env.reset()

# # Round-robin
# comm_env = CommunicationEnv(
#     SimpleChannel,
#     SimpleTraffic,
#     SimpleMobility,
#     "simple_rl",
#     RoundRobin.action_format,
#     RoundRobin.obs_space_format,
#     RoundRobin.calculate_reward,
# )

# rl_agent = RoundRobin(2, 1, [3])

# obs = comm_env.reset()
# for episode in np.arange(1, comm_env.max_number_episodes + 1):
#     print("Episode ", episode)
#     for step_number in tqdm(np.arange(comm_env.max_number_steps)):
#         sched_decision = rl_agent.step(obs)
#         obs, _, _, _ = comm_env.step(sched_decision)
#         if step_number == comm_env.max_number_steps - 1:
#             comm_env.reset()