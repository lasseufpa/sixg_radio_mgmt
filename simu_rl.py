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
training_episodes = 300
initial_test_episode = 900
# check_env(comm_env)
rl_agent = RLSimple(2, 1, [1], comm_env)
# rl_agent.train(comm_env.max_number_steps * training_episodes)
# rl_agent.save("sac")
rl_agent.load("sac", comm_env)

# RL Agent
obs = comm_env.reset(initial_test_episode)
# comm_env.logs = False
rewards = np.zeros(
    (comm_env.max_number_episodes - initial_test_episode, comm_env.max_number_steps)
)
for episode in np.arange(comm_env.max_number_episodes - initial_test_episode):
    # for step_number in tqdm(np.arange(comm_env.max_number_steps)):
    for step_number in np.arange(comm_env.max_number_steps):
        sched_decision = rl_agent.step(obs)
        obs, reward, _, _ = comm_env.step(sched_decision)
        rewards[episode, step_number] = reward
        if (
            step_number == comm_env.max_number_steps - 1
            and episode != comm_env.max_number_episodes - initial_test_episode - 1
        ):
            comm_env.reset()
np.savez_compressed("rewards_rl.npz", rewards=rewards)

# ## Round Robin agent
# obs = comm_env.reset(initial_test_episode)
# sched_decision = [[[1], [0]]]
# # comm_env.logs = False
# rewards = np.zeros(
#     (comm_env.max_number_episodes - initial_test_episode, comm_env.max_number_steps)
# )
# for episode in np.arange(comm_env.max_number_episodes - initial_test_episode):
#     # for step_number in tqdm(np.arange(comm_env.max_number_steps)):
#     for step_number in np.arange(comm_env.max_number_steps):
#         obs, reward, _, _ = comm_env.step(sched_decision)
#         sched_decision = np.roll(sched_decision, 1)
#         rewards[episode, step_number] = reward
#         if (
#             step_number == comm_env.max_number_steps - 1
#             and episode != comm_env.max_number_episodes - initial_test_episode - 1
#         ):
#             comm_env.reset()
# np.savez_compressed("rewards_rl.npz", rewards=rewards)
