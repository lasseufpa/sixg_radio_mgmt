import numpy as np
from tqdm import tqdm

from agents.round_robin import RoundRobin
from channels.simple import SimpleChannel
from comm_env import CommunicationEnv
from mobilities.simple import SimpleMobility
from traffics.simple import SimpleTraffic

round_robin = RoundRobin(2, 2, [2, 2])
comm_env = CommunicationEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    "simple",
    round_robin.action_format,
    round_robin.obs_space_format,
    round_robin.calculate_reward,
)
obs = comm_env.reset()
for episode in np.arange(1, comm_env.max_number_episodes + 1):
    print("Episode ", episode)
    for step_number in tqdm(np.arange(comm_env.max_number_steps)):
        sched_decision = round_robin.step(obs)
        print(sched_decision)
        obs, _, _, _ = comm_env.step(sched_decision)
        if step_number == comm_env.max_number_steps - 1:
            comm_env.reset()
