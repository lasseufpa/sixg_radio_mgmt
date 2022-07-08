import numpy as np
from tqdm import tqdm

from agents.round_robin_slice import RoundRobin
from channels.simple import SimpleChannel
from comm_env import CommunicationEnv
from mobilities.simple import SimpleMobility
from traffics.simple import SimpleTraffic

round_robin = RoundRobin(3, 2, np.array([8, 8]))
comm_env = CommunicationEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    "simple_slice",
    round_robin.action_format,
    round_robin.obs_space_format,
    round_robin.calculate_reward,
)
obs = comm_env.reset()
number_steps = 10
for step_number in tqdm(np.arange(comm_env.max_number_steps)):
    sched_decision = round_robin.step(obs)
    obs, _, end_ep, _ = comm_env.step(sched_decision)
    if end_ep:
        comm_env.reset()
