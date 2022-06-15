import numpy as np
from tqdm import tqdm

from agents.round_robin import RoundRobin
from comm_env import CommunicationEnv

comm_env = CommunicationEnv()
round_robin = RoundRobin(2, 2, [2, 2])
comm_env.reset()
for episode in np.arange(1, comm_env.max_number_episodes + 1):
    print("Episode ", episode)
    for step_number in tqdm(np.arange(comm_env.max_number_steps)):
        sched_decision = round_robin.step([], comm_env.basestation_ue_assoc)
        comm_env.step(sched_decision)
        if step_number == comm_env.max_number_steps - 1:
            comm_env.metrics_hist.save(comm_env.simu_name, comm_env.episode_number)
            comm_env.reset()
