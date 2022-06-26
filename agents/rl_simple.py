import numpy as np
from gym import spaces
from stable_baselines3 import SAC

from agents.agent import Agent
from comm_env import CommunicationEnv


class RLSimple(Agent):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: list,
        env: CommunicationEnv,
        hyperparameters: dict = {},
        seed: int = 0,
    ) -> None:
        super().__init__(max_number_ues, max_number_basestations, num_available_rbs)
        self.agent = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log="./tensorboard-logs/",
            seed=seed,
        )

    def step(self, obs_space: list) -> list:
        return self.agent.predict(obs_space, deterministic=True)[0]

    def train(self, total_timesteps: int) -> None:
        self.agent.learn(total_timesteps=int(total_timesteps), callback=[])

    @staticmethod
    def obs_space_format(obs_space: dict) -> list:
        formatted_obs_space = []
        hist_labels = [
            # "pkt_incoming",
            "dropped_pkts",
            # "pkt_effective_thr",
            "buffer_occupancies",
            # "spectral_efficiencies",
        ]
        for hist_label in hist_labels:
            if hist_label == "spectral_efficiencies":
                formatted_obs_space = np.append(
                    formatted_obs_space,
                    np.squeeze(np.sum(obs_space[hist_label], axis=2)),
                    axis=0,
                )
            else:
                formatted_obs_space = np.append(
                    formatted_obs_space, obs_space[hist_label], axis=0
                )

        return formatted_obs_space

    @staticmethod
    def calculate_reward(obs_space: dict) -> float:
        reward = -np.sum(obs_space["dropped_pkts"])
        return reward

    @staticmethod
    def get_action_space() -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(2,))

    @staticmethod
    def get_obs_space() -> spaces.Box:
        return spaces.Box(low=0, high=np.inf, shape=(2 * 2,), dtype=np.float64)

    @staticmethod
    def action_format(
        action: list,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: list,
    ) -> list:
        idx_chosen_ue = np.argmax(action)
        sched_decision = [
            [
                np.ones(num_available_rbs)
                if ue == idx_chosen_ue
                else np.zeros(num_available_rbs)
                for ue in np.arange(max_number_ues)
            ]
        ]

        return sched_decision


def main():
    pass


if __name__ == "__main__":
    main()
