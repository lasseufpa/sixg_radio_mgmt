import numpy as np

# from comm.mobility import Mobility


# class SimpleMobility(Mobility):
#     def __init__(self, max_number_ues: int) -> None:
#         super().__init__(max_number_ues)

#     def step(self, step_number: int, episode_number: int) -> list:
#         return np.ones((self.max_number_ues, 2))


class GridWorld:
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        ue_initial_positions: list,
        basestation_position: list,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.ue_positions = (
            ue_initial_positions  # [[1,2],... [3,4]], First UE on row 1 column 2
        )
        self.ue_previous_actions = ["up" for ue in self.ue_positions]
        self.basestation_position = basestation_position
        self.grid = np.zeros((n_rows, n_cols))
        self.grid[
            np.array(self.ue_positions)[:, 0],
            np.array(self.ue_positions)[:, 1],
        ] = 1  # 1 represent UEs
        self.grid[
            self.basestation_position[0], self.basestation_position[1]
        ] = 2  # 2 represents the basestation
        self.actions_move = [
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [0, 0],
        ]  # Up, right, down, left, stay
        self.prob_previous_action = 0.5

    def step(
        self,
    ):
        for ue in np.arange(len(self.ue_positions)):
            valid_actions = self.calc_valid_actions(self.ue_positions[ue])
            self.move_ue(ue, valid_actions)

    def calc_valid_actions(self, ue_position):
        valid_actions = [True, True, True, True, True]  # Up, right, down, left, stay

        for action_idx in np.arange(
            len(valid_actions) - 1
        ):  # Stay is always a valid action
            try:
                if (
                    self.grid[
                        ue_position[0] + self.actions_move[action_idx][0],
                        ue_position[1] + self.actions_move[action_idx][1],
                    ]
                    != 0
                ):
                    valid_actions[action_idx] = False
            except IndexError:  # in case the index is out of array bounds
                valid_actions[action_idx] = False

        return valid_actions

    def move_ue(self, ue_idx, valid_actions):
        """
        Still moving in the previous direction has a higher probability (0.5)
        and the remaining probability is divided among the other valid actions.
        """
        actions = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3,
            "stay": 4,
        }
        previous_action_idx = actions[self.ue_previous_actions[ue_idx]]
        if valid_actions[previous_action_idx]:
            if np.sum(valid_actions) != 1:
                prob_other_actions = self.prob_previous_action / (
                    np.sum(valid_actions) - 1
                )
                prob = np.array(valid_actions, dtype=int) * prob_other_actions
                prob[previous_action_idx] = 0.5
            elif np.sum(valid_actions) == 1:
                prob = np.array(valid_actions, dtype=int) * 0
                prob[previous_action_idx] = 1
        else:
            prob_other_actions = 1 / (np.sum(valid_actions))
            prob = np.array(valid_actions, dtype=int) * prob_other_actions

        action_choice = np.random.choice(5, p=prob)
        self.grid[self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1]] = 0
        self.grid[
            self.ue_positions[ue_idx][0] + self.actions_move[action_choice][0],
            self.ue_positions[ue_idx][1] + self.actions_move[action_choice][1],
        ] = 1
        self.ue_positions[ue_idx][0] = (
            self.ue_positions[ue_idx][0] + self.actions_move[action_choice][0]
        )
        self.ue_positions[ue_idx][1] = (
            self.ue_positions[ue_idx][1] + self.actions_move[action_choice][1]
        )
        self.ue_previous_actions[ue_idx] = list(actions.keys())[action_choice]


def main():
    gridworld = GridWorld(5, 5, [[3, 3], [2, 2]], [4, 0])
    for i in np.arange(10):
        gridworld.step()
        print(gridworld.grid)
        print("\n")


if __name__ == "__main__":
    main()
