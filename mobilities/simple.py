import numpy as np
from tqdm import tqdm

from comm.mobility import Mobility


class SimpleMobility(Mobility):
    # class SimpleMobility:
    def __init__(self, max_number_ues: int, episode: int) -> None:
        # super().__init__(max_number_ues)
        self.n_rows = 6
        self.n_cols = 6
        initial_positions = []
        for ue in np.arange(max_number_ues):
            initial_positions.append(
                [
                    np.random.randint(self.n_rows - 1),
                    np.random.randint(1, high=self.n_cols),
                ]
            )

        self.gridworld = GridWorld(
            self.n_rows, self.n_cols, initial_positions, [self.n_rows - 1, 0]
        )
        self.positions = self.read_files(episode)

    def step(self, step_number: int, episode_number: int) -> list:

        # return self.gridworld.step()
        return [self.positions[0][step_number], self.positions[1][step_number]]

    def read_files(self, episode):
        file = np.load("./mobility_val/ep{}.npz".format(episode))

        return [file.f.ue1, file.f.ue2]


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
        self.grid_horizontal_distance = 10
        self.grid_vertical_distance = 10
        self.block_vertical_distance = self.grid_vertical_distance / self.n_rows
        self.block_horizontal_distance = self.grid_horizontal_distance / self.n_cols

    def step(
        self,
    ):
        for ue in np.arange(len(self.ue_positions)):
            valid_actions = self.calc_valid_actions(self.ue_positions[ue])
            self.move_ue(ue, valid_actions)

        return self.pos_to_mag_angle()

    def calc_valid_actions(self, ue_position):
        valid_actions = [True, True, True, True, True]  # Up, right, down, left, stay

        for action_idx in np.arange(
            len(valid_actions) - 1
        ):  # Stay is always a valid action
            v_move = ue_position[0] + self.actions_move[action_idx][0]
            h_move = ue_position[1] + self.actions_move[action_idx][1]
            if (
                v_move < 0
                or v_move > (self.n_rows - 1)
                or h_move < 0
                or h_move > (self.n_cols - 1)
            ):
                valid_actions[action_idx] = False
            elif self.grid[v_move, h_move] != 0:
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
        # previous_action_idx = actions[self.ue_previous_actions[ue_idx]]

        if ue_idx == 0:
            prob = np.zeros(len(actions))
            main_actions = int(valid_actions[0]) + int(valid_actions[2])
            prob[[0, 2]] = 0.6 / main_actions if main_actions != 0 else 0
            other_actions = (
                int(valid_actions[1]) + int(valid_actions[3]) + int(valid_actions[4])
            )
            prob[[1, 3, 4]] = (
                (1 - np.sum(prob * valid_actions)) / other_actions
                if other_actions != 0
                else 0
            )
            prob = prob * valid_actions
        elif ue_idx == 1:
            prob = np.zeros(len(actions))
            main_actions = int(valid_actions[1]) + int(valid_actions[3])
            prob[[1, 3]] = 0.6 / main_actions if main_actions != 0 else 0
            other_actions = (
                int(valid_actions[0]) + int(valid_actions[2]) + int(valid_actions[4])
            )
            prob[[0, 2, 4]] = (
                (1 - np.sum(prob * valid_actions)) / other_actions
                if other_actions != 0
                else 0
            )
            prob = prob * valid_actions

        # if valid_actions[previous_action_idx]:
        #     if np.sum(valid_actions) != 1:
        #         prob_other_actions = self.prob_previous_action / (
        #             np.sum(valid_actions) - 1
        #         )
        #         prob = np.array(valid_actions, dtype=int) * prob_other_actions
        #         prob[previous_action_idx] = 0.5
        #     elif np.sum(valid_actions) == 1:
        #         prob = np.array(valid_actions, dtype=int) * 0
        #         prob[previous_action_idx] = 1
        # else:
        #     prob_other_actions = 1 / (np.sum(valid_actions))
        #     prob = np.array(valid_actions, dtype=int) * prob_other_actions

        action_choice = np.random.choice(len(actions), p=prob)
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

    def pos_to_mag_angle(self):
        pos_mag_angle = []
        for ue_pos in self.ue_positions:
            convert_axis = (
                np.abs(ue_pos[0] - (self.n_rows - 1)) * self.block_vertical_distance
                + ue_pos[1] * self.block_horizontal_distance * 1j
            )
            mag = np.linalg.norm(convert_axis)
            angle = np.angle(convert_axis)
            pos_mag_angle.append([mag, angle])

        return pos_mag_angle


def main():
    np.random.seed(10)
    grid_size = 6
    eps = 1000
    n_steps = 1000
    for ep in tqdm(np.arange(eps)):
        ue1_pos = np.zeros((n_steps, 2))
        ue2_pos = np.zeros((n_steps, 2))
        ue1_ini_pos = np.array([])
        ue2_ini_pos = np.array([])
        while np.array_equal(ue1_ini_pos, ue2_ini_pos) is True:
            ue1_ini_pos = [
                np.random.randint(0, grid_size - 1),
                np.random.randint(1, grid_size),
            ]
            ue2_ini_pos = [
                np.random.randint(0, grid_size - 1),
                np.random.randint(1, grid_size),
            ]
        ue1_pos[0] = np.array(ue1_ini_pos)
        ue2_pos[0] = np.array(ue2_ini_pos)
        gridworld = GridWorld(
            grid_size, grid_size, [ue1_ini_pos, ue2_ini_pos], [grid_size - 1, 0]
        )
        for step in tqdm(np.arange(1, n_steps), leave=False):
            gridworld.step()
            ue1_pos[step] = gridworld.ue_positions[0]
            ue2_pos[step] = gridworld.ue_positions[1]
            if (
                np.array_equal(ue1_pos[step], ue2_pos[step]) is True
                or np.array_equal(ue1_pos[step], np.array([5, 0])) is True
                or np.array_equal(ue2_pos[step], np.array([5, 0])) is True
            ):
                raise Exception(
                    "UEs in the same position or occupying basestation position"
                )

        np.savez_compressed(
            "./mobility_val/ep{}.npz".format(ep),
            ue1=np.array(ue1_pos),
            ue2=np.array(ue2_pos),
        )


def calc_valid_actions(n_rows, n_cols, ue_position):
    valid_actions = [True, True, True, True, True]  # Up, right, down, left, stay
    actions_move = [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
        [0, 0],
    ]  # Up, right, down, left, stay
    for action_idx in np.arange(
        len(valid_actions) - 1
    ):  # Stay is always a valid action
        # TODO remover movimentos pra basestation e ues na mesma posição
        v_move = ue_position[0] + actions_move[action_idx][0]
        h_move = ue_position[1] + actions_move[action_idx][1]
        if v_move < 0 or v_move > (n_rows - 1) or h_move < 0 or h_move > (n_cols - 1):
            valid_actions[action_idx] = False

    return valid_actions


def move_ue(
    n_rows,
    n_cols,
    ue_position,
    ue_idx,
    valid_actions,
):
    """
    Still moving in the previous direction has a higher probability (0.5)
    and the remaining probability is divided among the other valid actions.
    """
    grid = np.zeros((n_rows, n_cols))
    actions = {
        "up": 0,
        "right": 1,
        "down": 2,
        "left": 3,
        "stay": 4,
    }
    actions_move = [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
        [0, 0],
    ]  # Up, right, down, left, stay

    if ue_idx == 0:
        prob = np.zeros(len(actions))
        main_actions = int(valid_actions[0]) + int(valid_actions[2])
        prob[[0, 2]] = 0.6 / main_actions if main_actions != 0 else 0
        other_actions = (
            int(valid_actions[1]) + int(valid_actions[3]) + int(valid_actions[4])
        )
        prob[[1, 3, 4]] = (
            (1 - np.sum(prob * valid_actions)) / other_actions
            if other_actions != 0
            else 0
        )
        prob = prob * valid_actions
    elif ue_idx == 1:
        prob = np.zeros(len(actions))
        main_actions = int(valid_actions[1]) + int(valid_actions[3])
        prob[[1, 3]] = 0.6 / main_actions if main_actions != 0 else 0
        other_actions = (
            int(valid_actions[0]) + int(valid_actions[2]) + int(valid_actions[4])
        )
        prob[[0, 2, 4]] = (
            (1 - np.sum(prob * valid_actions)) / other_actions
            if other_actions != 0
            else 0
        )
        prob = prob * valid_actions

    for action_choice in np.arange(len(actions)):
        if valid_actions[action_choice] is True:
            grid[
                ue_position[0] + actions_move[action_choice][0],
                ue_position[1] + actions_move[action_choice][1],
            ] = prob[action_choice]

    return (prob, grid)


def main2():
    n_rows = 6
    n_cols = 6
    n_actions = 5

    for ue_idx in np.arange(2):
        pos_actions_prob = np.zeros((n_rows, n_cols, n_actions))
        matrix_pos_prob = np.zeros((n_rows, n_cols, n_rows, n_cols))
        for i in np.arange(n_rows):
            for j in np.arange(n_cols):
                ue_position = np.array([i, j])
                valid_actions = calc_valid_actions(n_rows, n_cols, ue_position)
                prob, grid = move_ue(n_rows, n_cols, ue_position, ue_idx, valid_actions)
                pos_actions_prob[i, j] = prob
                matrix_pos_prob[i, j] = grid
        np.savez_compressed(
            "mobility_ue{}.npz".format(ue_idx),
            pos_actions_prob=pos_actions_prob,
            matrix_pos_prob=matrix_pos_prob,
        )


if __name__ == "__main__":
    main()
