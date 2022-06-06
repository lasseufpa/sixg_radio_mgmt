import numpy as np


class Mobility:
    def __init__(self, max_number_ues: int) -> None:
        self.max_number_ues = max_number_ues

    def step(self) -> np.array:
        return np.ones(self.max_number_ues)


def main():
    pass


if __name__ == "__main__":
    main()
