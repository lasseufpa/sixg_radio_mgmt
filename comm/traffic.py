import numpy as np


class Traffic:
    def __init__(self, max_number_ues) -> None:
        self.max_number_ues = max_number_ues

    def step(self) -> list:
        return np.ones(self.max_number_ues) * 4


def main():
    pass


if __name__ == "__main__":
    main()
