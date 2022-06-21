from abc import ABC, abstractmethod


class Traffic(ABC):
    def __init__(self, max_number_ues) -> None:
        self.max_number_ues = max_number_ues

    @abstractmethod
    def step(self) -> list:
        pass


def main():
    pass


if __name__ == "__main__":
    main()
