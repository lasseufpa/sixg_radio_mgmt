import numpy as np


class Slices:
    """
    Slices class containing the slices functions. Each slice is assigned to a
    basestation.
    """

    def __init__(
        self,
        max_number_slices: int,
        max_number_ues: int,
        ue_assoc: np.array,
        requirements: dict = None,
    ) -> None:
        self.max_number_slices = max_number_slices
        self.max_number_ues = max_number_ues
        self.ue_assoc = ue_assoc  # Matrix of |Slices|x|UEs|
        self.requirements = requirements

    def update_assoc(
        self,
        ue_assoc: np.array = None,
    ) -> None:
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def update_slice_req(self, requirements: dict) -> None:
        self.requirements = requirements

    def get_number_ue_per_slice(self) -> np.array:
        return np.sum(self.ue_assoc, axis=1)


def main():
    requirements = {
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 0.001},
    }
    ue_assoc = [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1]]
    slices = Slices(2, 5, ue_assoc, requirements)
    print("Number of UEs per slice: {}".format(slices.get_number_ue_per_slice()))
    slices.update_assoc([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]])
    print("Number of UEs per slice: {}".format(slices.get_number_ue_per_slice()))
    print("Slice requirements:\n{}".format(slices.requirements))


if __name__ == "__main__":
    main()
