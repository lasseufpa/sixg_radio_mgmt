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
        slice_ue_assoc: list,
        slice_requirements: dict = None,
    ) -> None:
        self.max_number_slices = max_number_slices
        self.max_number_ues = max_number_ues
        self.slice_ue_assoc = slice_ue_assoc  # Matrix of |Slices|x|UEs|
        self.slice_requirements = slice_requirements

    def update_assoc(
        self,
        slice_ue_assoc: list,
    ) -> None:
        self.slice_ue_assoc = slice_ue_assoc

    def update_slice_req(self, slice_requirements: dict) -> None:
        self.slice_requirements = slice_requirements

    def get_number_ue_per_slice(self) -> list:
        return np.sum(self.slice_ue_assoc, axis=1)


def main():
    slice_requirements = {
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 0.001},
    }
    slice_ue_assoc = [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1]]
    slices = Slices(2, 5, slice_ue_assoc, slice_requirements)
    print("Number of UEs per slice: {}".format(slices.get_number_ue_per_slice()))
    slices.update_assoc([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]])
    print("Number of UEs per slice: {}".format(slices.get_number_ue_per_slice()))
    print("Slice requirements:\n{}".format(slices.slice_requirements))


if __name__ == "__main__":
    main()
