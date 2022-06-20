import numpy as np


class Basestations:
    def __init__(
        self,
        max_number_basestations: int,
        max_number_slices: int,
        basestation_slice_assoc: list,
        basestation_ue_assoc: list,
        bandwidths: list,
        carrier_frequencies: list,
        num_available_rbs: list,
    ) -> None:
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.basestation_slice_assoc = (
            basestation_slice_assoc  # Matrix with dimensions |basestations|x|slices|
        )
        self.basestation_ue_assoc = basestation_ue_assoc
        self.bandwidths = bandwidths
        self.carrier_frequencies = carrier_frequencies
        self.num_available_rbs = num_available_rbs

    def get_assoc(self) -> list:
        return self.basestation_slice_assoc

    def update_assoc(
        self,
        basestation_slice_assoc: list,
        basestation_ue_assoc: list,
    ) -> None:
        self.basestation_slice_assoc = basestation_slice_assoc
        self.basestation_ue_assoc = basestation_ue_assoc

    def get_number_slices_per_basestation(self) -> list:
        return np.sum(self.basestation_slice_assoc, axis=1)


def main():
    basestation_slice_assoc = [[0, 1, 1, 1], [1, 0, 0, 1]]
    basestations = Basestations(2, 4, basestation_slice_assoc)
    print(
        "Number of slices per basestation: {}".format(
            basestations.get_number_slices_per_basestation()
        )
    )
    basestations.update_assoc([[0, 1, 1, 0], [1, 0, 0, 1]])
    print(
        "Number of slices per basestation: {}".format(
            basestations.get_number_slices_per_basestation()
        )
    )


if __name__ == "__main__":
    main()
