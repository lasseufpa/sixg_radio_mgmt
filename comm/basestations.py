import numpy as np


class Basestations:
    def __init__(
        self,
        max_number_basestations: int,
        max_number_slices: int,
        slice_assoc: np.array,
        ue_assoc: np.array,
        bandwidths: np.array,
        carrier_frequencies: np.array,
        num_available_rbs: np.array,
    ) -> None:
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.slice_assoc = slice_assoc  # Matrix with dimensions |basestations|x|slices|
        self.ue_assoc = ue_assoc
        self.bandwidths = bandwidths
        self.carrier_frequencies = carrier_frequencies
        self.num_available_rbs = num_available_rbs

    def get_assoc(self) -> np.array:
        return self.slice_assoc

    def update_assoc(
        self,
        slice_assoc: np.array = None,
        ue_assoc: np.array = None,
    ) -> None:
        self.slice_assoc = slice_assoc if slice_assoc is not None else self.slice_assoc
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def get_number_slices_per_basestation(self) -> np.array:
        return np.sum(self.slice_assoc, axis=1)


def main():
    slice_assoc = np.array([[0, 1, 1, 1], [1, 0, 0, 1]])
    basestations = Basestations(2, 4, slice_assoc)
    print(
        "Number of slices per basestation: {}".format(
            basestations.get_number_slices_per_basestation()
        )
    )
    basestations.update_assoc(np.array([[0, 1, 1, 0], [1, 0, 0, 1]]))
    print(
        "Number of slices per basestation: {}".format(
            basestations.get_number_slices_per_basestation()
        )
    )


if __name__ == "__main__":
    main()
