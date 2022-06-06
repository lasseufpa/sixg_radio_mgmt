import os
from typing import Dict

import numpy as np


class Metrics:
    def __init__(
        self,
        root_path=str,
    ) -> None:
        self.root_path = root_path
        self.metrics_hist = {
            "pkt_incoming": [],
            "pkt_throughputs": [],
            "pkt_effective_thr": [],
            "buffer_occupancies": [],
            "buffer_latencies": [],
            "dropped_pkts": [],
            "mobility": [],
            "spectral_efficiencies": [],
            "traffics": [],
            "rewards": [],
        }

    def step(self, hist) -> None:
        for metric in hist.keys():
            self.metrics_hist[metric].append(hist[metric])

    def save(self, simu_name: str, episode_number: int) -> None:
        path = ("{}/hist/{}/").format(
            self.root_path,
            simu_name,
        )
        try:
            os.makedirs(path)
        except OSError:
            pass

        np.savez_compressed(path + "ep_" + str(episode_number), **self.metrics_hist)

    @staticmethod
    def read(root_path: str, simu_name: str, episode_number: int) -> Dict:
        path = "{}/hist/{}/ep_{}.npz".format(root_path, simu_name, episode_number)
        data = np.load(path)
        data_dict = {
            "pkt_incoming": data.f.pkt_incoming,
            "pkt_throughputs": data.f.pkt_throughputs,
            "pkt_effective_thr": data.f.pkt_effective_thr,
            "buffer_occupancies": data.f.buffer_occupancies,
            "buffer_latencies": data.f.buffer_latencies,
            "dropped_pkts": data.f.dropped_pkts,
            "mobility": data.f.mobility,
            "spectral_efficiencies": data.f.spectral_efficiencies,
            "traffics": data.f.traffics,
            "rewards": data.f.rewards,
        }
        return data_dict


def main():
    data = Metrics.read("./", "test", 1)
    print(data)


if __name__ == "__main__":
    main()
