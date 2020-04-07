import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy

import pyroomacoustics as pra
from process import bss_algorithms, process


def gen_args(config_fn):

    with open(config_fn, "r") as f:
        config = json.load(f)

    with open(config["metadata_fn"], "r") as f:
        metadata = json.load(f)

    args = []

    for label, room_list in metadata.items():

        n_rooms = len(room_list)
        n_channels = int(label[0])

        for room_id in range(n_rooms):

            for bss_algo in bss_algorithms.keys():

                args.append([n_channels, room_id, bss_algo, str(config_fn)])

    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run experiment in parallel")
    parser.add_argument("config_file", type=str, help="Path to the configuration file")
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Fix number of iterations to two for test purposes",
    )
    parser.add_argument(
        "-s", "--seq", action="store_true", help="Run the experiment sequentially",
    )
    args = parser.parse_args()

    sim_args = gen_args(args.config_file)

    if args.test:
        sim_args = sim_args[:2]

    all_results = []

    if args.seq:
        for this_args in sim_args:
            all_results += process(this_args)

    else:
        with Pool() as p:
            results = p.map(process, sim_args)
            for r in results:
                all_results += r
