# This file contains code to run the 1st experiment
#
# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import datetime
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

    # date of simulation in string format
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    all_results = []

    if args.seq:
        for this_args in sim_args:
            all_results += process(this_args)

    else:
        with Pool() as p:
            results = p.map(process, sim_args)
            for r in results:
                all_results += r

    filename = f"{date_str}_smd_results.json"
    with open(filename, "w") as f:
        json.dump(all_results, f)
