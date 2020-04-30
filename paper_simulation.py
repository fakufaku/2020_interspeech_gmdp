import argparse
import datetime
import json
import os
import traceback
from pathlib import Path

import numpy
import repsimtools

import pyroomacoustics as pra
from process import bss_algorithms, process

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])


def init(parameters):
    parameters["base_dir"] = base_dir


def gen_args(parameters):

    with open(parameters["metadata_fn"], "r") as f:
        metadata = json.load(f)

    args = []

    for label, room_list in metadata.items():

        n_rooms = len(room_list)
        n_channels = int(label[0])

        for room_id in range(n_rooms):

            for bss_algo in parameters["bss_algorithms"].keys():

                args.append([n_channels, room_id, bss_algo])

    return args


def one_loop(args):
    global parameters

    import sys

    sys.path.append(parameters["base_dir"])
    from process import process

    res = process(args, parameters)
    """
    try:
        res = process(args, parameters)

    except Exception:

        # get the traceback
        tb = traceback.format_exc()

        report = {
            "args": args,
            "tb": tb,
        }

        pid = os.getpid()

        # now write the problem to file
        fn_err = os.path.join(parameters["_results_dir"], "error_{}.json".format(pid))
        with open(fn_err, "a") as f:
            f.write(json.dumps(report, indent=4))
            f.write(",\n")

        res = []
    """

    return res


if __name__ == "__main__":

    repsimtools.run(
        one_loop,
        gen_args,
        func_init=init,
        base_dir=base_dir,
        results_dir="sim_results/",
        description="Simulation for OverIVA",
    )
