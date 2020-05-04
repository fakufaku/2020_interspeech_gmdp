import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from repsimtools import (DirtyGitRepositoryError, InvalidGitRepositoryError,
                         get_git_hash)

OUTPUT_DIR = Path("figures")

BEST_MAX_ITER = 10
CLASSIC_P = "2.0"
CLASSIC_Q = "2.0"


def find_best_pq_sdr(df, config):

    sub_data = []

    # without scaling
    # sub_data.append(df[df["proj_algo"] == "None"])

    # projection back
    sub_data.append(df[df["proj_algo"] == "projection_back"])

    # minimum distortion
    lo, hi, n_steps = config["minimum_distortion"]["p_list"]
    # p_vals = np.linspace(lo, hi, n_steps)
    # tmp
    p_vals = [f"{p:0.1f}" for p in np.linspace(lo, hi, n_steps)]

    # classic md
    sub_data.append(
        df[
            (df["proj_algo"] == "minimum_distortion")
            & (df["p"] == CLASSIC_P)
            & (df["q"] == CLASSIC_Q)
        ].replace({"proj_algo": {"minimum_distortion": "minimum_distortion_l2"}})
    )

    best_params = []

    pt_n_iter = df.pivot_table(
        index="q", columns="p", values="n_iter", aggfunc=np.median
    )

    # find the best variants under all conditions
    for bss_algo in df["bss_algo"].unique():
        for n_chan in df["n_channels"].unique():

            df_loc = df[
                (df["proj_algo"] == "minimum_distortion")
                & (df["bss_algo"] == bss_algo)
                & (df["n_channels"] == n_chan)
            ]

            pt_sdr = df_loc.pivot_table(
                index="q", columns="p", values="sdr", aggfunc=np.mean,
            )

            pt_sir = df_loc.pivot_table(
                index="q", columns="p", values="sir", aggfunc=np.mean,
            )

            print(bss_algo, n_chan)
            print(pt_sdr)
            print()

            # TBD replace by 2.0!
            sdr_l2 = pt_sdr[CLASSIC_P][CLASSIC_Q]

            max_sdr_lbl = "md_best_sdr"
            max_sdr = sdr_l2
            max_sdr_id = [CLASSIC_P, CLASSIC_Q]

            max_sir_lbl = "md_best_sir"
            max_sir = sdr_l2
            max_sir_id = [CLASSIC_P, CLASSIC_Q]

            max_n_iter_lbl = "md_best_sir_m10"
            max_n_iter = sdr_l2
            max_n_iter_id = [CLASSIC_P, CLASSIC_Q]

            for ip, p in enumerate(p_vals):
                for q in p_vals[ip:]:

                    if pt_sdr[p][q] > max_sdr:
                        max_sdr = pt_sdr[p][q]
                        max_sdr_id = [p, q]

                    if pt_sdr[p][q] >= sdr_l2 and pt_sir[p][q] > max_sir:
                        max_sir = pt_sir[p][q]
                        max_sir_id = [p, q]

                    if (
                        pt_sdr[p][q] >= sdr_l2
                        and pt_n_iter[p][q] <= BEST_MAX_ITER
                        and pt_sir[p][q] > max_n_iter
                    ):
                        max_n_iter = pt_sir[p][q]
                        max_n_iter_id = [p, q]

            sub_data.append(
                df_loc[
                    (df_loc["proj_algo"] == "minimum_distortion")
                    & (df_loc["p"] == max_sdr_id[0])
                    & (df_loc["q"] == max_sdr_id[1])
                ].replace({"proj_algo": {"minimum_distortion": max_sdr_lbl}})
            )

            best_params.append(
                {
                    "bss_algo": bss_algo,
                    "n_channels": n_chan,
                    "proj_algo": max_sdr_lbl,
                    "p": max_sdr_id[0],
                    "q": max_sdr_id[1],
                    "n_iter": pt_n_iter[max_sdr_id[0]][max_sdr_id[1]],
                }
            )

            sub_data.append(
                df_loc[
                    (df_loc["proj_algo"] == "minimum_distortion")
                    & (df_loc["p"] == max_sir_id[0])
                    & (df_loc["q"] == max_sir_id[1])
                ].replace({"proj_algo": {"minimum_distortion": max_sir_lbl}})
            )

            best_params.append(
                {
                    "bss_algo": bss_algo,
                    "n_channels": n_chan,
                    "proj_algo": max_sir_lbl,
                    "p": max_sir_id[0],
                    "q": max_sir_id[1],
                    "n_iter": pt_n_iter[max_sir_id[0]][max_sir_id[1]],
                }
            )

            sub_data.append(
                df_loc[
                    (df_loc["proj_algo"] == "minimum_distortion")
                    & (df_loc["p"] == max_n_iter_id[0])
                    & (df_loc["q"] == max_n_iter_id[1])
                ].replace({"proj_algo": {"minimum_distortion": max_n_iter_lbl}})
            )

            best_params.append(
                {
                    "bss_algo": bss_algo,
                    "n_channels": n_chan,
                    "proj_algo": max_n_iter_lbl,
                    "p": max_n_iter_id[0],
                    "q": max_n_iter_id[1],
                    "n_iter": pt_n_iter[max_n_iter_id[0]][max_n_iter_id[1]],
                }
            )

    sub_data = pd.concat(sub_data)
    best_params_df = pd.DataFrame(best_params)

    return sub_data, best_params_df


def print_table(sub_data, best_params, metrics=[]):
    """
    Merge the two data frames and print the result as a nice latex table
    """

    # compute the average metrics
    average_metrics = (
        sub_df.groupby(["bss_algo", "n_channels", "proj_algo"])
        .mean()
        .drop(columns=["room_id"])
    ).reset_index()

    # merge to get p, q and median(n_iter) all in the same dataframe
    average_metrics = average_metrics.merge(
        best_params, on=["bss_algo", "n_channels", "proj_algo"], how="outer"
    )

    # make
    average_metrics[average_metrics["proj_algo"] == "minimum_distortion_l2"][
        "p"
    ].replace({np.nan: "2.0"})
    average_metrics[average_metrics["proj_algo"] == "minimum_distortion_l2"][
        "q"
    ].replace({np.nan: "2.0"})
    average_metrics = average_metrics.rename(
        columns={"n_iter_x": "n_iter_avg", "n_iter_y": "n_iter_med"}
    )

    # shorter name
    avgmet = average_metrics

    # now print this into a table that we can import in latex

    # First row has only the name of the algorithms
    print(" & ", end="")
    for proj_algo in avgmet["proj_algo"].unique():
        print(f" & \\multicolumn{{3}}{{c}}{{ {proj_algo} }} ", end="")
    print(" \\\\")

    # Second row has the parameter names (algo/channels) and the metric names
    print("\\text{Algo} & \\text{Channels} ", end="")
    for proj_algo in avgmet["proj_algo"].unique():
        for metric in metrics:
            print(f" & \\text{{ {metric} }}", end="")
    print(" \\\\")

    for bss_algo in avgmet["bss_algo"].unique():
        for m, n_chan in enumerate(avgmet["n_channels"].unique()):

            if m == 0:
                print(f"\\text{{ {bss_algo} }} & \\text{{ {n_chan} }} ", end="")
            else:
                print(f" & \\text{{ {n_chan} }} ", end="")

            for proj_algo in avgmet["proj_algo"].unique():
                for metric in metrics:
                    val = avgmet[
                        (avgmet["bss_algo"] == bss_algo)
                        & (avgmet["n_channels"] == n_chan)
                        & (avgmet["proj_algo"] == proj_algo)
                    ][metric]
                    assert len(val) == 1
                    val = val[val.index[0]]
                    if isinstance(val, float):
                        print(f" & {val:0.2f}", end="")
                    else:
                        print(f" & {val}", end="")
            print(" \\\\")

    return avgmet


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sparse minimum distortion simulation results analaysis"
    )
    parser.add_argument("results_dir", type=Path, help="Location of simulation results")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Allow to run with uncommited modif in the repo",
    )
    args = parser.parse_args()

    with open(args.results_dir / "data.json") as f:
        data = json.load(f)

    with open(args.results_dir / "parameters.json") as f:
        params = json.load(f)

    # create a unique output directory name
    try:
        plot_hash = get_git_hash(".")
        git_dirty = False
    except DirtyGitRepositoryError:
        git_dirty = True
        if args.dummy:
            plot_hash = "dummy"
        else:
            raise DirtyGitRepositoryError(
                "Uncommited changes. Commit first, or run with --dummy option"
            )
    except InvalidGitRepositoryError:
        git_dirty = False
        plot_hash = "nogit"

    output_dir = OUTPUT_DIR / Path(
        f"{params['_date']}_{params['name']}_sim_{params['_git_sha']}_plot_{plot_hash}"
    )
    if not OUTPUT_DIR.exists():
        os.mkdir(OUTPUT_DIR)
    if not output_dir.exists():
        os.mkdir(output_dir)

    # concatenate all the results
    results = []
    bss_runtimes = []
    for r in data:
        for e in r:
            if "bss_runtime" in e:
                bss_runtimes.append(e["bss_runtime"])
            else:
                results.append(e)

    for r in results:
        for metric in ["sdr", "sir"]:
            r[metric] = np.mean(r[metric])

    df = pd.DataFrame(results)

    def draw_heatmap(*args, **kwargs):
        global ax

        data = kwargs.pop("data")
        d = data.pivot_table(
            index=args[1], columns=args[0], values=args[2], aggfunc=np.mean,
        )

        ax_hm = sns.heatmap(d, **kwargs)
        ax_hm.invert_yaxis()

        ax = plt.gca()

    for metric in ["sdr", "sir", "n_iter"]:
        fg = sns.FacetGrid(
            df,
            col="n_channels",
            row="bss_algo",
            # row_order=algo_order[n_targets],
            # margin_titles=True,
            # aspect=aspect,
            # height=height,
        )
        fg.map_dataframe(
            draw_heatmap,
            "p",
            "q",
            metric,
            # cbar=False,
            # vmin=0.0,
            # vmax=1.0,
            # xticklabels=[1, "", "", 4, "", "", 7, "", "", 10],
            # yticklabels=[10, "", "", 40, "", "", 70, "", "", 100],
            # yticklabels=yticklabels,
            square=True,
        )

        fg.set_titles(template=f"{metric} " + "| {row_name} | mics={col_name}")

        for suffix in ["png", "pdf"]:
            plt.savefig(output_dir / f"heatmaps_{metric}.{suffix}")

    """
    Now we will plot the box plots for key algorithms
    """

    sub_df, best_params = find_best_pq_sdr(df, params)

    print_table(sub_df, best_params, metrics=["sdr", "sir"])

    print_table(sub_df, best_params, metrics=["p", "q", "n_iter_med"])

    plt.show()
