import json
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sparse minimum distortion simulation results analaysis")
    parser.add_argument("results_dir", type=Path, help="Location of simulation results")
    args = parser.parse_args()
    

    with open(args.results_dir / "data.json") as f:
        data = json.load(f)

    with open(args.results_dir / "parameters.json") as f:
        params = json.load(f)

    # concatenate all the results
    results = []
    for r in data:
        results += r

    for r in results:
        for metric in ["sdr", "sir"]:
            r[metric] = np.mean(r[metric])

    df = pd.DataFrame(results)
