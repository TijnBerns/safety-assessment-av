import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from src.kde.parameters import UVParameters as uv_params
import data.data_utils as data_utils
from itertools import product
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from tqdm import tqdm

import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

LABELS = {
    "num_norm": "$N_\\textrm{norm}$",
    "num_event": "$N_\\textrm{event}$",
    "p_event": "$p_\\textrm{event}$",
    "correlation": "$\\rho$",
}

FIGSIZE = (3.0, 1.8)


def main():
    correlation = list(np.arange(0.1, 1, 0.1))

    parameter_sets = list(
        product(
            uv_params.distributions,
            uv_params.num_normal,
            uv_params.num_event,
            uv_params.p_event,
            correlation,
        )
    )
    results = []

    print(f"Total number of sets: {len(parameter_sets)}")
    for i, (distribution_str, num_normal, num_event, p_event, correlation) in tqdm(
        enumerate(parameter_sets)
    ):
        s_distribution, mv_distribution = uv_params.get_distributions(
            distribution_str, correlation
        )
        x_values = data_utils.get_evaluation_interval(s_distribution)
        threshold = data_utils.determine_threshold_analytical(frac_edge=p_event)

        n = 10
        sse_norm = 0
        sse_comb = 0
        for _ in range(n):
            # Generate data
            normal_data, event_data = data_utils.generate_data(
                mv_distribution, num_normal, num_event, threshold
            )
            combined_data = data_utils.combine_data(
                normal_data, event_data, threshold, p_event
            )

            # Construct CDFs
            cdf_norm = data_utils.EmpericalCDF()
            cdf_norm.fit(normal_data[:, 0])
            cdf_norm = cdf_norm.evaluate(x_values)
            cdf_comb = data_utils.EmpericalCDF()
            cdf_comb.fit(combined_data[:, 0])
            cdf_comb = cdf_comb.evaluate(x_values)
            cdf_true = s_distribution.cdf(x_values)

            # Evaluate CDFs
            sse_norm += np.sum(np.square(cdf_true - cdf_norm))
            sse_comb += np.sum(np.square(cdf_true - cdf_comb))

        sse_norm /= n
        sse_comb /= n

        res = {
            "distribution": distribution_str,
            "num_norm": num_normal,
            "num_event": num_event,
            "p_event": p_event,
            "correlation": str(round(correlation, 2)),
            "sse_norm": sse_norm,
            "sse_comb": sse_comb,
            "improvement": sse_norm - sse_comb,
        }

        pprint(res)

        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False, lineterminator="\n", sep=",")


def plot(path, groupby: str):
    df = pd.read_csv(path)
    df = df.where(df["correlation"] != 0).dropna()

    mean = df.groupby(groupby)["improvement"].mean()
    # std = df.groupby(groupby)['improvement'].std()
    lower = df.groupby(groupby)["improvement"].quantile(0.025)
    upper = df.groupby(groupby)["improvement"].quantile(0.975)
    x_ticks = df[groupby].unique()

    # Create plot
    plt.figure(figsize=FIGSIZE)
    plt.plot(mean, label="mean")
    plt.fill_between(
        mean.keys(),
        lower.values,
        upper.values,
        alpha=0.25,
        label="95 confidence interval",
    )
    # x_ticks = [x_ticks[0], np.median(x_ticks), x_ticks[-1]]
    plt.xticks(x_ticks, x_ticks)
    plt.xlabel(LABELS[groupby])
    plt.ylabel("improvement")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/data_combination_{groupby}.pgf")

    # Print correlation
    corrcoef = np.corrcoef(df["improvement"], df[groupby])
    print(f"Correlation ({groupby}): {corrcoef[0][1]}")


if __name__ == "__main__":
    # main()
    plot("results.csv", "correlation")
    plot("results.csv", "p_event")
    plot("results.csv", "num_norm")
    plot("results.csv", "num_event")
