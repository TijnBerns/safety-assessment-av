"""
Module used for downloading and preprocssing UCI data (Power, Hepmass, Miniboone, and Gas)
"""

import sys

sys.path.append("src")

import os
import utils
import wget
import click
import numpy as np
from pathlib import Path
import tarfile

from base import CustomDataset, split_data
from miniboone import MiniBoone
from power import Power
from gas import Gas
from hepmass import Hepmass

DOWNLOAD_URL = "https://zenodo.org/record/1161203/files/data.tar.gz?download=1"


def download_data():
    def members_start(member: tarfile.TarInfo):
        return (
            member.name.startswith(("data/miniboone"))
            or member.name.startswith(("data/hepmass"))
            or member.name.startswith(("data/gas"))
            or member.name.startswith(("data/power"))
        )

    def get_members(tar: tarfile.TarFile):
        l = len("data/")
        for member in tar.getmembers():
            if members_start(member):
                member.path = member.path[l:]
                yield member

    try:
        # Get datapath from os environment
        path = Path(os.environ["DATAROOT"])
        path.mkdir(parents=True, exist_ok=True)

        # Download data
        print(f"Downloading data to {path}...")
        wget.download(DOWNLOAD_URL, out=str(path))

        # Extract tar file
        print("Extracting tar files...")
        ar = tarfile.open(path / "data.tar.gz")
        ar.extractall(path=str(path), members=get_members(ar))
        ar.close()
    except KeyError:
        print(
            "Could not download data, make sure the DATAROOT environment variable is set via export."
        )


def select_variable(data):
    corr = np.abs(np.corrcoef(data.T))
    xi = np.argmax(np.mean(corr, axis=0))
    return xi, corr


def set_threshold(data, xi):
    p_event = 0.08
    threshold = np.percentile(data[:, xi], 100 - (100 * p_event))
    return threshold


def normalize(data: np.ndarray, mu=None, std=None):
    if mu is None or std is None:
        mu = data.mean(axis=0)
        std = data.std(axis=0)

    normalized_data = (data - mu) / std
    return normalized_data, mu, std


def compute_event_weight(normal, event, xi, threshold):
    n_non_event = np.sum(normal[:, xi] <= threshold)
    n_event = len(event) + len(normal) - n_non_event
    p_event = (len(normal) - n_non_event) / len(normal)
    return ((p_event * n_non_event) / (n_event)) / (1 - p_event)


def load_data(dataset):
    train, test = dataset.load_data()
    train, val = split_data(train)
    return train, val, test


def save_splits(dataset: CustomDataset):
    # 1. Split data
    train, val, test = load_data(dataset)
    test_copy = np.array(test)
    val_copy = np.array(val)
    all_train = np.vstack((train, val))
    all_data = np.vstack((all_train, test))

    # 2. Select variable on which threshold is set
    # This is set on the attribute with the highest mean correlation wrt all other variables
    xi, corr = select_variable(all_data)

    # 3. Set the threshold for p_event
    threshold_unnormalized = set_threshold(all_data, xi)

    # 4. Split train data into normal and event
    train_normal, train_event = split_data(train, frac=0.2)
    train_event = train_event[train_event[:, xi] > threshold_unnormalized]
    test_normal = test[test[:, xi] <= threshold_unnormalized]
    test_event = test[test[:, xi] > threshold_unnormalized]
    utils.save_np(dataset.root / "normal_train_unnormalized.npy", train_normal)
    utils.save_np(dataset.root / "event_train_unnormalized.npy", train_event)

    # 5. Normalize data
    train_normal, mu, std = normalize(train_normal)
    train_event = normalize(train_event, mu, std)[0]
    val = normalize(val, mu, std)[0]
    test = normalize(test, mu, std)[0]
    test_normal = normalize(test_normal, mu, std)[0]
    test_event = normalize(test_event, mu, std)[0]
    threshold = (threshold_unnormalized - mu[xi]) / std[xi]

    # To reproduce results in normal spline flow paper also store entire train set
    _, mu, std = normalize(all_train)
    train_all = normalize(train, mu, std)[0]
    val_all = normalize(val_copy, mu, std)[0]
    test_all = normalize(test_copy, mu, std)[0]
    _threshold = (threshold_unnormalized - mu[xi]) / std[xi]

    # 7. Save all datasplits
    # Train splits
    utils.save_np(dataset.root / "_train.npy", train_all)
    utils.save_np(dataset.root / "normal_train.npy", train_normal)
    utils.save_np(dataset.root / "event_train.npy", train_event)

    # Validation splits
    utils.save_np(dataset.root / "_val.npy", val_all)
    utils.save_np(dataset.root / "val.npy", val)
    # save_np(dataset.root / 'normal_val.npy', normal_val)
    # save_np(dataset.root / 'event_val.npy', event_val)

    # Test splits
    utils.save_np(dataset.root / "_test.npy", test_all)
    utils.save_np(dataset.root / "test.npy", test)
    utils.save_np(dataset.root / "test_normal.npy", test_normal)
    utils.save_np(dataset.root / "test_event.npy", test_event)

    # 7. Save stats of splits
    stats = {
        "root": str(dataset.root),
        "_train.npy": len(train_all),
        "normal_train.npy": len(train_normal),
        "event_train.npy": len(train_event),
        "val.npy": len(val),
        "_test.npy": len(test_all),
        "test.npy": len(test),
        "test_normal.npy": len(test_normal),
        "test_event.npy": len(test_event),
        "attributes": train_all.shape,
        "mu": mu.tolist(),
        "std": std.tolist(),
        "Xi": int(xi),
        "corr": np.max(np.mean(corr, axis=0)),
        "threshold": threshold,
        "_threshold": _threshold,
        "weight": compute_event_weight(train_normal, train_event, xi, threshold),
    }
    utils.save_json(dataset.root / "stats.json", stats)


@click.command()
@click.option("--download", type=bool, default=False)
def main(download: bool):
    utils.seed_all(2023)
    if download:
        download_data()

    print("Preprocessing: MiniBoone")
    save_splits(MiniBoone())

    print("Preprocessing: Power")
    save_splits(Power())

    print("Preprocessing: Gas")
    save_splits(Gas())

    print("Preprocessing: Hepmass")
    save_splits(Hepmass())


if __name__ == "__main__":
    main()
