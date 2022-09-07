from typing import Dict

from task_similarity import IoUTaskSimilarity
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import os
import matplotlib.pyplot as plt
import numpy as np


weights = 1.0 / 10 ** np.arange(30)
SHIFTS = [0, 0, 1, 2, 3]


def func(X: np.ndarray, shift: int):
    dim = X.shape[-1]
    return (X - shift) ** 2 @ weights[:dim]


def get_observations(shift: int, dim: int, rad: int, n_evals: int, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
    X = rng.random((n_evals, dim)) * (2 * rad) - rad
    ob = {f"x{d}": X[:, d] for d in range(dim)}
    ob["loss"] = func(X, shift=shift)
    return ob


def main(dim: int, rad: int, seed: int, max_dim: int, n_evals: int) -> np.ndarray:
    config_space = CS.ConfigurationSpace()
    for d in range(dim):
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", -rad, rad))

    observations_set = []
    rng = np.random.RandomState(seed)
    for shift in SHIFTS:
        observations_set.append(get_observations(shift, dim=dim, rad=rad, n_evals=n_evals, rng=rng))

    ts = IoUTaskSimilarity(
        n_samples=1 << 10,
        config_space=config_space,
        observations_set=observations_set,
        # max_dim=max_dim,
        promising_quantile=0.15,
        default_min_bandwidth_factor=1e-2,
        rng=np.random.RandomState(seed),
    )
    return ts.compute(method="total_variation", task_pairs=[(0, i) for i in range(1, len(SHIFTS))])[0][1:]


if __name__ == "__main__":
    """
    Figure 1. (separate by shift)
        x axis: max_dim
        y axis: similarity
        plot: max_dim
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font

    R, D, S = 5, 20, 10
    n_evals = 20
    means = {shift: np.zeros(D) for shift in SHIFTS[1:]}
    stes = {shift: np.zeros(D) for shift in SHIFTS[1:]}
    for max_dim in range(1, D + 1):
        print(f"max_dim: {max_dim}")
        data = np.zeros((len(SHIFTS) - 1, S))
        for seed in range(S):
            data[:, seed] += main(dim=D, rad=R, seed=seed, max_dim=max_dim, n_evals=n_evals)

        print(data.mean(axis=-1))
        for i, shift in enumerate(SHIFTS[1:]):
            index = max_dim - 1 if max_dim is not None else -1
            means[shift][index] = data[i].mean()
            stes[shift][index] = data[i].std() / np.sqrt(S)

    fig, ax = plt.subplots(figsize=(12, 6))
    dx = np.arange(1, D + 1)
    ax.set_xlabel("Dimension $d^\\prime$ after the reduction")
    ax.set_ylabel("Similarity $\\hat{s}$")
    ax.set_ylim(0, 1)
    colors = ["orange", "red", "purple", "blue"]
    for i, (shift, color) in enumerate(zip(SHIFTS[1:], colors)):
        m, s = means[shift], stes[shift]
        print(m - s, m, m + s)
        ax.plot(dx, m, label=f"Shift = {shift}", color=color)
        ax.fill_between(dx, m - s, m + s, alpha=0.2, color=color)

    ax.grid()
    ax.legend()

    os.makedirs("figs/", exist_ok=True)
    # plt.savefig("figs/dimension_reduction_demo.pdf", bbox_inches='tight')
    plt.show()
