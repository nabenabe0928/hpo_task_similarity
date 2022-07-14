from typing import Dict

from task_similarity import IoUTaskSimilarity
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np


lb, ub, dim = -5, 5, 20
weights = 1.0 / 10 ** np.arange(dim)
rng = np.random.RandomState(0)


def func(X, shift):
    return (X - shift) ** 2 @ weights


def get_observations(shift: int) -> Dict[str, np.ndarray]:
    X = rng.random((n_evals, dim)) * (ub - lb) + lb
    ob = {f"x{d}": X[:, d] for d in range(dim)}
    ob["loss"] = func(X, shift=shift)
    return ob


if __name__ == "__main__":
    n_evals = 1000

    config_space = CS.ConfigurationSpace()
    for d in range(dim):
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", -5, 5))

    observations_set = []
    shifts = [0, 0, 1, 2, 3, 4]
    for shift in shifts:
        observations_set.append(get_observations(shift))

    ts = IoUTaskSimilarity(
        n_samples=1 << 14,
        config_space=config_space,
        observations_set=observations_set,
        dim_reduction_rate=0.8,
        max_dim=3,
        promising_quantile=0.15,
        rng=np.random.RandomState(0)
    )
    print(ts.compute(method="total_variation", task_pairs=[(0, i) for i in range(1, len(shifts))]))
