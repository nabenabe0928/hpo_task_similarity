from task_similarity import IoUTaskSimilarity
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np


lb, ub, dim = -5, 5, 3
weights = np.ones(dim)


def func(X, shift):
    return (X - shift) ** 2 @ weights


if __name__ == "__main__":
    n_evals = 100

    config_space = CS.ConfigurationSpace()
    for d in range(dim):
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", -5, 5))

    observations_set = []
    for shift in range(6):
        X = np.random.random((n_evals, dim)) * (ub - lb) + lb
        ob = {f"x{d}": X[:, d] for d in range(dim)}
        ob["loss"] = func(X, shift=shift)
        observations_set.append(ob)

    ts = IoUTaskSimilarity(n_samples=1 << 10, config_space=config_space, observations_set=observations_set)
    print(ts.compute(task_pairs=[(i, j) for i in range(6) for j in range(6)]))
