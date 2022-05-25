from task_similarity import IoUTaskSimilarity
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

import matplotlib.pyplot as plt


lb, ub, shift = -5, 5, 0


def func1(X):
    return np.sum(X**2, axis=-1)


def func2(X):
    return np.sum((X - shift) ** 2, axis=-1)


def visualize_overlap(ts: IoUTaskSimilarity) -> None:
    sim = ts.compute(task_pairs=[(0, 1)])[0, 1]
    n_grids = 100
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 8))

    for i in range(2):
        ax = axes[i]
        pe = ts._parzen_estimators[i]
        promising_indices = ts._promising_indices[i]
        X, Y = np.meshgrid(np.linspace(lb, ub, n_grids), np.linspace(lb, ub, n_grids))
        X, Y = X.reshape(n_grids ** 2), Y.reshape(n_grids ** 2)
        Z = pe.pdf([X, Y])

        shape_ = (n_grids, n_grids)
        X, Y, Z = X.reshape(shape_), Y.reshape(shape_), Z.reshape(shape_)
        ax.contourf(X, Y, Z)
        ax.scatter(ts._samples[0][promising_indices], ts._samples[1][promising_indices], color="red", alpha=0.05)

        idx1, idx2 = ts._promising_indices[i], ts._promising_indices[- i - 1]
        indices_both = promising_indices[np.in1d(idx1, idx2, assume_unique=True)]
        ax.scatter(ts._samples[0][indices_both], ts._samples[1][indices_both], color="blue", alpha=0.05)

    plt.suptitle(f"Task similarity: {sim:.4f}")
    plt.show()


if __name__ == "__main__":
    n_evals, dim = 100, 2

    config_space = CS.ConfigurationSpace()
    for d in range(dim):
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", -5, 5))

    X1 = np.random.random((n_evals, dim)) * (ub - lb) + lb
    X2 = np.random.random((n_evals, dim)) * (ub - lb) + lb

    ob1 = {f"x{d}": X1[:, d] for d in range(dim)}
    ob1["loss"] = func1(X1)
    ob2 = {f"x{d}": X2[:, d] for d in range(dim)}
    ob2["loss"] = func2(X2)
    ts = IoUTaskSimilarity(n_samples=1 << 14, config_space=config_space, observations_set=[ob1, ob2])
    visualize_overlap(ts)
