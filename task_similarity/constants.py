from typing import List, NamedTuple, Optional

import ConfigSpace as CS

import numpy as np


class _IoUTaskSimilarityParameters(NamedTuple):
    """
    Args:
        config_space (CS.ConfigurationSpace):
            The configuration space for the parzen estimator.
        objective_names (List[str]):
            The names of the objective metrics.
        promising_quantile (float):
            The quantile of the promising configs.
        default_min_bandwidth_factor (float):
            The factor of min bandwidth.
            For example, when we take 0.1, the bandwidth will be larger
            than 0.1 * (ub - lb).
        larger_is_better_objectives (Optional[List[int]]):
            The indices of the objectives that are better when larger.
        rng (np.random.RandomState):
            The random number generator.
        n_resamples (Optional[int]):
            How many resamplings we use for the parzen estimator.
            If None, we do not use resampling.
        dim_reduction_factor (float):
            eta in the paper.
            This parameter controls the number of dimensions to reduce.
    """

    n_samples: int
    config_space: CS.ConfigurationSpace
    promising_quantile: float
    rng: np.random.RandomState
    objective_names: List[str]
    default_min_bandwidth_factor: float
    dim_reduction_factor: float
    larger_is_better_objectives: Optional[List[int]]
    n_resamples: Optional[int]
