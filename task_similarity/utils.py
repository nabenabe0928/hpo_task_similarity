from typing import Dict, List, Literal, Optional

import ConfigSpace as CS

from fast_pareto import nondominated_rank

import numpy as np

from parzen_estimator import MultiVariateParzenEstimator, get_multivar_pdf, over_resample

from task_similarity.constants import _IoUTaskSimilarityParameters


def _calculate_order(
    observations: Dict[str, np.ndarray],
    objective_names: List[str],
    larger_is_better_objectives: Optional[List[int]],
    # TODO: Make it possible to change the method from outside
    tie_break_method: Literal["crowding_distance", "avg_rank"] = "crowding_distance",
) -> np.ndarray:
    if len(objective_names) == 1:
        _sign = 1 if larger_is_better_objectives is None else -1
        order = np.argsort(_sign * observations[objective_names[0]])
    else:
        costs = np.array([observations[name] for name in objective_names]).T
        ranks = nondominated_rank(
            costs=costs, larger_is_better_objectives=larger_is_better_objectives, tie_break=tie_break_method
        )
        order = np.argsort(ranks)

    return order


def _get_promising_pdf(
    observations: Dict[str, np.ndarray],
    params: _IoUTaskSimilarityParameters,
) -> MultiVariateParzenEstimator:
    hp_names = params.config_space.get_hyperparameter_names()
    n_observations = observations[params.objective_names[0]].size
    n_promisings = max(1, int(params.promising_quantile * n_observations))
    order = _calculate_order(
        observations=observations,
        objective_names=params.objective_names,
        larger_is_better_objectives=params.larger_is_better_objectives,
    )

    promising_indices = order[:n_promisings]
    promising_configs = {}
    for hp_name in hp_names:
        promising_configs[hp_name] = observations[hp_name][promising_indices]

    if params.n_resamples is None:
        return get_multivar_pdf(
            observations=promising_configs,
            config_space=params.config_space,
            default_min_bandwidth_factor=params.default_min_bandwidth_factor,
            prior=False,
        )
    else:
        return over_resample(
            config_space=params.config_space,
            observations=promising_configs,
            n_resamples=params.n_resamples,
            rng=params.rng,
            default_min_bandwidth_factor=params.default_min_bandwidth_factor,
        )


def _get_promising_pdfs(
    observations_set: List[Dict[str, np.ndarray]],
    params: _IoUTaskSimilarityParameters,
) -> List[MultiVariateParzenEstimator]:
    """
    Get the promising distributions for each task.

    Args:
        observations_set (List[Dict[str, np.ndarray]]):
            The observations for each task.
        params (IoUTaskSimilarityParameters):
            The parameters for the task similarity measure class.

    Returns:
        promising_pdfs (List[MultiVariateParzenEstimator]):
            The list of the promising distributions of each task.
            The shape is (n_tasks, ).
    """
    promising_pdfs: List[MultiVariateParzenEstimator] = []
    for observations in observations_set:
        promising_pdfs.append(_get_promising_pdf(observations=observations, params=params))

    return promising_pdfs


def _get_hypervolume(config_space: CS.ConfigurationSpace) -> float:
    """
    Compute the hypervolumen given the config space.

    Args:
        config_space (CS.ConfigurationSpace):
            The configuration space for the parzen estimator.

    Returns:
        hypervolume (float):
            The hypervolume of the config space.
    """
    hp_names = config_space.get_hyperparameter_names()
    hv = 1.0
    for hp_name in hp_names:
        config = config_space.get_hyperparameter(hp_name)
        config_type = config.__class__.__name__
        if config_type.startswith("Categorical"):
            hv *= len(config.choices)
        elif config_type.startswith("Ordinal"):
            hv *= config.meta["upper"] - config.meta["lower"]
        else:
            hv *= config.upper - config.lower

    return hv
