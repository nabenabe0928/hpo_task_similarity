from typing import Dict, List, Tuple

import ConfigSpace as CS

import numpy as np

from parzen_estimator import MultiVariateParzenEstimator, ParzenEstimatorType


def compute_importance(
    promising_pdfs: List[MultiVariateParzenEstimator],
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:

    pdf_vals_dict: Dict[str, List[np.ndarray]] = {hp_name: [] for hp_name in promising_pdfs[0].param_names}
    for pdf in promising_pdfs:
        samples = pdf.uniform_sample(n_samples=1 << 8, rng=rng, return_dict=True)
        pdf_vals = pdf.dimension_wise_pdf(X=samples, return_dict=True)
        for hp_name, pdf_val in pdf_vals.items():
            pdf_vals_dict[hp_name].append(pdf_val)

    task_wise_hp_importance: Dict[str, np.ndarray] = {}
    for hp_name, pdf_vals in pdf_vals_dict.items():
        L = promising_pdfs[0][hp_name].domain_size
        imp = np.mean((np.asarray(pdf_vals) - 1 / L) ** 2, axis=-1) * (L**2)
        task_wise_hp_importance[hp_name] = imp

    return task_wise_hp_importance


def reduce_dimension(
    hp_importance: Dict[str, float],
    dim_after: int,
    config_space: CS.ConfigurationSpace,
    promising_pdfs: List[MultiVariateParzenEstimator],
) -> Tuple[List[MultiVariateParzenEstimator], CS.ConfigurationSpace]:

    dim = len(hp_importance)
    if dim_after == dim:
        return promising_pdfs, config_space
    if dim_after == 0:
        return [], CS.ConfigurationSpace()

    if dim_after < 0 or dim_after > dim:
        raise ValueError(f"dim_after must be in [0, dim={dim}], but got {dim_after}")

    new_config_space = CS.ConfigurationSpace()
    n_tasks = len(promising_pdfs)
    hp_importance = {k: v for k, v in sorted(hp_importance.items(), key=lambda item: -item[1])}
    parzen_estimators_list: List[Dict[str, ParzenEstimatorType]] = [{} for _ in range(n_tasks)]

    for d, hp_name in enumerate(hp_importance.keys()):
        if d >= dim_after:
            break

        new_config_space.add_hyperparameter(config_space.get_hyperparameter(hp_name))
        for i in range(n_tasks):
            parzen_estimators_list[i][hp_name] = promising_pdfs[i][hp_name]

    new_promising_pdfs = [
        MultiVariateParzenEstimator(parzen_estimators) for parzen_estimators in parzen_estimators_list
    ]

    return new_promising_pdfs, new_config_space
