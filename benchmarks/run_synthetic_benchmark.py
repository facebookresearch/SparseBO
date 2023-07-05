#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.storage.json_store.encoder import object_to_json
from ax.utils.measurement.synthetic_functions import hartmann6
from benchmark_gs import get_generation_strategy


def branin_augment(x_vec, augment_dim):
    assert len(x_vec) == augment_dim
    x1, x2 = (
        15 * x_vec[0] - 5,
        15 * x_vec[1],
    )  # Only dimensions 0 and augment_dim-1 affect the value of the function
    t1 = x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * np.cos(x1)
    return t1**2 + t2 + 10


def hartmann6_augment(x_vec, augment_dim):
    assert len(x_vec) == augment_dim
    return hartmann6.f(np.array(x_vec[:6]))


def nnz_exact(x: List[float], sparse_point: List[float]):
    return len(x) - (np.array(x) == np.array(sparse_point)).sum()


def run_single_objective_branin_benchmark(
    strategy_name: str,
    num_sobol_trials: int = 8,
    num_trials: int = 100,
    augment_dim: int = 10,
    strategies_args: Optional[Dict[str, Any]] = None,
) -> str:
    # set zero as the baseline to shrink towards
    sparse_point = [0 for _ in range(augment_dim)]
    gs = get_generation_strategy(
        strategy_name=strategy_name,
        sparse_point=sparse_point,
        num_sobol_trials=num_sobol_trials,
        strategies_args=strategies_args,
    )
    if "L0" in strategy_name:
        penalty_name = "L0_norm"
    else:
        penalty_name = "L1_norm"

    axc = AxClient(generation_strategy=gs)

    experiment_parameters = [
        {
            "name": f"parameter_{i}",
            "type": "range",
            "bounds": [0, 1],
            "value_type": "float",
            "log_scale": False,
        }
        for i in range(augment_dim)
    ]

    objective_metrics = {
        "objective": ObjectiveProperties(minimize=False),
    }
    if "MOO" in strategy_name:
        sparse_objective_threshold = strategies_args.get(
            "sparsity_threshold", augment_dim
        )
        objective_metrics = {
            # I previously use: -10
            "objective": ObjectiveProperties(minimize=False, threshold=-10),
            # I previously use: 15 * augment_dim
            penalty_name: ObjectiveProperties(
                minimize=True, threshold=sparse_objective_threshold
            ),
        }

    axc.create_experiment(
        name="sourcing_experiment",
        parameters=experiment_parameters,
        objectives=objective_metrics,
    )

    def evaluation(parameters):
        # put parameters into 1-D array
        x = [parameters.get(param["name"]) for param in experiment_parameters]
        res = branin_augment(x_vec=x, augment_dim=augment_dim)
        if penalty_name == "L0_norm":
            penalty_value = nnz_exact(x, sparse_point)
        else:
            penalty_value = np.linalg.norm(x, ord=1)
        eval_res = {
            # flip the sign to maximize
            "objective": (-res, 0.0),
            penalty_name: (penalty_value, 0.0),
        }
        return eval_res

    for _ in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        res = evaluation(parameters)
        axc.complete_trial(trial_index=trial_index, raw_data=res)

    res = json.dumps(object_to_json(axc.experiment))
    with open(f'results/synthetic_branin_{strategy_name}_rep_{irep}.json', "w") as fout:
       json.dump(res, fout)
    return res


def run_single_objective_branin_benchmark_reps(
    strategy: str,
    augment_dim: int = 10,
    num_sobol_trials: int = 8,
    num_trials: int = 50,
    reps: int = 20,
    strategy_args: Optional[Dict[str, Any]] = None,
):
    res = {strategy: []}

    for irep in range(reps):
        res[strategy].append(
            run_single_objective_branin_benchmark(
                strategy_name=strategy,
                num_sobol_trials=num_sobol_trials,
                num_trials=num_trials,
                augment_dim=augment_dim,
                strategies_args=strategy_args,
            )
        )
        with open(f'results/synthetic_branin_{strategy}.json', "w") as fout:
            json.dump(res, fout)


def run_single_objective_hartmann6_benchmark(
    strategy_name: str,
    num_sobol_trials: int = 8,
    num_trials: int = 100,
    augment_dim: int = 20,
    strategies_args: Optional[Dict[str, Any]] = None,
) -> str:
    # set zero as the baseline to shrink towards
    sparse_point = [0 for _ in range(augment_dim)]
    gs = get_generation_strategy(
        strategy_name=strategy_name,
        sparse_point=sparse_point,
        num_sobol_trials=num_sobol_trials,
        strategies_args=strategies_args,
    )
    if "L0" in strategy_name:
        penalty_name = "L0_norm"
    else:
        penalty_name = "L1_norm"

    axc = AxClient(generation_strategy=gs)

    experiment_parameters = [
        {
            "name": f"parameter_{i}",
            "type": "range",
            "bounds": [0, 1],
            "value_type": "float",
            "log_scale": False,
        }
        for i in range(augment_dim)
    ]

    objective_metrics = {
        "objective": ObjectiveProperties(minimize=False),
    }
    if "MOO" in strategy_name:
        sparse_penalty_threshold = strategies_args.get(
            "sparsity_threshold", augment_dim
        )
        objective_metrics = {
            "objective": ObjectiveProperties(minimize=False, threshold=0),
            penalty_name: ObjectiveProperties(
                minimize=True, threshold=sparse_penalty_threshold
            ),
        }

    axc.create_experiment(
        name="hartmann6_augment_experiment",
        parameters=experiment_parameters,
        objectives=objective_metrics,
    )

    def evaluation(parameters):
        # put parameters into 1-D array
        x = [parameters.get(param["name"]) for param in experiment_parameters]
        res = hartmann6_augment(x_vec=x, augment_dim=augment_dim)
        if penalty_name == "L0_norm":
            penalty_value = nnz_exact(x, sparse_point)
        else:
            penalty_value = np.linalg.norm(x, ord=1)
        eval_res = {
            # flip the sign to maximize
            "objective": (-res, 0.0),
            penalty_name: (penalty_value, 0.0),
        }
        return eval_res

    for _ in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        res = evaluation(parameters)
        axc.complete_trial(trial_index=trial_index, raw_data=res)

    res = json.dumps(object_to_json(axc.experiment))
    with open(f'results/synthetic_hartmann6_{strategy_name}_rep_{irep}.json', "w") as fout:
       json.dump(res, fout)
    return res


def run_single_objective_hartmann6_benchmark_reps(
    strategy: str,
    augment_dim: int = 10,
    num_sobol_trials: int = 8,
    num_trials: int = 50,
    reps: int = 20,
    strategy_args: Optional[Dict[str, Any]] = None,
):
    res = {strategy: []}

    for irep in range(reps):
        res[strategy].append(
            run_single_objective_hartmann6_benchmark(
                strategy_name=strategy,
                num_sobol_trials=num_sobol_trials,
                num_trials=num_trials,
                augment_dim=augment_dim,
                strategies_args=strategy_args,
            )
        )
        with open(f'results/synthetic_hartmann6_{strategy}.json', "w") as fout:
            json.dump(res, fout)


if __name__ == '__main__':
    # Run all of the benchmark replicates.

    run_single_objective_branin_benchmark_reps(
        strategy="Sobol",
        augment_dim=10,
        num_sobol_trials=8,
        num_trials=20,
        reps=1,
    )
