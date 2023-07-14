#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.storage.json_store.encoder import object_to_json
from benchmark_gs import get_generation_strategy
from synthetic_benchmark import nnz_exact
from sklearn.svm import SVR
from xgboost import XGBRegressor


class SVRObjective:
    def __init__(self, n_features):
        self.train_x, self.train_y, self.test_x, self.test_y = load_uci_data(
            seed=0, n_features=n_features
        )
        self.x_dim = self.train_x.shape[-1]
        self.dimension = self.x_dim + 3
        self.lb = np.zeros(self.dimension)
        self.ub = np.ones(self.dimension)

    def __call__(self, x):
        assert x.shape == (self.x_dim + 3,)
        assert (x >= self.lb).all() and (x <= self.ub).all()
        lengthscales = 1 * x[: self.x_dim]
        epsilon = 0.01 * 10 ** (2 * x[-3])  # Default = 0.1
        C = 0.01 * 10 ** (4 * x[-2])  # Default = 1.0
        gamma = (1 / self.x_dim) * 0.1 * 10 ** (2 * x[-1])  # Default = 1.0 / self.x_dim

        model = SVR(
            C=C, epsilon=epsilon, gamma=gamma, tol=0.001, cache_size=1000, verbose=True
        )
        model.fit(self.train_x * lengthscales, self.train_y.copy())
        pred = model.predict(self.test_x * lengthscales)
        mse = ((pred - self.test_y) ** 2).mean(axis=0)
        return -1 * math.sqrt(mse)


def load_uci_data(seed, n_features):
    df = pd.read_csv("slice_localization_data.csv", index_col=[0])
    print(df.head())
    data = df.to_numpy()

    # Get the input data
    X = data[:, :-1]
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = data[:, -1]
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[
        :10000
    ]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    # Use Xgboost to figure out feature importances and keep only the most important features
    xgb = XGBRegressor(max_depth=8).fit(X, y)
    inds = (-xgb.feature_importances_).argsort()
    X = X[:, inds[:n_features]]

    # Train/Test split
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y


def run_single_objective_svm_benchmark(
    strategy_name: str,
    irep: int,
    num_sobol_trials: int = 8,
    num_trials: int = 100,
    augment_dim: int = 20,
    strategies_args: Optional[Dict[str, Any]] = None,
) -> str:

    # set zero as the baseline to shrink towards
    sparse_point = [0 for _ in range(augment_dim)] + [0.5, 0.5, 0.5]
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

    svr = SVRObjective(n_features=augment_dim)

    axc = AxClient(generation_strategy=gs)

    experiment_parameters = [
        {
            "name": f"svm_parameter_{i}",
            "type": "range",
            "bounds": [0, 1],
            "value_type": "float",
            "log_scale": False,
        }
        for i in range(augment_dim + 3)
    ]

    objective_metrics = {
        "objective": ObjectiveProperties(minimize=False),
    }
    if "MOO" in strategy_name:
        penalty_th = strategies_args.get("sparsity_threshold", augment_dim)
        objective_metrics = {
            "objective": ObjectiveProperties(minimize=False, threshold=-1),
            penalty_name: ObjectiveProperties(minimize=True, threshold=penalty_th),
        }

    axc.create_experiment(
        name="svm_experiment",
        parameters=experiment_parameters,
        objectives=objective_metrics,
    )

    def evaluation(parameters):
        # put parameters into 1-D array
        x = [parameters.get(param["name"]) for param in experiment_parameters]
        res = svr(np.array(x))
        if penalty_name == "L0_norm":
            penalty_value = nnz_exact(x, sparse_point)
        else:
            penalty_value = np.linalg.norm(x, ord=1)
        eval_res = {
            # The sign is fliped in svr to maximize
            "objective": (res, 0.0),
            penalty_name: (penalty_value, 0.0),
        }
        return eval_res

    for _ in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        res = evaluation(parameters)
        axc.complete_trial(trial_index=trial_index, raw_data=res)

    res = json.dumps(object_to_json(axc.experiment))
    with open(f'results/svm_{strategy_name}_rep_{irep}.json', "w") as fout:
       json.dump(res, fout)
    return res


def run_single_objective_svm_benchmark_reps(
    strategy: str,
    augment_dim: int = 100,
    num_sobol_trials: int = 20,
    num_trials: int = 100,
    reps: int = 20,
    strategy_args: Optional[Dict[str, Any]] = None,
):
    res = {strategy: []}

    for irep in range(reps):
        res[strategy].append(
            run_single_objective_svm_benchmark(
                strategy_name=strategy,
                irep=irep,
                num_sobol_trials=num_sobol_trials,
                num_trials=num_trials,
                augment_dim=augment_dim,
                strategies_args=strategy_args,
            )
        )
        with open(f'results/svm_{strategy}.json', "w") as fout:
            json.dump(res, fout)


if __name__ == '__main__':
    # Run all of the benchmark replicates.
    run_single_objective_svm_benchmark_reps(
        strategy="Sobol",
        augment_dim=20,
        num_sobol_trials=8,
        num_trials=20,
        reps=1,
        strategy_args={},
    )
