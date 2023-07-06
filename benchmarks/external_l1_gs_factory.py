#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Optional, Tuple

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_defaults import _get_acquisition_func
from ax.models.torch.fully_bayesian import get_fully_bayesian_acqf
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.penalized import L1Penalty, PenalizedAcquisitionFunction
from botorch.models.model import Model
from baseline_gs_factory import TORCH_DEVICE
from torch import Tensor


def get_er_l1_saas_gs(
    num_sobol_trials: int,
    sparse_point: Tensor,
    regularization_parameter: float,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
) -> GenerationStrategy:
    gs = GenerationStrategy(
        name="L1_external_SAAS",
        steps=[
            GenerationStep(model=get_sobol, num_trials=num_sobol_trials),
            GenerationStep(
                model=functools.partial(
                    get_L1_SAAS_external_penalized,
                    sparse_point=sparse_point,
                    regularization_parameter=regularization_parameter,
                ),
                num_trials=-1,
                model_kwargs={"torch_device": torch_device},
            ),
        ],
    )
    return gs


def get_er_l1_gp_gs(
    num_sobol_trials: int,
    sparse_point: Tensor,
    regularization_parameter: float,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
) -> GenerationStrategy:
    gs = GenerationStrategy(
        name="L1_external_GP",
        steps=[
            GenerationStep(model=get_sobol, num_trials=num_sobol_trials),
            GenerationStep(
                model=functools.partial(
                    get_L1_external_penalized,
                    sparse_point=sparse_point,
                    regularization_parameter=regularization_parameter,
                ),
                num_trials=-1,
                model_kwargs={"torch_device": torch_device},
            ),
        ],
    )
    return gs


def get_L1_SAAS_external_penalized(
    experiment: Experiment,
    data: Data,
    sparse_point: Tensor,
    regularization_parameter: Optional[float] = 0.01,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
) -> TorchModelBridge:
    """Instantiates a model using SAAS GP with ER-L1 ACQF for single objective optimization."""
    return Models.FULLYBAYESIAN(
        experiment=experiment,
        search_space=experiment.search_space,
        data=data,
        acqf_constructor=get_fully_bayesian_acqf_nei_l1_external,
        num_samples=256,
        warmup_steps=512,
        disable_progbar=True,
        transforms=Cont_X_trans + Y_trans,
        default_model_gen_options={
            "acquisition_function_kwargs": {
                "chebyshev_scalarization": False,
                "sequential": True,
                "regularization_parameter": regularization_parameter,
                "sparse_point": sparse_point,
            },
        },
        torch_dtype=torch.double,
        torch_device=torch_device,
    )


def get_L1_external_penalized(
    experiment: Experiment,
    data: Data,
    sparse_point: Tensor,  # TODO: make this ObservationFeatures
    regularization_parameter: float = 0.01,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
) -> TorchModelBridge:
    """Instantiates a model using standard GP with ER-L1 ACQF for single objective optimization."""
    return Models.BOTORCH(
        acqf_constructor=get_NEI_external_L1_penalized,
        default_model_gen_options={
            "acquisition_function_kwargs": {
                "chebyshev_scalarization": False,
                "sequential": True,
                "regularization_parameter": regularization_parameter,
                "sparse_point": sparse_point,
            },
        },
        experiment=experiment,
        data=data,
        search_space=experiment.search_space,
        transforms=Cont_X_trans + Y_trans,
        torch_dtype=torch.double,
        torch_device=torch_device,
    )


def get_fully_bayesian_acqf_nei_l1_external(
    model: Model,
    objective_weights: Tensor,
    sparse_point: Tensor,
    regularization_parameter: float,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    r"""Instantiates a penalized qNoisyExpectedImprovement acquisition function,
    in which the penalty term is a L1 penalty added externally to the ACQF."""
    # default acqf_constructor is get_NEI
    raw_acqf = get_fully_bayesian_acqf(
        model=model,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        X_observed=X_observed,
        X_pending=X_pending,
        **kwargs,
    )
    penalty_func = L1Penalty(init_point=sparse_point)
    acqf = PenalizedAcquisitionFunction(
        raw_acqf=raw_acqf,
        penalty_func=penalty_func,
        regularization_parameter=regularization_parameter,
    )
    return acqf


def get_NEI_external_L1_penalized(
    model: Model,
    objective_weights: Tensor,
    sparse_point: Tensor,
    regularization_parameter: float,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    r"""Instantiates a PenalizedAcquisitionFunction that adds L1 penalty
    to qNoisyExpectedImprovement acquisition function. External means the L1 penalty
    is added outside of the NEI acuqisition function."""
    # this is the same function as get_NEI_external_L1_penalized in ax.fb.models.torch.parego_l1_penalized
    raw_acqf = _get_acquisition_func(
        model=model,
        acquisition_function_name="qNEI",
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        X_observed=X_observed,
        X_pending=X_pending,
        **kwargs,
    )
    penalty_func = L1Penalty(init_point=sparse_point)
    acqf = PenalizedAcquisitionFunction(
        raw_acqf=raw_acqf,
        penalty_func=penalty_func,
        regularization_parameter=regularization_parameter,
    )
    return acqf
