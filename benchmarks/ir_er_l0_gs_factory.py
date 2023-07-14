#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from regularized_bo import ExternalRegularizedL0, InternalRegularizedL0
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.models import FixedNoiseGP, SaasFullyBayesianSingleTaskGP
from baseline_gs_factory import TORCH_DEVICE
from torch import Tensor


def get_ir_l0_gs(
    sparse_point: Tensor,
    num_sobol_trials: int,
    gp_model_name: str,
    regularization_parameter: float,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
):
    if gp_model_name == "SAAS":
        surrogate = Surrogate(SaasFullyBayesianSingleTaskGP)
    elif gp_model_name == "GP":
        surrogate = Surrogate(botorch_model_class=FixedNoiseGP)

    gs = GenerationStrategy(
        name=f"L0_internal_{gp_model_name}",
        steps=[
            GenerationStep(  # Initialization step
                model=Models.SOBOL,
                num_trials=num_sobol_trials,
            ),
            GenerationStep(  # BayesOpt step
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                model_kwargs={
                    "surrogate": surrogate,
                    "acquisition_class": InternalRegularizedL0,
                    "botorch_acqf_class": qNoisyExpectedImprovement,
                    "torch_device": torch_device,
                    "acquisition_options": {
                        "target_point": torch.tensor(
                            sparse_point, device=torch_device, dtype=torch.double
                        ),
                        "regularization_parameter": regularization_parameter,
                    },
                },
            ),
        ],
    )
    return gs


def get_er_l0_gs(
    sparse_point: Tensor,
    num_sobol_trials: int,
    gp_model_name: str,
    regularization_parameter: float,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
):
    if gp_model_name == "SAAS":
        surrogate = Surrogate(SaasFullyBayesianSingleTaskGP)
    elif gp_model_name == "GP":
        surrogate = Surrogate(botorch_model_class=FixedNoiseGP)

    gs = GenerationStrategy(
        name=f"L0_external_{gp_model_name}",
        steps=[
            GenerationStep(  # Initialization step
                model=Models.SOBOL,
                num_trials=num_sobol_trials,
            ),
            GenerationStep(  # BayesOpt step
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                model_kwargs={
                    "surrogate": surrogate,
                    "acquisition_class": ExternalRegularizedL0,
                    "botorch_acqf_class": qNoisyExpectedImprovement,
                    "torch_device": torch_device,
                    "acquisition_options": {
                        "target_point": torch.tensor(
                            sparse_point, device=torch_device, dtype=torch.double
                        ),
                        "regularization_parameter": regularization_parameter,
                    },
                },
            ),
        ],
    )
    return gs
