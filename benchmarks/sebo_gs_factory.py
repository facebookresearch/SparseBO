#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
# from ax.models.torch.botorch_modular.sebo import SEBOAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models import FixedNoiseGP, SaasFullyBayesianSingleTaskGP
from baseline_gs_factory import TORCH_DEVICE
from torch import Tensor


def get_sebo_gs(
    sparse_point: Tensor,
    penalty_name: str,
    num_sobol_trials: int,
    gp_model_name: str,
    sparsity_threshold: Optional[float] = None,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
):
    if gp_model_name == "SAAS":
        surrogate = Surrogate(SaasFullyBayesianSingleTaskGP)
    elif gp_model_name == "GP":
        surrogate = Surrogate(
            botorch_model_class=FixedNoiseGP, allow_batched_models=False
        )

    if sparsity_threshold is None:
        sparsity_threshold = sparse_point.shape[-1]

    gs = GenerationStrategy(
        name=f"NEHVI_{penalty_name}_MOO_SAAS",
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
                    "acquisition_class": SEBOAcquisition,
                    "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
                    "torch_device": torch_device,
                    "acquisition_options": {
                        "penalty": penalty_name,
                        "target_point": torch.tensor(
                            sparse_point, device=torch_device, dtype=torch.double
                        ),
                        "sparsity_threshold": sparsity_threshold,
                        "prune_baseline": False,
                    },
                },
            ),
        ],
    )
    return gs
