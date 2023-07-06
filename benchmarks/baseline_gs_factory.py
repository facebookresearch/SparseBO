#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
from ax.modelbridge.torch import TorchModelBridge

TORCH_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def get_saasbo_gs(
    num_sobol_trials: int, torch_device: Optional[torch.device] = TORCH_DEVICE
) -> GenerationStrategy:
    gs = GenerationStrategy(
        name="SAASBO",
        steps=[
            GenerationStep(model=get_sobol, num_trials=num_sobol_trials),
            GenerationStep(
                model=get_SAASBO,
                num_trials=-1,
                model_kwargs={"torch_device": torch_device},
            ),
        ],
    )
    return gs


def get_SAASBO(
    experiment: Experiment,
    data: Data,
    torch_device: Optional[torch.device] = TORCH_DEVICE,
) -> TorchModelBridge:
    """Instantiates a SAASBO model for single objective optimization."""
    return Models.FULLYBAYESIAN(
        num_samples=256,
        warmup_steps=512,
        disable_progbar=True,
        experiment=experiment,
        data=data,
        search_space=experiment.search_space,
        transforms=Cont_X_trans + Y_trans,
        torch_dtype=torch.double,
        torch_device=torch_device,
    )
