#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Dict, List, Optional

import torch
from ax.modelbridge.factory import get_GPEI, get_sobol
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy
from ax.utils.common.logger import get_logger
from baseline_gs_factory import (
    get_saasbo_gs,
    TORCH_DEVICE,
)
from external_l1_gs_factory import get_er_l1_gp_gs, get_er_l1_saas_gs
from internal_l1_gs_factory import get_ir_l1_gp_gs, get_ir_l1_saas_gs
from sebo_gs_factory import get_sebo_gs

logger: Logger = get_logger(__name__)


def get_generation_strategy(
    strategy_name: str,
    num_sobol_trials: int,
    sparse_point: List,
    strategies_args: Optional[Dict[str, Any]] = None,
) -> GenerationStrategy:

    if strategies_args.get("torch_device", None) is not None:
        torch_device = torch.device(strategies_args.get("torch_device"))
    else:
        torch_device = TORCH_DEVICE
    sparse_point = torch.tensor(sparse_point, device=torch_device, dtype=torch.double)

    logger.info(f"torch device: {torch_device}")

    if strategy_name == "Sobol":
        gs = GenerationStrategy(
            name="Sobol", steps=[GenerationStep(model=get_sobol, num_trials=-1)]
        )
    elif strategy_name == "GPEI":
        gs = GenerationStrategy(
            name="GPEI",
            steps=[
                GenerationStep(model=get_sobol, num_trials=num_sobol_trials),
                GenerationStep(model=get_GPEI, num_trials=-1),
            ],
        )
    elif strategy_name == "SAASBO":
        gs = get_saasbo_gs(num_sobol_trials=num_sobol_trials, torch_device=torch_device)
    elif strategy_name == "REMBO":
        gs = REMBOStrategy(
            D=len(sparse_point),
            d=strategies_args.get(strategy_name, {}).get("d", 8),
            init_per_proj=num_sobol_trials,
        )
    elif "L1_internal_SAAS" in strategy_name:
        gs = get_ir_l1_saas_gs(
            num_sobol_trials=num_sobol_trials,
            sparse_point=sparse_point,
            regularization_parameter=strategies_args.get(
                "regularization_parameter", 0.1
            ),
            torch_device=torch_device,
        )
    elif "L1_internal_GP" in strategy_name:
        gs = get_ir_l1_gp_gs(
            num_sobol_trials=num_sobol_trials,
            sparse_point=sparse_point,
            regularization_parameter=strategies_args.get(
                "regularization_parameter", 0.1
            ),
            torch_device=torch_device,
        )
    elif "L1_external_SAAS" in strategy_name:
        gs = get_er_l1_saas_gs(
            num_sobol_trials=num_sobol_trials,
            sparse_point=sparse_point,
            regularization_parameter=strategies_args.get(
                "regularization_parameter", 0.1
            ),
            torch_device=torch_device,
        )
    elif "L1_external_GP" in strategy_name:
        gs = get_er_l1_gp_gs(
            num_sobol_trials=num_sobol_trials,
            sparse_point=sparse_point,
            regularization_parameter=strategies_args.get(
                "regularization_parameter", 0.1
            ),
            torch_device=torch_device,
        )
    elif "NEHVI_L1_MOO_SAAS" in strategy_name:
        gs = get_sebo_gs(
            sparse_point=sparse_point,
            penalty_name="L1_norm",
            num_sobol_trials=num_sobol_trials,
            gp_model_name="SAAS",
            sparsity_threshold=strategies_args.get(
                "sparsity_threshold",
                sparse_point.shape[-1],
            ),
            torch_device=torch_device,
        )
    elif "NEHVI_L0_MOO_SAAS" in strategy_name:
        gs = get_sebo_gs(
            sparse_point=sparse_point,
            penalty_name="L0_norm",
            num_sobol_trials=num_sobol_trials,
            gp_model_name="SAAS",
            sparsity_threshold=strategies_args.get(
                "sparsity_threshold",
                sparse_point.shape[-1],
            ),
            torch_device=torch_device,
        )
    elif "NEHVI_L1_MOO_GP" in strategy_name:
        gs = get_sebo_gs(
            sparse_point=sparse_point,
            penalty_name="L1_norm",
            num_sobol_trials=num_sobol_trials,
            gp_model_name="GP",
            sparsity_threshold=strategies_args.get(
                "sparsity_threshold",
                sparse_point.shape[-1],
            ),
            torch_device=torch_device,
        )
    elif "NEHVI_L0_MOO_GP" in strategy_name:
        gs = get_sebo_gs(
            sparse_point=sparse_point,
            penalty_name="L0_norm",
            num_sobol_trials=num_sobol_trials,
            gp_model_name="GP",
            sparsity_threshold=strategies_args.get(
                "sparsity_threshold",
                sparse_point.shape[-1],
            ),
            torch_device=torch_device,
        )
    return gs
