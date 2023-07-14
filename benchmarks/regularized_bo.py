#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.models.torch.botorch_modular.sebo import (
    get_batch_initial_conditions,
    SEBOAcquisition,
)
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.penalized import (
    L0PenaltyApprox,
    L0PenaltyApproxObjective,
    PenalizedAcquisitionFunction,
    PenalizedMCObjective,
)
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.optim import (
    Homotopy,
    HomotopyParameter,
    LogLinearHomotopySchedule,
    optimize_acqf_homotopy,
)
from botorch.posteriors.fully_bayesian import MCMC_DIM
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.objective import get_objective_weights_transform
from botorch.utils.transforms import is_fully_bayesian
from torch import Tensor


logger: Logger = get_logger(__name__)


class ExternalRegularizedL0(Acquisition):
    """
    Implement the external regularizied acquisition function with L0 norm.

    The ER-L0 takes a regularization parameter and add L0 norm directly to the
    acqusition function. The regularization parameter controls the target sparisty
    level. It uses the same optimization method as SEBO-L0 i.e. a differentiable
    relaxation based on homotopy continuation to efficiently optimize for sparsity.
    """

    def __init__(
        self,
        surrogates: Dict[str, Surrogate],
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: Type[AcquisitionFunction],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(surrogates) > 1:
            raise ValueError("ER-L0 does not support support multiple surrogates.")
        surrogate = surrogates[Keys.ONLY_SURROGATE]

        tkwargs = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = options or {}
        self.target_point: Tensor = options.get("target_point", None)
        if self.target_point is None:
            raise ValueError("please provide target point.")
        self.target_point.to(**tkwargs)  # pyre-ignore

        self.regularization_parameter: float = options.get(
            "regularization_parameter", 0.0
        )

        # construct determinsitic model for penalty term
        self.penalty_name = "L0_norm"
        # pyre-fixme[4]: Attribute must be annotated.
        self.penalty_term = self._construct_penalty()

        # instantiate botorch_acqf_class
        super().__init__(
            surrogates={"regularized_bo": surrogate},
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_class=botorch_acqf_class,
            options=options,
        )
        raw_acqf = self.acqf
        self.acqf = PenalizedAcquisitionFunction(
            raw_acqf=raw_acqf,
            penalty_func=self.penalty_term,
            regularization_parameter=self.regularization_parameter,
        )
        # create X_pareto for gen batch initial
        Xs = self.surrogates["regularized_bo"].Xs
        all_Y = torch.cat(
            # pyre-ignore
            [d.Y.values for d in self.surrogates["regularized_bo"].training_data],
            dim=-1,
        )
        Y_pareto = torch.cat(
            [
                all_Y,
                self.penalty_term(Xs[0].unsqueeze(1)).unsqueeze(-1),
            ],
            dim=-1,
        )
        # pyre-ignore
        self.X_pareto = self._obtain_X_pareto(Y_pareto=Y_pareto, **tkwargs)

    def _obtain_X_pareto(self, Y_pareto: Tensor, **tkwargs: Any) -> Tensor:
        ow = torch.cat([self._full_objective_weights, torch.tensor([-1], **tkwargs)])
        ind_pareto = is_non_dominated(Y_pareto * ow)
        X_pareto = self.surrogates["regularized_bo"].Xs[0][ind_pareto].clone()
        return X_pareto

    def _construct_penalty(self) -> GenericDeterministicModel:
        """Construct a penalty term to be added to ER-L0 acqusition function.
        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        L0 = L0PenaltyApprox(target_point=self.target_point, a=1e-3)
        return GenericDeterministicModel(f=L0)

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate a set of candidates via multi-start optimization. Obtains
        candidates and their associated acquisition function values.

        Args:
            n: The number of candidates to generate.
            search_space_digest: A ``SearchSpaceDigest`` object containing search space
                properties, e.g. ``bounds`` for optimization.
            inequality_constraints: A list of tuples (indices, coefficients, rhs),
                with each tuple encoding an inequality constraint of the form
                ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that post-processes an optimization
                result appropriately (i.e., according to `round-trip`
                transformations).
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``.
        """
        candidates, expected_acquisition_value = SEBOAcquisition.optimize(
            self=self,  # pyre-ignore
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
            optimizer_options=optimizer_options,
        )
        return candidates, expected_acquisition_value

    def _optimize_with_homotopy(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Optimize ER ACQF with L0 norm using homotopy."""
        # extend to fixed a no homotopy_schedule schedule
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()

        homotopy_schedule = LogLinearHomotopySchedule(start=0.1, end=1e-3, num_steps=30)

        # Prepare arguments for optimizer
        optimizer_options_with_defaults = optimizer_argparse(
            self.acqf,
            bounds=bounds,
            q=n,
            optimizer_options=optimizer_options,
        )

        def callback():  # pyre-ignore
            X_pending = self.acqf.X_pending
            self.acqf.__init__(
                raw_acqf=self.acqf.raw_acqf,
                penalty_func=self.penalty_term,
                regularization_parameter=self.regularization_parameter,
            )
            self.acqf.model = self.surrogates["regularized_bo"].model
            self.acqf.set_X_pending(X_pending)

        homotopy = Homotopy(
            homotopy_parameters=[
                HomotopyParameter(
                    parameter=self.penalty_term._f.a,
                    schedule=homotopy_schedule,
                )
            ],
            callbacks=[callback],
        )
        batch_initial_conditions = get_batch_initial_conditions(
            acq_function=self.acqf,
            raw_samples=optimizer_options_with_defaults["raw_samples"],
            X_pareto=self.X_pareto,
            target_point=self.target_point,
            num_restarts=optimizer_options_with_defaults["num_restarts"],
            **{"device": self.device, "dtype": self.dtype},
        )
        candidates, expected_acquisition_value = optimize_acqf_homotopy(
            q=n,
            acq_function=self.acqf,
            bounds=bounds,
            homotopy=homotopy,
            num_restarts=optimizer_options_with_defaults["num_restarts"],
            raw_samples=optimizer_options_with_defaults["raw_samples"],
            post_processing_func=rounding_func,
            fixed_features=fixed_features,
            batch_initial_conditions=batch_initial_conditions,
        )
        return candidates, expected_acquisition_value


class InternalRegularizedL0(ExternalRegularizedL0):
    """
    Implement the internal regularizied acquisition function with L0 norm.

    The IR-L0 takes a regularization parameter and add L0 norm directly to the
    objective function. The regularization parameter controls the target sparisty
    level. It uses the same optimization method as SEBO-L0 i.e. a differentiable
    relaxation based on homotopy continuation to efficiently optimize for sparsity.
    """

    def __init__(
        self,
        surrogates: Dict[str, Surrogate],
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: Type[AcquisitionFunction],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(surrogates) > 1:
            raise ValueError("IR-L0 does not support support multiple surrogates.")
        surrogate = surrogates[Keys.ONLY_SURROGATE]

        tkwargs = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = options or {}
        self.target_point: Tensor = options.get("target_point", None)
        if self.target_point is None:
            raise ValueError("please provide target point.")
        self.target_point.to(**tkwargs)  # pyre-ignore

        self.regularization_parameter: float = options.get(
            "regularization_parameter", 0.0
        )
        # if fully-bayesian model is used,
        # decide dim to expand penalty term to match dimension of objective
        self.expand_dim: Optional[int] = None
        if is_fully_bayesian(surrogate.model):
            self.expand_dim = MCMC_DIM + 1

        # construct determinsitic model for penalty term
        self.penalty_name = "L0_norm"
        # pyre-fixme[4]: Attribute must be annotated.
        self.penalty_term = self._construct_penalty()

        # instantiate botorch_acqf_class
        Acquisition.__init__(
            self=self,
            surrogates={"regularized_bo": surrogate},
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_class=botorch_acqf_class,
            options=options,
        )
        # create X_pareto for gen batch initial
        Xs = self.surrogates["regularized_bo"].Xs
        all_Y = torch.cat(
            # pyre-ignore
            [d.Y.values for d in self.surrogates["regularized_bo"].training_data],
            dim=-1,
        )
        Y_pareto = torch.cat(
            [
                all_Y,
                self.penalty_term(Xs[0]).transpose(1, 0),
            ],
            dim=-1,
        )
        # pyre-ignore
        self.X_pareto = self._obtain_X_pareto(Y_pareto=Y_pareto, **tkwargs)

    def _construct_penalty(self) -> GenericDeterministicModel:
        """Construct a penalty term to be added to the objective function to be used in IR-L0.
        Returns:
            A "1 x batch_shape x q" tensor representing the penalty for each point.
            The first dimension corresponds to the dimension of MC samples.
        """
        L0 = L0PenaltyApproxObjective(target_point=self.target_point, a=1e-3)
        return GenericDeterministicModel(f=L0)

    def get_botorch_objective_and_transform(
        self,
        botorch_acqf_class: Type[AcquisitionFunction],
        model: Model,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
        risk_measure: Optional[RiskMeasureMCObjective] = None,
    ) -> Tuple[Optional[MCAcquisitionObjective], Optional[PosteriorTransform]]:
        """Construct the penalized objective by adding the penalty term to
        the original objective.
        """
        if outcome_constraints is not None:
            raise RuntimeError(
                "Outcome constraints are not supported for PenalizedMCObjective in "
                + "InternalRegularized Acqf."
            )

        obj_tf: Callable[
            [Tensor, Optional[Tensor]], Tensor
        ] = get_objective_weights_transform(objective_weights)

        def objective(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
            return obj_tf(samples)  # pyre-ignore

        mc_objective_kwargs = {
            "penalty_objective": self.penalty_term,
            "regularization_parameter": self.regularization_parameter,
            "expand_dim": self.expand_dim,
        }
        objective = PenalizedMCObjective(objective=objective, **mc_objective_kwargs)
        return objective, None

    def _optimize_with_homotopy(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Optimize IR ACQF with L0 norm using homotopy."""
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()

        homotopy_schedule = LogLinearHomotopySchedule(start=0.1, end=1e-3, num_steps=30)

        # Prepare arguments for optimizer
        optimizer_options_with_defaults = optimizer_argparse(
            self.acqf,
            bounds=bounds,
            q=n,
            optimizer_options=optimizer_options,
        )
        print(f"optimizer options: {optimizer_options_with_defaults}")

        def callback():  # pyre-ignore
            X_pending = self.acqf.X_pending
            self.acqf.__init__(  # pyre-ignore
                X_baseline=self.X_observed,
                model=self.surrogates["regularized_bo"].model,
                objective=self.acqf.objective,
                posterior_transform=self.acqf.posterior_transform,
                prune_baseline=self.options.get("prune_baseline", True),
                cache_root=self.options.get("cache_root", True),
            )
            self.acqf.set_X_pending(X_pending)

        homotopy = Homotopy(
            homotopy_parameters=[
                HomotopyParameter(
                    parameter=self.penalty_term._f.a,
                    schedule=homotopy_schedule,
                )
            ],
            callbacks=[callback],
        )
        # need to know sparse dimensions
        batch_initial_conditions = get_batch_initial_conditions(
            acq_function=self.acqf,
            raw_samples=optimizer_options_with_defaults["raw_samples"],
            X_pareto=self.X_pareto,
            target_point=self.target_point,
            num_restarts=optimizer_options_with_defaults["num_restarts"],
            **{"device": self.device, "dtype": self.dtype},
        )

        candidates, expected_acquisition_value = optimize_acqf_homotopy(
            q=n,
            acq_function=self.acqf,
            bounds=bounds,
            homotopy=homotopy,
            num_restarts=optimizer_options_with_defaults["num_restarts"],
            raw_samples=optimizer_options_with_defaults["raw_samples"],
            post_processing_func=rounding_func,
            fixed_features=fixed_features,
            batch_initial_conditions=batch_initial_conditions,
        )
        return candidates, expected_acquisition_value
