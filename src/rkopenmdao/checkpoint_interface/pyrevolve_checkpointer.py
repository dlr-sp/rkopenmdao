# pylint: disable=missing-module-docstring
from dataclasses import dataclass, field
from copy import deepcopy
import warnings

import pyrevolve as pr
import numpy as np
from rkopenmdao.integration_control import StepTerminationIntegrationControl
from rkopenmdao.time_integration_state import TimeIntegrationState

from .checkpoint_interface import CheckpointInterface
from .runge_kutta_integrator_pyrevolve_classes import (
    RungeKuttaCheckpoint,
    RungeKuttaForwardOperator,
    RungeKuttaReverseOperator,
)


@dataclass
class PyrevolveCheckpointer(CheckpointInterface):
    """Checkpointer where checkpointing is done via pyRevolve. Most memory efficient
    version, but doesn't support online checkpointing (i.e. unknown number of time
    steps), and never will unless it is implemented in PyRevolve."""

    revolver_type: str = "Memory"
    revolver_options: dict = field(default_factory=dict)

    def __post_init__(self):
        """Sets up all permanent data derived from initialization arguments."""
        self._first_complete_state = deepcopy(self.state)
        checkpoint = RungeKuttaCheckpoint(self.state)

        self.revolver_class_type = self._setup_revolver_class_type(self.revolver_type)
        if isinstance(self.integration_control, StepTerminationIntegrationControl):
            num_steps = self.integration_control.num_steps
        else:
            raise TypeError(
                """
            Does not support online checkpointing yet:
            IntegrationControl must be of type StepTerminationIntegrationControl
            """
            )
        for key, value in self.revolver_options.items():
            if self.revolver_type == "MultiLevel" and key == "storage_list":
                storage_list = []
                for storage_type, options in value.items():
                    if storage_type == "Numpy":
                        storage_list.append(pr.NumpyStorage(checkpoint.size, **options))
                    elif storage_type == "Disk":
                        storage_list.append(pr.DiskStorage(checkpoint.size, **options))
                    elif storage_type == "Bytes":
                        storage_list.append(pr.BytesStorage(checkpoint.size, **options))
                self.revolver_options[key] = storage_list
            else:
                self.revolver_options[key] = value
        self.revolver_options["checkpoint"] = checkpoint
        self.revolver_options["n_timesteps"] = num_steps - 1

        if "n_checkpoints" not in self.revolver_options:
            if self.revolver_type not in ["MultiLevel", "Base"]:
                self.revolver_options["n_checkpoints"] = 1 if num_steps == 1 else None

        self.revolver_options["fwd_operator"] = RungeKuttaForwardOperator(
            self.state,
            self.run_step_func,
            self.integration_control,
        )
        self.revolver_options["rev_operator"] = RungeKuttaReverseOperator(
            self.state,
            self.state_perturbation,
            self.run_step_jacvec_rev_func,
            self.integration_control,
        )
        self._revolver = None

    def create_checkpointer(self):
        """Creates the actual revolver object used for checkpointing by Pyrevolve."""
        self._revolver = self.revolver_class_type(**self.revolver_options)

    def _setup_revolver_class_type(self, revolver_type: str):
        """Returns fitting class for given revolver type."""
        if revolver_type == "SingleLevel":
            return pr.SingleLevelRevolver
        elif revolver_type == "MultiLevel":
            warnings.warn(
                """MultiLevelRevolver currently has problems where certain numbers of
                checkpoints work and others don't (without an obvious reason why).
                Use with care."""
            )
            return pr.MultiLevelRevolver
        elif revolver_type == "Disk":
            return pr.DiskRevolver
        elif revolver_type == "Base":
            return pr.BaseRevolver
        elif revolver_type == "Memory":
            return pr.MemoryRevolver
        else:
            raise TypeError(
                "Given revolver_type is invalid. Options are 'Memory', 'Disk',"
                f"'SingleLevel', 'MultiLevel' and Base. Given was '{revolver_type}'."
            )

    def iterate_forward(self, initial_state):
        """Runs forward iteration of internal Pyrevolve-Revolver"""
        self._revolver.fwd_operator.time_integration_state.set(initial_state)
        self._revolver.fwd_operator.apply(t_start=-1, t_end=0)
        self._first_complete_state.set(
            self._revolver.fwd_operator.time_integration_state
        )
        self.state.set(self._revolver.fwd_operator.time_integration_state)
        self._revolver.apply_forward()
        return self._revolver.fwd_operator.time_integration_state

    def iterate_reverse(self, final_state_perturbation):
        """Runs reverse iteration of internal Pyrevolve-Revolver"""
        self.state_perturbation.set(final_state_perturbation)
        self._revolver.apply_reverse()
        self._revolver.rev_operator.time_integration_state.set(
            self._first_complete_state
        )
        self._revolver.rev_operator.apply(t_start=-1, t_end=0)
        return self._revolver.rev_operator.time_integration_state_perturbations
