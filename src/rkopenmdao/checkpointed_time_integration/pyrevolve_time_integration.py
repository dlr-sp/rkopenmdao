"""Memory efficient checkpointing implementation using pyRevolve."""

from dataclasses import dataclass, field
from collections.abc import Callable
from copy import deepcopy
import warnings

import numpy as np
import pyrevolve as pr

from rkopenmdao.checkpointed_time_integration.checkpointed_time_integration import (
    CheckpointedTimeIntegration,
)
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps
from rkopenmdao.states import TimeIntegrationState


@dataclass
class PyrevolveTimeIntegration(CheckpointedTimeIntegration):
    """
    Checkpointer where checkpointing is done via pyRevolve. Memory efficient,
    but doesn't support online checkpointing for adaptive time stepping.

    Parameters
    ----------
    termination_criterion: TerminationCriterion
        Condition on when to stop the forward iteration.
    run_step_func: Callable[[int, TimeIntegrationState], TimeIntegrationState]
        Function for the computation of one step of the forward (primal) time
        integration. Input is the current time step, as well as the state of the time
        integration at the start of the step, return value the state at the end of the
        same step.
    run_step_jacvec_rev_func: Callable[
        [int, TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
        Function for the computation of one step of the reverse (linear) time
        integration. Inputs are the current (reverse) step, the state of the time
        integration during that step acting as linearization point, as well as the
        perturbations coming from the end of the time step. Return value is the
        perturbation for the start of the time step.
    state: TimeIntegrationState
        Time integration state on which all computations for the forward (primal) time
        integration are performed.
    state_perturbation: TimeIntegrationState
        Time integration state on which all computations for the reverse (linear) time
        integration are performed.
    revolver_type: str
        String representing the type of revolver to use. Options are 'Memory', 'Disc'
        'Base', 'SingleLevel', and 'MultiLevel' Default is 'Memory'.
    revolver_options: dict
        Configuration options passed to the revolver. Default is an empty dict.
    """

    revolver_type: str = "Memory"
    revolver_options: dict = field(default_factory=dict)

    _cached_input: TimeIntegrationState = field(init=None)
    _first_complete_state: TimeIntegrationState = field(init=None)
    _pyrevolve_state: TimeIntegrationState = field(init=None)
    _pyrevolve_state_perturbations: TimeIntegrationState = field(init=None)
    _revolver: pr.BaseRevolver | None = field(init=None)
    _revolver_class_type: type = field(init=None)

    def __post_init__(self):
        """Sets up all permanent data derived from initialization arguments."""
        self._cached_input = self.create_empty_primal_integration_state()
        self._first_complete_state = self.create_empty_primal_integration_state()
        self._pyrevolve_state = self.create_empty_primal_integration_state()
        self._pyrevolve_state_perturbations = (
            self.create_empty_derivative_integration_state()
        )
        checkpoint = TimeIntegrationCheckpoint(self._pyrevolve_state)

        self._revolver_class_type = self._setup_revolver_class_type(self.revolver_type)
        if isinstance(
            self.time_integration_config.termination_criterion, PredefinedNumberOfSteps
        ):
            num_steps = (
                self.time_integration_config.termination_criterion.number_of_steps
            )
        else:
            raise TypeError("""
                Does not support online checkpointing yet:
                Termination criterion must be of type PredefinedNumberOfSteps.
                """)
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

        self.revolver_options["fwd_operator"] = ForwardOperator(
            self._pyrevolve_state,
            self._run_step,
        )
        self.revolver_options["rev_operator"] = ReverseOperator(
            self._pyrevolve_state,
            self._pyrevolve_state_perturbations,
            self._run_step_adjoint_derivative,
        )
        self._revolver = None

    def _setup_revolver_class_type(self, revolver_type: str) -> type:
        """
        Returns specific class for the requested revolver type.

        Parameters
        ----------
        revolver_type: str
            String representing revolver type.

        Returns
        -------
        revolver_type: type
            Type used for creating the revolver.
        """
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

    def integrate(self, initial_state: TimeIntegrationState) -> TimeIntegrationState:
        """
        Runs forward iteration using internal revolver by pyRevolve.

        Parameters
        ----------
        initial_state: TimeIntegrationState
            The state at the beginning of the time integration process.

        Returns
        -------
        final_state: TimeIntegrationState
            The resulting state after completing all time steps.
        """
        self._cached_input = deepcopy(initial_state)
        self._revolver = self._revolver_class_type(**self.revolver_options)
        self._revolver.fwd_operator.time_integration_state.set(initial_state)
        self._revolver.fwd_operator.apply(t_start=-1, t_end=0)
        self._first_complete_state.set(
            self._revolver.fwd_operator.time_integration_state
        )
        self._pyrevolve_state.set(self._revolver.fwd_operator.time_integration_state)
        self._revolver.apply_forward()
        return [self._revolver.fwd_operator.time_integration_state]

    def integrate_adjoint_derivative(
        self,
        initial_state: TimeIntegrationState,
        final_state_perturbations: list[TimeIntegrationState],
    ) -> TimeIntegrationState:
        """
        Runs reverse iteration using internal revolver by pyRevolve.

        Parameters
        ----------
        final_state_perturbation: TimeIntegrationState
            The perturbation of the final state used as starting point for reverse
            iteration.

        Returns
        -------
        initial_state_perturbation: TimeIntegrationState
            The resulting perturbation of the initial state after completing all time
            steps.
        """
        if not self._revolver:
            self.integrate(initial_state)
        self._pyrevolve_state_perturbations.set(final_state_perturbations[0])
        self._revolver.apply_reverse()
        self._revolver.rev_operator.time_integration_state.set(
            self._first_complete_state
        )
        self._revolver.rev_operator.apply(t_start=-1, t_end=0)
        return self._revolver.rev_operator.time_integration_state_perturbations


class TimeIntegrationCheckpoint(pr.Checkpoint):
    """
    Blueprint for one checkpoint.

    Parameters
    ----------
    _symbol: dict
        Storage for making checkpoints accessible to pyRevolve.
    _time_integration_state: TimeIntegrationState
        Time integration state consistent with `_symbol`.
    """

    _symbol: dict
    _time_integration_state: TimeIntegrationState

    def __init__(self, time_integration_state: TimeIntegrationState):
        self._time_integration_state = time_integration_state
        self._symbols = time_integration_state.to_dict()

    def get_data_location(self, timestep) -> list:
        """
        Gets location for data at given time step.

        Parameters
        ----------
        timestep: Any
            Unused argument required by the interface.

        Returns
        -------
        locations: list
            Memory locations where checkpoint data is saved.
        """
        # pylint: disable=unused-argument
        # Here the method of getting the data is independent of the time step, but the
        # interface requires the argument.
        locations = self._get_data_impl(self._symbols, location=True)
        return locations

    def get_data(self, timestep) -> list:
        """
        Gets data at given time step.

        Parameters
        ----------
        timestep: Any
            Unused argument required by the interface.

        Returns
        -------
        locations: list
            Data if currently loaded checkpoint.
        """
        data = self._get_data_impl(self._symbols, location=False)
        return data

    def _get_data_impl(self, state_part: dict | np.ndarray, location: bool) -> list:
        """
        Recursive implementation of `get_data` and `get_data_location`.

        Parameters
        ----------
        state_part: dict | np.ndarray
            Current sub-part which is traversed to get data (location).
        location: bool
            Flag indicating whether location of data (True) or copy of data (False)
            is requested.

        Returns
        -------
        result: list
            List containing data (locations) of already traversed part of dict.
        """
        result = []
        if isinstance(state_part, dict):
            for part in state_part.values():
                result += self._get_data_impl(part, location)
        elif isinstance(state_part, np.ndarray):
            if state_part.size > 0:
                result.append(state_part if location else state_part.copy())
        else:
            raise TypeError(
                f"Unexpected type {type(state_part)} in RungeKuttaCheckpoint"
            )
        return result

    @property
    def size(self) -> int:
        """
        The memory consumption of the data contained in this checkpoint.

        Returns
        -------
        size: int
            Memory size of checkpoint in Bytes.
        """
        return self._size_impl(self._symbols)

    def _size_impl(self, state_part: dict | np.ndarray) -> int:
        """
        Recursive implementation of the size property.

        Parameters
        ----------
        state_part: dict | np.ndarray
            Current sub-part which is traversed to get checkpoint size.

        Returns
        -------
        size: int
            Size of already traversed part of checkpoint.
        """
        size = 0
        if isinstance(state_part, dict):
            for part in state_part.values():
                size += self._size_impl(part)
        elif isinstance(state_part, np.ndarray):
            size += state_part.size
        else:
            raise TypeError(
                f"Unexpected type {type(state_part)} in RungeKuttaCheckpoint"
            )
        return size

    @property
    def dtype(self) -> type:
        """
        Data type used for single values of the checkpoint.

        Returns
        -------
        dtype: type
            Data type used by single values of the checkpoint.
        """
        return np.float64


@dataclass
class ForwardOperator(pr.Operator):
    """
    Forward operator of the RungeKuttaIntegrator (i.e. the normal time
    integration).

    Parameters
    ----------
    time_integration_state: TimeIntegrationState
        State on which internal calculations are performed.
    fwd_operation: Callable[[int, TimeIntegrationState], TimeIntegrationState]
        Function applying one single step of time integration.
    """

    time_integration_state: TimeIntegrationState
    fwd_operation: Callable[[int, TimeIntegrationState], TimeIntegrationState]

    def apply(self, **kwargs):
        # PyRevolve only ever uses t_start and t_end as arguments, but the interface is
        # how it is.
        """
        Does forward (primal) integration from step `t_start` + 1 to `t_end` + 1.

        Parameters
        ----------
        t_start: int
            First integrated time step shifted by one.
        t_end: int
            Last integration time step shifted by one.
        """
        for step in range(kwargs["t_start"] + 1, kwargs["t_end"] + 1):
            self.fwd_operation(step, self.time_integration_state)


@dataclass
class ReverseOperator(pr.Operator):
    """
    Backward operator of the Runge-Kutta-integrator (i.e. one reverse step).

    Parameters
    ----------
    time_integration_state: TimeIntegrationState
        State which contains the current linearization point.
    time_integration_state_perturbations: TimeIntegrationState
        Perturbation state on which internal calculations are performed.
    rev_operation: Callable[
        [int, TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
        Function applying one single reverse step of time integration.
    """

    time_integration_state: TimeIntegrationState
    time_integration_state_perturbations: TimeIntegrationState
    rev_operation: Callable[
        [int, TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]

    def apply(self, **kwargs):
        """
        Does reverse (linear) integration from step `t_end` + 2 to `t_start` + 2.

        Parameters
        ----------
        t_start: int
            First integrated time step shifted by one.
        t_end: int
            Last integration time step shifted by one.
        """
        # PyRevolve only ever uses t_start and t_end as arguments, but the interface is
        # how it is.
        for step in reversed(range(kwargs["t_start"] + 2, kwargs["t_end"] + 2)):
            self.rev_operation(
                step,
                self.time_integration_state,
                self.time_integration_state_perturbations,
            )
