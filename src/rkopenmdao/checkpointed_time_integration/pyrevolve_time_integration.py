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
    Memory-efficient checkpointing using pyRevolve for reverse-mode integration.

    This implementation uses the pyRevolve library to implement adaptive checkpointing
    strategies for efficient reverse-mode (adjoint) time integration. It supports
    multiple strategies that trade off memory usage against computation time:

    - Memory (default): In-memory adaptive checkpointing
    - Disk: Checkpointing to disk for large problems
    - SingleLevel: Simple single-level reversible checkpointing
    - MultiLevel: Multi-level hierarchical checkpointing (use with caution)
    - Base: Base pyRevolve implementation

    The strategy works by storing a subset of intermediate states (checkpoints)
    and recomputing missing states as needed during the reverse integration.
    This achieves O(sqrt(n_steps)) memory usage with only O(log(n_steps))
    recomputation per step.

    Parameters
    ----------
    ode : DiscretizedODE
        The discretized ODE system to integrate (inherited from base class).
    time_discretization_scheme : TimeDiscretizationSchemeInterface
        The time discretization scheme (inherited from base class).
    error_controller : ErrorController
        The adaptive step size controller (inherited from base class).
    error_measurer : ErrorMeasurer
        The error measurement strategy (inherited from base class).
    time_integration_config : IntegrationConfig
        Integration configuration including termination criterion
        (inherited from base class).
    integrate_callbacks : list[Callback]
        Callbacks for primal integration (inherited from base class).
    integrate_derivative_callbacks : list[Callback]
        Callbacks for derivative integration (inherited from base class).
    integrate_adjoint_derivative_callbacks : list[Callback]
        Callbacks for adjoint derivative integration (inherited from base class).
    revolver_type : str, optional
        Type of pyRevolve strategy to use. Options are 'Memory', 'Disk',
        'SingleLevel', 'MultiLevel', and 'Base'. Default is 'Memory'.
    revolver_options : dict, optional
        Additional options passed to the pyRevolve constructor. Can include:
        - n_checkpoints: Number of checkpoints to store (auto-computed if not given)
        - storage_list: For MultiLevel, list of storage backends
        - Other options specific to the revolver type

    Notes
    -----
    **Termination Criterion Requirement:**
    This implementation requires a termination criterion with a fixed number of steps
    (PredefinedNumberOfSteps). Online checkpointing (determining steps adaptively)
    is not yet supported.

    **Memory vs Computation Tradeoff:**
    - More checkpoints = less recomputation, more memory
    - Fewer checkpoints = more recomputation, less memory
    - The 'n_checkpoints' option controls this tradeoff

    **MultiLevelRevolver Warning:**
    The MultiLevelRevolver has known issues where certain numbers of checkpoints work
    while others don't, without an obvious pattern. Use with caution and test thoroughly.

    **Storage Options:**
    For 'MultiLevel' strategy, the 'storage_list' option accepts a list of
    (storage_type, options) tuples where storage_type can be 'Numpy', 'Disk', or 'Bytes'.
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
        """Sets up pyRevolve with checkpointing configuration."""
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

    def integrate(
        self, initial_state: TimeIntegrationState
    ) -> list[TimeIntegrationState]:
        """
        Performs forward integration using pyRevolve checkpointing.

        This method uses the configured pyRevolve strategy to integrate forward in time
        while storing checkpoints as needed for efficient reverse integration.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The initial state containing:
            - discretization_state: Initial values of state variables
            - step_size_suggestion: Initial step size suggestion
            - step_size_history: Initial step size history
            - error_history: Initial error history

        Returns
        -------
        list[TimeIntegrationState]
            List containing the final state after completing all time steps.
            The list contains a single element.

        Notes
        -----
        The forward integration proceeds as follows:
        1. Initialize the pyRevolve operator with the forward integration callback
        2. Apply the forward operator to integrate from t=-1 to t=0 (internal indexing)
        3. Store the first complete state for later use in reverse integration
        4. Apply the forward strategy to complete the integration

        The checkpointing strategy stores intermediate states as determined by
        the 'n_checkpoints' option and the number of time steps.
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
        Performs reverse (adjoint) integration using pyRevolve checkpointing.

        This method integrates backward in time to compute the gradient of a cost
        function with respect to initial conditions. It uses stored checkpoints to
        reconstruct intermediate states as needed.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The initial state from the forward integration (linearization point).
            This is used to reinitialize the integration if needed.

        final_state_perturbations : list[TimeIntegrationState]
            List containing the final state perturbation (adjoint variable) which
            represents the gradient of the cost function with respect to the final state.
            Must contain exactly one element.

        Returns
        -------
        TimeIntegrationState
            The perturbation of the initial state after reverse integration.
            Contains:
            - discretization_state: Gradient of cost function w.r.t. initial conditions
            - Empty or zero step_size_suggestion, step_size_history, error_history

        Notes
        -----
        The reverse integration proceeds as follows:
        1. If not already run, perform forward integration to establish checkpoints
        2. Set the final perturbation state as the starting point for reverse
        3. Apply the reverse strategy to integrate backward through time
        4. Apply the reverse operator to complete the integration from t=-1 to t=0
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
    pyRevolve checkpoint implementation for TimeIntegrationState.

    This class implements the pyRevolve Checkpoint interface to enable checkpointing
    of TimeIntegrationState objects. It provides methods for accessing checkpoint data
    and determining checkpoint size for memory management.

    Notes
    -----
    The checkpoint stores a copy of the state's data in a dictionary format.
    The pyRevolve library uses the get_data() and get_data_location() methods
    to manage checkpoint storage and retrieval.

    The size property allows pyRevolve to estimate memory usage and make
    checkpointing decisions accordingly.
    """

    _symbols: dict
    _time_integration_state: TimeIntegrationState

    def __init__(self, time_integration_state: TimeIntegrationState):
        """
        Initialize the checkpoint with a time integration state.

        Parameters
        ----------
        time_integration_state : TimeIntegrationState
            The state to checkpoint. The state is converted to a dictionary
            representation via time_integration_state.to_dict().
        """
        self._time_integration_state = time_integration_state
        self._symbols = time_integration_state.to_dict()

    def get_data_location(self, timestep) -> list:
        """
        Gets memory locations for checkpoint data at a given time step.

        This method is part of the pyRevolve Checkpoint interface and is used
        to retrieve references to stored data during reverse integration.

        Parameters
        ----------
        timestep : Any
            Unused argument required by the pyRevolve interface.
            The checkpoint data location is independent of the time step.

        Returns
        -------
        list
            Memory locations (references) where checkpoint data is stored.
            These are typically references to the internal numpy arrays.
        """
        locations = self._get_data_impl(self._symbols, location=True)
        return locations

    def get_data(self, timestep) -> list:
        """
        Gets checkpoint data at a given time step.

        This method is part of the pyRevolve Checkpoint interface and is used
        to retrieve checkpoint data during forward or reverse integration.

        Parameters
        ----------
        timestep : Any
            Unused argument required by the pyRevolve interface.
            The checkpoint data is independent of the time step.

        Returns
        -------
        list
            Copies of the checkpoint data as numpy arrays.
        """
        data = self._get_data_impl(self._symbols, location=False)
        return data

    def _get_data_impl(self, state_part: dict | np.ndarray, location: bool) -> list:
        """
        Recursive implementation of data access methods.

        Traverses the nested dictionary structure of the checkpoint state to
        collect either data copies or memory locations.

        Parameters
        ----------
        state_part : dict | np.ndarray
            Current sub-part of the state being traversed.
        location : bool
            If True, return memory locations (references to arrays).
            If False, return copies of the data.

        Returns
        -------
        list
            List of collected data or locations from the traversed structure.

        Raises
        ------
        TypeError
            If state_part is neither a dict nor a numpy array.
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
        Memory consumption of the checkpoint in bytes.

        Returns the total memory used by all arrays in the checkpoint state.

        Returns
        -------
        int
            Memory size of checkpoint in Bytes.
        """
        return self._size_impl(self._symbols)

    def _size_impl(self, state_part: dict | np.ndarray) -> int:
        """
        Recursive implementation of size calculation.

        Traverses the nested dictionary structure to sum the sizes of all arrays.

        Parameters
        ----------
        state_part : dict | np.ndarray
            Current sub-part of the state being traversed.

        Returns
        -------
        int
            Total size of the traversed part in bytes.

        Raises
        ------
        TypeError
            If state_part is neither a dict nor a numpy array.
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
        Data type used for checkpoint values.

        Returns
        -------
        type
            The numpy dtype (np.float64) used for checkpoint values.
        """
        return np.float64


@dataclass
class ForwardOperator(pr.Operator):
    """
    Forward (primal) integration operator for pyRevolve.

    This operator wraps the primal integration step and is called by pyRevolve
    during forward integration to advance the solution by one time step.

    Parameters
    ----------
    time_integration_state : TimeIntegrationState
        The state on which forward integration is performed.
        Modified in-place during apply().
    fwd_operation : Callable[[int, TimeIntegrationState], None]
        Function that applies one time integration step.
        Takes the iteration number and state as arguments.
        Must modify the state in-place.
    """

    time_integration_state: TimeIntegrationState
    fwd_operation: Callable[[int, TimeIntegrationState], None]

    def apply(self, **kwargs):
        """
        Performs forward integration from step t_start+1 to t_end+1.

        This method is called by pyRevolve during forward integration to advance
        the solution through multiple time steps.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments from pyRevolve containing:
            - t_start: int, first time step (shifted by one internally)
            - t_end: int, last time step (shifted by one internally)
        """
        for step in range(kwargs["t_start"] + 1, kwargs["t_end"] + 1):
            self.fwd_operation(step, self.time_integration_state)


@dataclass
class ReverseOperator(pr.Operator):
    """
    Reverse (adjoint) integration operator for pyRevolve.

    This operator wraps the adjoint integration step and is called by pyRevolve
    during reverse integration to compute gradients backward in time.

    Parameters
    ----------
    time_integration_state : TimeIntegrationState
        The primal state (linearization point) at the current time level.
    time_integration_state_perturbations : TimeIntegrationState
        The perturbation state on which reverse integration is performed.
        Modified in-place during apply().
    rev_operation : Callable[[int, TimeIntegrationState, TimeIntegrationState], None]
        Function that applies one reverse integration step.
        Takes the iteration number, primal state, and perturbation state as arguments.
        Must modify the perturbation state in-place.

    See Also
    --------
    ForwardOperator : Forward (primal) integration operator
    PyrevolveTimeIntegration : Uses this operator for reverse integration
    """

    time_integration_state: TimeIntegrationState
    time_integration_state_perturbations: TimeIntegrationState
    rev_operation: Callable[[int, TimeIntegrationState, TimeIntegrationState], None]

    def apply(self, **kwargs):
        """
        Performs reverse (adjoint) integration from step t_end+2 to t_start+2.

        This method is called by pyRevolve during reverse integration to compute
        gradients backward through time.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments from pyRevolve containing:
            - t_start: int, first time step (shifted by two internally)
            - t_end: int, last time step (shifted by two internally)
        """
        for step in reversed(range(kwargs["t_start"] + 2, kwargs["t_end"] + 2)):
            self.rev_operation(
                step,
                self.time_integration_state,
                self.time_integration_state_perturbations,
            )
