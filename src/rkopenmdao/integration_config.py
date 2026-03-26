"""Class containing basic configuration options for time integration."""

from dataclasses import dataclass

from rkopenmdao.termination_criterion import TerminationCriterion


@dataclass
class IntegrationConfig:
    """
    Class containing basic configuration options for time integration.

    Parameters
    ----------
    use_adaptive_time_stepping: bool
        Toggles the use of adaptive time stepping.
    termination_criterion: TerminationCriterion
        Criterion which controls when to stop the time integration.
    initial_step_size: float
        Step size with which the time integration is started. Will most likely be
        changed in case of adaptive time stepping.
    """

    use_adaptive_time_stepping: bool
    termination_criterion: TerminationCriterion
    initial_step_size: float
