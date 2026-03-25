from dataclasses import dataclass, field

from rkopenmdao.termination_criterion import TerminationCriterion


@dataclass
class IntegrationConfig:
    use_adaptive_time_stepping: bool
    termination_criterion: TerminationCriterion
    initial_step_size: float
