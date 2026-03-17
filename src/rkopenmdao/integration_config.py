from dataclasses import dataclass, field

from rkopenmdao.termination_criterion import TerminationCriterion


@dataclass
class IntegrationConfig:
    use_adaptive_integration: bool
    termination_criterion: TerminationCriterion
