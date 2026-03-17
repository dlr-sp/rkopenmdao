from dataclasses import dataclass


@dataclass
class OMDataExchange:
    step_size: float = 1.0e-3
    stage_factor: float = 1.0
    # time: float = 0.0
