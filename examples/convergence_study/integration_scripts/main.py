"""Simulate all problems homogeneously and adaptively"""

from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from .adaptive import adaptive_simulation
from .avg_homogeneous import avg_homogeneous_simulation
from .homogeneous import homogeneous_simulation

## Simulate Problem
### Integrate homogenously
# homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)

### Integrate adaptively
# adaptive_simulation(PROBLEM, BUTCHER_TABLEAUX)

### Integrate homogenously with Avg. time step size wrt. adaptive method
avg_homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
