from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from adaptive import adaptive_simulation
from avg_homogenous import avg_homogeneous_simulation
from homogenous import homogeneous_simulation

## Simulate Problem
### Integrate adaptively
adaptive_simulation(PROBLEM, BUTCHER_TABLEAUX)

### Integrate homogenously
avg_homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)

### Integrate homogenously with Avg. time step size wrt. adaptive method
homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
