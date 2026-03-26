# Changelog

## Unreleased

- Split `IntegrationControl` into three classes
    1. `IntegrationConfig`, containing general configuration options for the time integration.
    2. `TerminationCriterion`, the condition on when to terminate time integration.
    3. `OMDataExchange`, the object with which the time integration exchanges data with the OpenMDAO based ODE implementation.
- Changes time from being a "pseudo-constant" to a variable in the OpenMDAO ODE implementation and time integration in order to allow in the future for optimization scenarios with termination criteratia that need derivatives wrt. to eg initial or final time.
- Add custom explicit and implicit component types for use with OpenMDAO-based time integration.
- Removed heat equation example
- Removed convergence test from `arc/rkopenmdao/utils`.

## Version 0.2

 - Initial version before change tracking