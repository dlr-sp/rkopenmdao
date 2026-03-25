# Changelog

## Unreleased

- Split `IntegrationControl` into three classes
    1. `IntegrationConfig`, containing general configuration options for the time integration.
    2. `TerminationCriterion`, the condition on when to terminate time integration.
    3. `OMDataExchange`, the object with which the time integration exchanges data with the OpenMDAO based ODE implementation.
- Changes time from being a "pseudo-constant" to a variable in the OpenMDAO ODE implementation amnd time integration in order to allow in the future for optimization scenarios with termination criteratia that need derivatives wrt. to eg initial or final time.  

## Version 0.2

 - Initial version before change tracking