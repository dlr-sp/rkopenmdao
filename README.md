# Runge-Kutta-OpenMDAO

Prototype for solving time-dependent problems in OpenMDAO using Runge-Kutta-schemes.

## Description

Runge-Kutta-schemes are widely used methods for solving initial value problems. Implementing this into OpenMDAO is done by using nesting:
An inner OpenMDAO-Problem is used to model one Runge-Kutta stage of an instationary multidisciplinary problem.
An outer explicit component loops over the time steps and time stages, running the inner problem to compute the stage values.

## Getting started

### Installing

TODO

### Execution

TODO

## License

This work is licensed under the conditions of the BSD license, see LICENSE.txt.

