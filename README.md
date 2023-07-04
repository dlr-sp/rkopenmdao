# Runge-Kutta-OpenMDAO

Prototype for solving time-dependent problems in OpenMDAO using Runge-Kutta-schemes.

## Description

Runge-Kutta-schemes are widely used methods for solving initial value problems. Implementing this into OpenMDAO is done by using nesting:
An inner OpenMDAO-Problem is used to model one Runge-Kutta stage of an instationary multidisciplinary problem.
An outer explicit component loops over the time steps and time stages, running the inner problem to compute the stage values.

## Getting started

### Installing

Start by cloning this repository, e.g. with
    
    TODO
    git clone github-url /path/to/directory

and use

    cd /path/to/directory

to change into the directory.
Then, if you aren't already in a python environment, create one with

    python -m venv /path/to/venv

and source it via

    source /path/to/venv/bin/activate

Then use

    pip install .

to install the package.
### Execution

In the directory 

    src/runge_kutta_openmdao/examples

of this repository are files for the solution of a heat equation, once the analytical solution, once via a monolithic Runge-Kutta time integration, and once via a nested approach in OpenMDAO where additionally the domain is split into two halves.
When you are in a virtual environment as described above,

    python /path/to/src/runge_kutta_openmdao/examples/example.py

lets you execute an example. These write HDF5-files containing a time series of the numerical solution to the directory from which you executed the examples.

## License

This work is licensed under the conditions of the BSD license, see LICENSE.txt.

