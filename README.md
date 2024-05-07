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

In the examples directory of this repository are files for the solution of a heat equation, some using OpenMDAO and this prototype, and for comparison an analytical solution and another discretized solution where neither OpenMDAO nor this extension is used.
Their purpose is mainly as a mathematical example to show how this library is meant to be used.
When you are in a virtual environment as described above,

    python /path/to/examples/*example_file*.py

lets you execute an example. These write HDF5-files containing a time series of the numerical solution to the directory from which you executed the examples.
If you want a more guided explanation, you can look at

    doc/user_guide.ipynb

which is a small Jupyter file that shows the main ways of using this extension.
## License

This work is licensed under the conditions of the BSD license, see LICENSE.txt.

The software is provided as is.
We sincerely welcome your feedback on issues, bugs and possible improvements.
Please use the issue tracker of the project for the corresponding communication or make a fork.
Our priority and time line for working on the issues depend on the project and its follow ups.
This may lead to issue and tickets, which are not pursued.
In case you need an urgent fix, please contact us directly for discussing possible forms of collaboration (direct contribution, projects, contracting, ...): Institute of Software Methods for Product Virtualization.

