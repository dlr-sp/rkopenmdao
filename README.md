# RKOpenMDAO: Runge-Kutta methods in OpenMDAO

Prototype for solving time-dependent problems in [OpenMDAO](https://openmdao.org/) using Runge-Kutta-schemes.

## Description

Runge-Kutta-schemes are widely used methods for solving initial value problems. Implementing them into OpenMDAO is done by using nesting:
An inner OpenMDAO-Problem is used to model one Runge-Kutta stage of an unsteady multidisciplinary problem.
An outer explicit component loops over the time steps and time stages, running the inner problem to compute the stage updates.

At the current time, diagonally-implicit and explicit Runge-Kutta schemes are supported.

## Getting started

### Installing

Start by cloning this repository, e.g. with
```bash
git clone https://github.com/dlr-sp/rkopenmdao.git /path/to/directory
```
and use
```bash
cd /path/to/directory
```
to change into the directory.
Then, if you aren't already in a python environment, create one with
```bash
python -m venv /path/to/venv
```
and source it via
```bash
source /path/to/venv/bin/activate
```
Then use
```bash
pip install .
```
to install the package (read-only).

Note that the file-writing features of RKOpenMDAO use [h5py](https://docs.h5py.org/en/stable/index.html). 
Prebuilt versions of it (like via pip) usually don't come with MPI support.
If you want to use file-writing in conjunction with MPI, you will need to install h5py manually.
When already having a parallel HDF5 install, this can in most cases be done with 
```bash
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-binary=h5py h5py
```
For more information, please have a look at https://docs.h5py.org/en/stable/build.html and https://docs.h5py.org/en/stable/mpi.html.


If you want to also install dev-dependencies, use
```bash
pip install ".[dev]"
```
instead.

Editable installation (Recommended for developers):
To modify RKOpenMDAO, it's recommended to install it in *[editable](https://setuptools.pypa.io/en/latest/setuptools.html#development-mode)* (development) mode by using the `-e` flag with `pip`.
This ensures that any changes you make to the source code are reflected when you import RKOpenMDAO in Python.
Additionally, you should install the required packages for running RKOpenMDAO's tests and documentation generator.
You can install all necessary development dependencies by running:

```bash
pip install -e ".[dev]"
```

### Execution

In the examples directory of this repository are files for the solution of a heat equation, some using OpenMDAO and this prototype, and for comparison an analytical solution and another discretized solution where neither OpenMDAO nor this extension is used.
Their purpose is mainly as a mathematical example to show how this library is meant to be used.
When you are in a virtual environment as described above,
```bash
python /path/to/examples/*example_file*.py
```
lets you execute an example. These write HDF5-files containing a time series of the numerical solution to the directory from which you executed the examples.
If you want a more guided explanation, you can look at
```bash
doc/user_guide.ipynb
```
which is a small Jupyter file that shows the main ways of using this extension.
## License

This work is licensed under the conditions of the BSD-3-Clause license, see LICENSE.txt.

The software is provided as is.
We sincerely welcome your feedback on issues, bugs and possible improvements.
Please use the issue tracker of the project for the corresponding communication or make a fork.
Our priority and time line for working on the issues depend on the project and its follow ups.
This may lead to issue and tickets, which are not pursued.
In case you need an urgent fix, please contact us directly for discussing possible forms of collaboration (direct contribution, projects, contracting, ...): [Institute of Software Methods for Product Virtualization](https://www.dlr.de/sp).

