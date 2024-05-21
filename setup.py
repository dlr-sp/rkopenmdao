from setuptools import setup, find_packages

setup(
    name="rkopenmdao",
    version="0.1",
    description="Runge-Kutta time integration in OpenMDAO",
    author="Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR)",
    author_email="florian.ross@dlr.de",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "openmdao",
        "scipy",
        "numpy",
        "matplotlib",
        "h5py",
        "numba",
        "pyrevolve",
        "mpi4py",
        "petsc4py",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mpi",
            "pylint",
        ]
    },
)
