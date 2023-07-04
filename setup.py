from setuptools import setup, find_packages

setup(
    name="runge_kutta_openmdao",
    version="0.1",
    description="Runge-Kutta methods for OpenMDAO",
    author="DLR-SP-SUM",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "scipy",
        "numpy",
        "openmdao",
        "matplotlib",
        "h5py",
        "pytest",
        "numba",
        "pyrevolve",
    ],
)
