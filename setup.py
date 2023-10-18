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
