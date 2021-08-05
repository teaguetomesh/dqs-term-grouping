import io
from setuptools import find_packages, setup

# This reads the __version__ variable from dqs/_version.py
__version__ = ""
exec(open("dqs/_version.py").read())

name = "Digital Quantum Simulation"

description = "Improve the execution of Hamiltonian simulation algorithms by optimizing the program execution order"

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()

# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

dqs_packages = ["dqs"] + [
    "dqs." + package for package in find_packages(where="dqs")
]

# Sanity check
assert __version__, "Version string cannot be empty"

setup(
    name=name,
    version=__version__,
    url="https://github.com/SupertechLabs/SuperstaQ",
    author="Teague Tomesh",
    author_email="ttomesh@princeton.edu",
    python_requires=(">=3.8.0"),
    install_requires=requirements,
    license="N/A",
    description=description,
    long_description=long_description,
    packages=superstaq_packages,
    package_data={"dqs": ["py.typed"]},
)
