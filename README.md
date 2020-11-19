[![codecov](https://codecov.io/gh/MicroPhen/pyFOOMB/branch/main/graph/badge.svg?token=7WALTIPP6O)](https://codecov.io/gh/MicroPhen/pyFOOMB)
[![Tests](https://github.com/MicroPhen/pyFOOMB/workflows/Tests/badge.svg)](https://github.com/MicroPhen/pyFOOMB/actions)
[![DOI](https://zenodo.org/badge/309308898.svg)](https://zenodo.org/badge/latestdoi/309308898)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# pyFOOMB

__*Py*thon *F*ramework for *O*bject *O*riented *M*odelling of *B*ioprocesses__

Intented application is the acessible modelling of simple to medium complex bioprocess models, by programmatic means. In contrast to 'full-blown' software suites, `pyFOOMB` can be used by scientists with little programming skills in the easy-access language Python.
`pyFOOMB` comes with a MIT license, and anyone interested in using, understanding, or contributing to pyFOOMB is happily invited to do so.

`pyFOOMB` relies on the `assimulo` package (<https://jmodelica.org/assimulo>), providing an interface to the SUNDIALS CVode integrator for systems of differential equations, as well as event handling routines. For optimization, i.e. model calibration from data, the `pygmo` package is used, which provides Python bindings for the `pagmo2` package implementing the Asynchronous Generalized Islands Model.

To faciliate rapid starting for new users, a continously growing collection of Jupyter notebooks is provided. These serve to demonstrate basic and advanced concepts and functionalities (also beyond the pure functions of the `pyFOOMB` package). Also, the examples can be used as building blocks for developing own bioprocess models and corresponding workflows. 

Check also our [preprint](https://www.biorxiv.org/content/10.1101/2020.11.10.376665v1) at bioRχiv introducing `pyFOOMB` with two more elaborated application examples that reproduce real-world data from literature.

Literature:

* Andersson C, Führer C, and Akesson J (2015). Assimulo: A unified framework for ODE solvers. _Math Comp Simul_ __116__:26-43
* Hindmarsh AC, _et al_ (2005). SUNDIALS: Suite of nonlinear and differential/algebraic equation solvers. _ACM Trans Math Softw_ __31__:363-396
* Biscani F, Izzo D (2020). A parallel global multiobjective framework for optimization: pagmo. _J Open Source Softw_ __5__:2338

## Requirements (provided as environment.yml)

* python 3.7 or 3.8
* numpy
* scipy
* joblib
* pandas
* xlrd
* openpyxl
* matplotlib
* seaborn
* jupyter
* psutil
* assimulo (via conda-forge)
* pygmo (via conda-forge)

## Installation

1) Download the code repository to your computer, this is done the best way using `git clone`: In a shell, navigate to the folder where you want the repository to be located.
2) Open a terminal / shell and clone the repository
via `git clone https://github.com/MicroPhen/pyFOOMB.git`
3) cd (*change directory*) into the newly cloned repository : `cd pyfoomb`
4) Verify that you are in the repo folder, where the file `environment.yml` is found (`dir` for Windows, `ls` for Linux/Mac).
5) Exceute `conda env create -f environment.yml`.
This will create a conda environment named `pyfoomb`, with the current version of the just cloned git repository.
6) To make sure, your environment will refer always the state of your git repo (i.e., after own code modifications or after pulling from remote), run `pip install -e ../pyfoomb`. 
