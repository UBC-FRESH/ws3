[![Tests](https://github.com/UBC-FRESH/ws3/actions/workflows/ci.yml/badge.svg?branch=feature%2Fpytest)](https://github.com/UBC-FRESH/ws3/actions/workflows/ci.yml)

# ws3 - Wood Supply Simulation System

**ws3** (Wood Supply Simulation System) is a Python package for modeling landscape-level wood supply planning problems.

Read the tutorial [here](https://egh-ws3.readthedocs.io/en/latest/index.html).

## Table of Contents

- [Installation](#installation) 
- [Modules](#modules)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Installation

We recommend installing `ws3` package into a Python venv (virtual environment) to minimize interactions with system-level packages. 

In [**000_venv_python_kernel_setup.ipynb**](https://github.com/ghasemiegh/ws3/blob/dev/examples/000_venv_python_kernel_setup.ipynb) we provide the instructions for how to set up a new venv-sandboxed Python kernel and make it available in your JupyterLab environment, assuming that you are running this notebook in a standard linux-based environment and a regular (non-root) using running commands in a bash terminal. 

## Modules 

`ws3` consists of the following main modules:

- **common.py**: Contains definitions for global attributes, functions, and classes that might be used anywhere in the package.
- **core.py**: Contains `Interpolator` class used by ``Curve`` class to interpolate between real data points.
- **forest.py**: Implements functions for building and running wood supply simulation models.
- **opt.py**: Implements functions for formulating and solving optimization problems. 
- **spatial.py**: Implements the `ForestRaster` class, which can be used to allocate an aspatial disturbance schedule (for example, an optimal solution to a wood supply problem generated by an instance of the `forest.ForestModel` class) to a rasterized representation of the forest inventory.

## Usage 

Multiple examples are available to demonstrate the utilization of ws3. Below is an overview explaining each of these examples:

- [**010_ws3_model_example-fromscratch.ipynb**](https://github.com/ghasemiegh/ws3/blob/dev/examples/010_ws3_model_example-fromscratch.ipynb): This example builds a new `ws3` model from scratch.
- [**020_ws3_model_example-woodstock.ipynb**](https://github.com/ghasemiegh/ws3/blob/dev/examples/020_ws3_model_example-woodstock.ipynb): This example builds a `ws3` model from Woodstock-format text input files.
- [**030_ws3_libcbm_sequential-fromscratch.ipynb**](https://github.com/ghasemiegh/ws3/blob/dev/examples/030_ws3_libcbm_sequential-fromscratch.ipynb): This example creates the linkages between `ws3` and `libcbm` from scratch (i.e., all code required to create these linkages is developed directly in this notebook).
- [**031_ws3_libcbm_sequential-builtin.ipynb**](https://github.com/ghasemiegh/ws3/blob/dev/examples/031_ws3_libcbm_sequential-builtin.ipynb): This example replicates what **030_ws3_libcbm_sequential-fromscratch.ipynb** does, but using `ws3` built-in `CBM` linkage functions.
- [**040_ws3_libcbm_neilsonhack-fromscratch.ipynb**](https://github.com/ghasemiegh/ws3/blob/dev/examples/040_ws3_libcbm_neilsonhack-fromscratch.ipynb): This example shows how to implement the Neilson hack (i.e., generate carbon yield curves from a CBM for use in a forest estate model) using `ws3` and `libcbm`.

## License

**MIT License**
Copyright (c) 2015-2024 Gregory Paradis.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

## Acknowledgments

TBD

## Contact

For questions, feedback, or issues related to the project, please contact:
Gregory Paradis - gregory.paradis@ubc.ca
