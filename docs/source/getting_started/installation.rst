Installation
============

This page covers how to install ws3 and its dependencies for development.

System Requirements
-------------------

ws3 requires:

- **Python**: 3.9 or later (3.12 recommended)
- **Operating System**: Linux, macOS, or Windows
- **Disk Space**: ~500 MB for package + dependencies
- **RAM**: 4+ GB recommended for large models

Optional Dependencies
---------------------

ws3 has several optional dependency groups. Install only what you need:

.. code-block:: bash

   # Core package (required)
   pip install ws3

   # Development dependencies (testing, linting)
   pip install "ws3[dev]"

   # Documentation (Sphinx, napoleon, nbsphinx)
   pip install "ws3[docs]"

   # Gurobi solver (commercial license required)
   pip install "ws3[gurobi]"

   # All optional dependencies
   pip install "ws3[dev,docs,gurobi]"

Installation from Source
------------------------

For development, install from the repository:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/UBC-FRESH/ws3.git
   cd ws3

   # Create a virtual environment
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate

   # Install in development mode with all extras
   pip install -e ".[dev,docs]"

   # Verify installation
   python -c "import ws3; print(f'ws3 {ws3.__version__}')"

Platform-Specific Notes
-----------------------

Linux
~~~~~

On Ubuntu/Debian, you may need system packages for geospatial dependencies:

.. code-block:: bash

   sudo apt-get install -y \
       libgdal-dev \
       libgeos-dev \
       libproj-dev \
       pandoc

macOS
~~~~~

On macOS, install Homebrew dependencies first:

.. code-block:: bash

   brew install gdal geos proj pandoc

Windows
~~~~~~~

On Windows, use conda for geospatial dependencies:

.. code-block:: powershell

   conda create -n ws3 python=3.12
   conda activate ws3
   conda install -c conda-forge gdal geos proj pandoc
   pip install -e ".[dev,docs]"

Known Compatibility Issues
--------------------------

PaCal Library
~~~~~~~~~~~~~

The PaCal library (used for some growth curve calculations) has known
compatibility issues with newer numpy versions. ws3 sets
``PACAL_BROKEN = True`` in ``ws3.common`` to work around this.

Functions that depend on PaCal will not work without a patched version.
If you need PaCal functionality, use numpy < 2.0 or apply the ws3
workaround.

Gurobi License
~~~~~~~~~~~~~~

The Gurobi solver requires a commercial license. If you don't have one,
ws3 will fall back to HiGHS (via PuLP) as the default solver.

To use Gurobi:

1. Install Gurobi: ``pip install gurobi``
2. Set your Gurobi license key: ``export GRB_LICENSE_FILE=/path/to/license``
3. Install ws3 with Gurobi support: ``pip install "ws3[gurobi]"``