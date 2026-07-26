.. _howto-reproducibility:

=================
Reproducibility Workflows
=================

Goal
----

Set up reproducible workflows for ws3 modeling:

* Version control for models and data
* Automated validation checks
* Documentation of model configuration
* Reproducible analysis pipelines

Prerequisites
-------------

* Completed :doc:`model-validation`
* Familiarity with version control (Git)
* A working ws3 installation

Step-by-Step Instructions
-------------------------

**Step 1: Organize Project Structure**

.. code-block:: text

   my_ws3_project/
   ├── data/
   │   ├── inventory.csv
   │   └── observed/
   ├── scripts/
   │   ├── prepare_data.py
   │   ├── run_model.py
   │   └── validate.py
   ├── configs/
   │   ├── model_config.yaml
   │   └── optimization_params.yaml
   ├── results/
   │   ├── schedules/
   │   └── validation/
   └── README.md

**Step 2: Create Configuration Files**

.. code-block:: yaml

   # configs/model_config.yaml
   model:
     name: "My WS3 Model"
     horizon: 5
     periods: [0, 1, 2, 3, 4]

   development_types:
     - code: "DT001"
       species: "SP"
       site_index: 50
       age: 10
       area: 100.0

   actions:
     - code: "CLEARCUT"
       descr: "Clearcut harvest"
       components: ["volume"]
       transitions:
         DT001: DT001_REGEN

**Step 3: Create Data Preparation Script**

.. code-block:: python

   # scripts/prepare_data.py
   import pandas as pd
   import yaml

   # Load configuration
   with open('configs/model_config.yaml', 'r') as f:
       config = yaml.safe_load(f)

   # Load and validate inventory
   df = pd.read_csv('data/inventory.csv')

   # Apply configuration filters
   df = df[df['species'] == config['development_types'][0]['species']]

   # Save processed data
   df.to_csv('data/inventory_processed.csv', index=False)

**Step 4: Create Model Run Script**

.. code-block:: python

   # scripts/run_model.py
   import yaml
   from ws3.forest import ForestModel
   from ws3.opt import solve_optimization

   # Load configuration
   with open('configs/model_config.yaml', 'r') as f:
       config = yaml.safe_load(f)

   # Create and configure model
   model = ForestModel()

   # Add development types
   for dt in config['development_types']:
       model.add_development_type(
           code=dt['code'],
           species=dt['species'],
           site_index=dt['site_index'],
           age=dt['age'],
           area=dt['area']
       )

   # Add actions
   for action in config['actions']:
       model.add_action(**action)

   # Run optimization
   solution = solve_optimization(
       model=model,
       horizon=config['model']['horizon'],
       objective='maximize_volume'
   )

   # Save results
   schedule = solution.get_schedule()
   schedule.to_csv('results/schedules/schedule.csv', index=False)

**Step 5: Create Validation Script**

.. code-block:: python

   # scripts/validate.py
   import pandas as pd

   # Load schedule
   schedule = pd.read_csv('results/schedules/schedule.csv')

   # Run validation checks
   total_area = schedule['area_ha'].sum()
   print(f"Total scheduled area: {total_area:.1f} ha")

   # Check for negative areas
   if (schedule['area_ha'] < 0).any():
       print("ERROR: Negative areas found!")
   else:
       print("All areas are non-negative")

   # Check for missing periods
   periods = sorted(schedule['period'].unique())
   expected_periods = list(range(5))
   if periods != expected_periods:
       print(f"WARNING: Missing periods: {set(expected_periods) - set(periods)}")

**Step 6: Document Workflow**

.. code-block:: markdown

   # README.md
   ## My WS3 Project

   ### Setup
   1. Install dependencies: `pip install -r requirements.txt`
   2. Prepare data: `python scripts/prepare_data.py`
   3. Run model: `python scripts/run_model.py`
   4. Validate: `python scripts/validate.py`

   ### Configuration
   - Model config: `configs/model_config.yaml`
   - Optimization params: `configs/optimization_params.yaml`

**Step 7: Version Control**

.. code-block:: bash

   git init
   git add .
   git commit -m "Initial model setup"

   # Add .gitignore
   echo "results/" >> .gitignore
   echo "__pycache__/" >> .gitignore
   echo "*.pyc" >> .gitignore
   git add .gitignore
   git commit -m "Add .gitignore"

Expected Output
---------------

* Organized project structure
* Configuration files for reproducibility
* Automated scripts for data prep, model run, and validation
* Version-controlled project

Troubleshooting
---------------

**Issue: Scripts fail**

* Check file paths are correct
* Verify configuration YAML is valid
* Ensure all dependencies are installed

**Issue: Results not reproducible**

* Check that random seeds are set if using stochastic elements
* Verify all inputs are version-controlled
* Document all parameter values

**Issue: Validation fails**

* Check data quality
* Verify model configuration
* Review validation logic

Next Steps
----------

* :doc:`model-validation` — Validate your model
* :doc:`running-optimization` — Run optimization
* :doc:`custom-area-selector` — Custom area selection