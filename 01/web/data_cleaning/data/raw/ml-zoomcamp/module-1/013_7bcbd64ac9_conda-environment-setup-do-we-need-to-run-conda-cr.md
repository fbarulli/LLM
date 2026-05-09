---
id: 7bcbd64ac9
question: 'Conda Environment Setup: Do we need to run ''conda create'' and ''conda
  activate'' every time?'
sort_order: 13
---

To set up a Conda environment for the project:

- **Initial Setup**: Run the following command only once to create the environment:
  
  ```bash
  conda create -n ml-zoomcamp
  ```

- **Activating Environment**: Each time you want to work on the project, activate the environment:
  
  ```bash
  conda activate ml-zoomcamp
  ```

- **Exporting Environment**: To export your existing environment to a YAML file:
  
  ```bash
  conda env export > environment.yml
  ```

- **Recreating Environment**: Use the YAML file to recreate the environment:
  
  ```bash
  conda env create -f environment.yml
  ```