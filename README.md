# Neural Network MTD
Code repository for paper: "AI-MTD: Zero-Trust Artificial Intelligence Model Security Based on Moving Target Defense"

## Setup
- Create Conda environment: `conda create -f environment_cuda.yml -n mtd`
- Activate the environment: `conda activate mtd`
- Install `model_mtd`: `pip install -e .`
- Run Tests (Optional): `pytest -v`

## Running Experiment Scripts:
- Model evaluation before/after MTD: `nohup python -u scripts/mtd_model_eval.py > out_eval.txt`
- Runtime measurements: `nohup python -u scripts/mtd_model_time.py > out_time.txt`

## Visualize Results:
- Use [results_process.ipynb](results/results_process.ipynb) to visualize the results from the experiments.
