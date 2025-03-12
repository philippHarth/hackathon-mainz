# Dissecting origins of wiring specificity

## Minimal Local Setup

### Windows 
Excute the following commands using the PowerShell:
```
cd <path-to-repository>
conda create -n hackathon python=3.11
conda activate hackathon
pip install -U -r requirements-win.txt   
```
Register the conda environment with **ipykernel** to use it from within Jupyter notebooks:
```
conda activate hackathon:
python -m ipykernel install --user --name i --display-name "hackathon"
```

### Linux
As above, but use a regular command shell and the requirements-linux.txt-file: 
```
pip install -U -r requirements-linux.txt   
```


## Optional Installs
Optional: manually install a [CuPy version](https://docs.cupy.dev/en/stable/install.html) that matches your local GPU setup, e.g.:
```
pip install cupy-cuda12x==13.2.0
```

Optional: Create cuDF-based Python Environment for Synapse Data Preprocessing.
Requires [cuDF](https://github.com/rapidsai/cudf), which has extended GPU hardware requirements, **only supported under linux**; please adjust for your local GPU setup.
```
cd <path-to-repository>
conda create -n preproc python=3.11
conda activate preproc 

pip install cudf-cu11==24.6.0
pip install ipykernel==6.29.5
```
Register Jupyter notebook kernel:
```
python -m ipykernel install --user --name i --display-name "preproc"
```