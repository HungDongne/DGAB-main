# DGAB-main

A Financial Fraud Detection Framework.

## Usage

### Data processing

1. Run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets; 
2. Run `python feature_engineering/data_process.py` to pre-process all datasets needed in this repo.

### Training & Evaluation
To run the `DGAB` model, simply execute:
```bash
python main.py
```
By default, it uses the `S-FFSD` dataset. You can specify other datasets and seeds:
```bash
python main.py --dataset yelp --seed 42
```
The configuration file can be found in `config/dgab_cfg.yaml`.

## Repo Structure
The repository is organized as follows:
- `data/`: dataset files;
- `config/`: configuration files for the DGAB model;
- `feature_engineering/`: data processing scripts;
- `methods/dgab/`: implementation of the DGAB model;
- `main.py`: main entry point for the DGAB framework;
- `requirements.txt`: package dependencies;

## Requirements
```text
torch==2.4+cu124
dgl
pydantic
torchdata==0.7.1
numpy<2.0
pandas
scikit-learn
matplotlib
seaborn
rtdl_num_embeddings
torch_geometric
```