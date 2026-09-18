# FedArena

Federated learning benchmark spanning tabular, vision, and language modalities with natural client partitions.

## 1. Requirements

- Python 3.10.x
- CUDA 11.8+ (GPU required for training)
- Linux (tested on Ubuntu 22.04)

## 2. Installation

```bash
git clone <repo-url>
cd Federated-Research-LIB-dev
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. Data preparation

Before running experiments, prepare the required datasets according to the instructions in:

[DATASET_PIPELINES.md](DATASET_PIPELINES.md)

This document describes dataset-specific download instructions, preprocessing steps, expected directory structure, and configuration requirements.

After preparing the data, update the paths in `src/configs/dataset/*.yaml` to point to your local dataset locations.

## 4. Running an experiment

### Single method on a dataset

```bash
source .venv/bin/activate

# Example: FedAvg on covid19
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=covid19 dataset@test_dataset=covid19 \
    distribution.client_column=country \
    losses@loss=mse trainer=tabular_trainer \
    tuning.metric=R2 federated_params.server_saving_metrics=[R2] \
    tuning.direction=maximize \
    optimizer=adamw \
    tuning.study_name=fedavg_covid
```

### Using launch scripts

Pre-configured launch scripts are in `scripts/launch/<dataset>/<method>.sh`:

```bash
bash scripts/launch/covid19/fedavg.sh
```

Edit `CUDA_VISIBLE_DEVICES` in the script to select a free GPU.

### Methods requiring a tuned FedAvg baseline

Some federated methods reuse hyperparameters (`lr`, `weight_decay`, and `round_epochs`) from a previously tuned **FedAvg** study.

For such methods:

1. Run FedAvg first. This creates

   ```
   results/<dataset>/fedavg/<study>/meta.json
   ```

   containing the selected hyperparameters.

2. Then run the desired method and specify the corresponding FedAvg study:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python src/tune.py \
       federated_method=<method> \
       federated_method.fedavg_study_name=<fedavg_study_name> \
       dataset@train_dataset=covid19 dataset@test_dataset=covid19 \
       ...
       tuning.study_name=<study_name>
   ```

## 5. Key parameters

| Parameter | Description |
|-----------|-------------|
| `federated_method` | Method name (fedavg, pfedme, ditto, fedrep, fedprox, feddyn, scaffold, fednova, fedspeed, mime, mimelite, fedyogi, local_only) |
| `dataset@train_dataset` | Dataset config name (covid19, flight_price_prediction, expedia, fannie_mae, bureau_b, cct, flair, fed_isic2019, amazon, news_agg, yelp, peta) |
| `distribution.client_column` | Column defining the federated partition (e.g., country, airline, seller, center) |
| `trainer` | `tabular_trainer`, `cifar_trainer`, or `text_trainer` |
| `tuning.metric` | Primary metric for model selection (R2, ROC-AUC, macro_AP, MAP@5, balanced_accuracy) |
| `tuning.study_name` | Unique name for the Optuna study (avoids collisions) |
| `federated_params.amount_of_clients` | Number of federated clients |
| `federated_params.communication_rounds` | Number of communication rounds |
| `training_params.batch_size` | Local batch size |
| `manager.batch_size` | Number of clients processed in parallel per round (must divide amount_of_clients) |

## 6. Output structure

```
results/
  <dataset>/
    <method>/
      <study_name>/
        meta.json          # best trial params, objective value
        tables/             # multi-seed CSV tables
outputs/                     # run logs (gitignored)
tuning_studies/              # Optuna .db files (gitignored)
```
