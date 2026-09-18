#!/usr/bin/env bash
source .venv/bin/activate

# FedAvg tuning on Fannie Mae (loan default within H months, binary classification;
# metric ROC-AUC; FL clients = seller/originating bank; OOT split train=2012Q1 70%,
# val=2012Q2, test=2012Q3 is BAKED into the parquet by scripts/prepare_fannie_mae.py —
# client_*_frac overrides are not needed and would be ignored).
# Numerical normalization: add train_dataset.num_policy=noisy_quantile to switch.
# To run another variant (d60 / oot15 / ...): regenerate with prepare_fannie_mae.py
# and override train_dataset.data_sources.train_map_file=[<path>].
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=fannie_mae dataset@test_dataset=fannie_mae \
    distribution.client_column=seller \
    losses@loss=ce trainer=tabular_trainer \
    tuning.metric=ROC-AUC federated_params.server_saving_metrics=[ROC-AUC] tuning.direction=maximize \
    federated_params.amount_of_clients=9 \
    training_params.batch_size=1024 \
    manager.batch_size=9 \
    train_dataset.data_sources.train_map_file=[data/fannie_mae/fannie_mae_d90_h84_oot100_ex-wells.parquet] \
    optimizer=adamw \
    tuning.n_trials=50 \
    training_params.num_workers=0 \
    tuning.study_name=adamw_nopruner
