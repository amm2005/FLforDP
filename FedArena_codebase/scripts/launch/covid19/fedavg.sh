#!/usr/bin/env bash
source .venv/bin/activate

# FedAvg tuning for the ten-country COVID-19 paper task.
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=covid19 dataset@test_dataset=covid19 \
    distribution.client_column=country \
    losses@loss=mse trainer=tabular_trainer \
    tuning.metric=R2 federated_params.server_saving_metrics=[R2] tuning.direction=maximize \
    federated_params.amount_of_clients=10 \
    federated_params.client_train_frac=0.6 federated_params.client_val_frac=0.2 federated_params.client_test_frac=0.2 \
    training_params.batch_size=512 training_params.num_workers=0 manager.batch_size=5 \
    train_dataset.data_sources.train_map_file=[data/covid19/train_df.csv] \
    train_dataset.data_sources.test_map_file=[data/covid19/test_df.csv] \
    optimizer=adamw \
    tuning.n_trials=50 \
    tuning.study_name=adamw_nopruner
