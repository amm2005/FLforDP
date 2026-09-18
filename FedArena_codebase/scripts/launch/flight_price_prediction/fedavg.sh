#!/usr/bin/env bash
source .venv/bin/activate

# FedAvg tuning for the paper Flight Price Prediction task.
# The loader reads Kaggle's Clean_Dataset.csv and creates features and splits online.
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=flight_price_prediction dataset@test_dataset=flight_price_prediction \
    distribution=column distribution.client_column=airline \
    losses@loss=mse trainer=tabular_trainer \
    tuning.metric=R2 federated_params.server_saving_metrics=[R2] tuning.direction=maximize \
    federated_params.amount_of_clients=6 \
    federated_params.client_train_frac=0.7 federated_params.client_val_frac=0.15 federated_params.client_test_frac=0.15 \
    training_params.batch_size=1024 manager.batch_size=6 \
    train_dataset.data_sources.train_map_file=[data/flight_price_prediction/Clean_Dataset.csv] \
    optimizer=adamw \
    tuning.n_trials=50 \
    tuning.study_name=adamw_nopruner
