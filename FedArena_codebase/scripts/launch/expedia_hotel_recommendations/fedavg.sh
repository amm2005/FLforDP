#!/usr/bin/env bash
source .venv/bin/activate

# FedAvg tuning for Expedia (100 classes; clients are point-of-sale sites).
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=expedia dataset@test_dataset=expedia \
    distribution.client_column=site_name \
    losses@loss=ce trainer=tabular_trainer \
    tuning.metric=MAP@5 federated_params.server_saving_metrics=[MAP@5] tuning.direction=maximize \
    federated_params.amount_of_clients=12 \
    federated_params.client_train_frac=0.7 federated_params.client_val_frac=0.15 federated_params.client_test_frac=0.15 \
    training_params.batch_size=1024 manager.batch_size=12 \
    optimizer=adamw \
    tuning.study_name=adamw_new_dataset \
    train_dataset.data_sources.train_map_file=[data/expedia_hotel_recommendations/expedia_6000_100000.parquet]
