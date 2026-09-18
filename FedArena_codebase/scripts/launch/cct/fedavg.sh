#!/usr/bin/env bash
source .venv/bin/activate

# FedAvg tuning for the five-camera CCT paper task.
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=cct \
    dataset@test_dataset=cct \
    distribution=column \
    distribution.client_column=location \
    models@models_dict.model1=resnet18_groupnorm \
    losses@loss=ce \
    trainer=cifar_trainer \
    training_params.batch_size=256 \
    training_params.num_workers=0 \
    federated_params.amount_of_clients=5 \
    train_dataset.n_clients_select=5 \
    federated_params.client_train_frac=0.7 federated_params.client_val_frac=0.15 federated_params.client_test_frac=0.15 \
    manager.batch_size=5 \
    federated_params.server_saving_metrics=[macro_AP] \
    federated_params.communication_rounds=50 \
    tuning.metric=macro_AP \
    tuning.direction=maximize \
    tuning.n_trials=50 \
    tuning.study_name=adamw_nopruner_256 \
    optimizer=adamw \
    train_dataset.data_sources.train_map_file=[data/cct/caltech_images_20210113.json] \
    train_dataset.data_sources.test_map_file=[data/cct/caltech_images_20210113.json] \
    train_dataset.images_dir=data/cct/cct_images_256
