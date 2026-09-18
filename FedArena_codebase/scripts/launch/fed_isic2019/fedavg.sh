#!/usr/bin/env bash
source .venv/bin/activate

# FedAvg tuning on the six-center Fed-ISIC2019 paper task.
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=fed_isic2019 \
    dataset@test_dataset=fed_isic2019 \
    distribution=column \
    distribution.client_column=center \
    models@models_dict.model1=resnet18_groupnorm \
    losses@loss=focal_weighted \
    trainer=cifar_trainer \
    training_params.batch_size=256 \
    training_params.num_workers=5 \
    federated_params.amount_of_clients=6 \
    manager.batch_size=6 \
    federated_params.server_pool_enabled=True \
    federated_params.client_train_frac=0.7 \
    federated_params.client_val_frac=0.2 \
    federated_params.client_test_frac=0.1 \
    federated_params.server_val_frac=0.5 \
    federated_params.server_test_frac=0.0 \
    train_dataset.images_dir=data/fed_isic2019/ISIC_2019_Training_Input_preprocessed \
    "train_dataset.data_sources.train_map_file=[data/fed_isic2019/fed_isic2019_map.csv]" \
    "train_dataset.data_sources.test_map_file=[data/fed_isic2019/fed_isic2019_map.csv]" \
    federated_params.server_saving_metrics=[balanced_accuracy] \
    federated_params.communication_rounds=50 \
    tuning.metric=balanced_accuracy \
    tuning.direction=maximize \
    tuning.n_trials=50 \
    tuning.study_name=adamw_nopruner \
    optimizer=adamw
