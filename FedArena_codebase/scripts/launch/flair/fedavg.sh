#!/usr/bin/env bash
source .venv/bin/activate

# FedAvg tuning for the top-ten-user FLAIR paper task.
CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=flair \
    dataset@test_dataset=flair \
    distribution=column \
    distribution.client_column=user_id \
    models@models_dict.model1=resnet18_groupnorm \
    losses@loss=bce \
    trainer=cifar_trainer \
    training_params.batch_size=256 \
    training_params.num_workers=5 \
    federated_params.amount_of_clients=10 \
    train_dataset.n_clients_select=10 \
    manager.batch_size=5 \
    federated_params.client_train_frac=0.6 federated_params.client_val_frac=0.2 federated_params.client_test_frac=0.2 \
    train_dataset.images_dir=data/flair/small_images \
    "train_dataset.data_sources.train_map_file=[data/flair/labels_and_metadata.json]" \
    federated_params.server_saving_metrics=[macro_AP] \
    federated_params.communication_rounds=50 \
    tuning.metric=macro_AP \
    tuning.direction=maximize \
    tuning.n_trials=50 \
    tuning.study_name=adamw_nopruner \
    optimizer=adamw
