#!/usr/bin/env bash

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=6 python src/tune.py \
    federated_method=fedprox \
    models@models_dict.model1=bert_tiny \
    dataset@train_dataset=news_agg dataset@test_dataset=news_agg \
    distribution.client_column=PUBLISHER distribution.top_n_clients=10 \
    losses@loss=ce trainer=text_classification_trainer \
    tuning.metric=macro_AP federated_params.server_saving_metrics=[macro_AP] tuning.direction=maximize \
    federated_params.client_train_frac=0.7 federated_params.client_val_frac=0.15 federated_params.client_test_frac=0.15 \
    federated_params.amount_of_clients=10 \
    training_params.batch_size=128 manager.batch_size=10 \
    tuning.study_name=adamw \
    optimizer=adamw tuning.n_trials=0