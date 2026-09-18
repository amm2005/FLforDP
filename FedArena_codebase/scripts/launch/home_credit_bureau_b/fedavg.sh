#!/usr/bin/env bash
source .venv/bin/activate


# ---------------------------------------------------------------------------
# RUN 2 (active) — variant B (natural history share) top-5 + batch_size 256

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python src/tune.py \
    federated_method=fedavg \
    dataset@train_dataset=home_credit_bureau_b dataset@test_dataset=home_credit_bureau_b \
    distribution.client_column=credor_3940957M \
    losses@loss=ce trainer=tabular_trainer \
    tuning.metric=ROC-AUC federated_params.server_saving_metrics=[ROC-AUC] tuning.direction=maximize \
    tuning.n_trials=50 \
    federated_params.amount_of_clients=5 \
    federated_params.client_train_frac=0.7 \
    federated_params.client_val_frac=0.15 \
    federated_params.client_test_frac=0.15 \
    training_params.batch_size=256 \
    training_params.num_workers=0 \
    manager.batch_size=5 \
    optimizer=adamw \
    tuning.study_name=adamw_nq_bs256_natural \
    train_dataset.data_sources.train_map_file=[data/home_credit_bureau_b/bureau_b_h6_natural.parquet]
