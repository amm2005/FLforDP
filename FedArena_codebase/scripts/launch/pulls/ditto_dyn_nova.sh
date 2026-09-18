export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=3 python src/tune.py \
    federated_method=ditto \
    dataset@train_dataset=home_credit_bureau_b dataset@test_dataset=home_credit_bureau_b \
    distribution.client_column=credor_3940957M \
    losses@loss=ce trainer=tabular_trainer \
    tuning.metric=ROC-AUC federated_params.server_saving_metrics=[ROC-AUC] tuning.direction=maximize \
    tuning.n_trials=50 \
    federated_params.amount_of_clients=5 \
    training_params.batch_size=256 \
    training_params.num_workers=0 \
    manager.batch_size=5 \
    optimizer=adamw \
    train_dataset.num_policy=noisy_quantile \
    tuning.study_name=adamw_nq_bs256_natural_2 \
    train_dataset.data_sources.train_map_file=[data/home_credit_bureau_b/bureau_b_h6_natural.parquet]

CUDA_VISIBLE_DEVICES=3 python src/tune.py \
    federated_method=fedrep \
    dataset@train_dataset=home_credit_bureau_b dataset@test_dataset=home_credit_bureau_b \
    distribution.client_column=credor_3940957M \
    losses@loss=ce trainer=tabular_trainer \
    tuning.metric=ROC-AUC federated_params.server_saving_metrics=[ROC-AUC] tuning.direction=maximize \
    tuning.n_trials=50 \
    federated_params.amount_of_clients=5 \
    training_params.batch_size=256 \
    training_params.num_workers=0 \
    manager.batch_size=5 \
    optimizer=adamw \
    train_dataset.num_policy=noisy_quantile \
    tuning.study_name=adamw_nq_bs256_natural_2 \
    train_dataset.data_sources.train_map_file=[data/home_credit_bureau_b/bureau_b_h6_natural.parquet]

CUDA_VISIBLE_DEVICES=3 python src/tune.py \
    federated_method=pfedme \
    dataset@train_dataset=home_credit_bureau_b dataset@test_dataset=home_credit_bureau_b \
    distribution.client_column=credor_3940957M \
    losses@loss=ce trainer=tabular_trainer \
    tuning.metric=ROC-AUC federated_params.server_saving_metrics=[ROC-AUC] tuning.direction=maximize \
    tuning.n_trials=50 \
    federated_params.amount_of_clients=5 \
    training_params.batch_size=256 \
    training_params.num_workers=0 \
    manager.batch_size=5 \
    optimizer=adamw \
    train_dataset.num_policy=noisy_quantile \
    tuning.study_name=adamw_nq_bs256_natural_2 \
    train_dataset.data_sources.train_map_file=[data/home_credit_bureau_b/bureau_b_h6_natural.parquet]
