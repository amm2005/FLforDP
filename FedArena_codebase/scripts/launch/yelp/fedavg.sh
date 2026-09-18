source .venv/bin/activate

#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python src/tune.py \
    federated_method=fedavg \
    models@models_dict.model1=bert_tiny \
    dataset@train_dataset=yelp dataset@test_dataset=yelp \
    distribution.client_column=business_id distribution.top_n_clients=10 \
    losses@loss=ce trainer=text_classification_trainer \
    tuning.metric=macro_AP federated_params.server_saving_metrics=[macro_AP] tuning.direction=maximize \
    federated_params.amount_of_clients=10 \
    federated_params.client_train_frac=0.7 federated_params.client_val_frac=0.15 federated_params.client_test_frac=0.15 \
    training_params.batch_size=128 manager.batch_size=1 \
    optimizer=adamw \
    tuning.study_name=adamw_new_dataset
