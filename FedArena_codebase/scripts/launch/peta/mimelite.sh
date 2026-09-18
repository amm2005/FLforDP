# Reference launch: MimeLite (Mime without SVRG) on PETA.
# Multi-label pedestrian-attribute classification; federated split by `source_dataset`.
# Image support: base Client._model_forward dispatches len(inputs)==1 -> model(image).
# MimeLite hyperparameters (round_epochs; lr/wd via common) are tuned by Optuna
# (federated_methods/mimelite/search_space.py). betas/eps are FIXED to the paper's
# multi-epoch values (beta2=0.99, eps=1e-3); adamw.yaml defaults are unusable for
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=2 python src/tune.py \
    federated_method=mimelite \
    dataset@train_dataset=peta \
    dataset@test_dataset=peta \
    distribution=column \
    distribution.client_column=source_dataset \
    models@models_dict.model1=resnet18_groupnorm \
    losses@loss=bce \
    trainer=cifar_trainer \
    training_params.batch_size=256 manager.batch_size=5 \
    training_params.num_workers=5 \
    federated_params.amount_of_clients=10 \
    federated_params.client_train_frac=0.7 federated_params.client_val_frac=0.15 federated_params.client_test_frac=0.15 \
    federated_params.server_saving_metrics=[macro_AP] \
    federated_params.communication_rounds=50 \
    tuning.metric=macro_AP \
    tuning.direction=maximize \
    tuning.n_trials=50 \
    tuning.study_name=adamw_beta2_0_99_eps_1e-3 \
    'optimizer.betas=[0.9, 0.99]' \
    optimizer.eps=1e-3 \
    optimizer=adamw
