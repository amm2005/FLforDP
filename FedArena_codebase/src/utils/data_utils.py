from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def get_dataset_loader(
    dataset,
    cfg,
    drop_last=True,
):
    num_workers = cfg.training_params.num_workers
    collator = getattr(dataset, "collator", None)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=(dataset.mode == "train"),
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
        timeout=(
            OmegaConf.select(cfg, "training_params.loader_timeout", default=600)
            if num_workers > 0
            else 0
        ),
    )
    assert (
        len(loader) > 0
    ), f"len(dataloader) is 0, either lower the batch size, or put drop_last=False"
    return loader


def print_df_distribution(df, num_classes, num_clients, pathology_names=None):
    is_multilabel = isinstance(df["target"].iloc[0], list)
    if is_multilabel:
        df["class_target"] = df["target"].apply(
            lambda x: [i for i, val in enumerate(x) if val == 1] or [-1]
        )

    print(f"Total usage data: {len(df[df['client'] != -1])}")

    client_groups = df.groupby("client")
    valid_clients = set(df["client"].unique())

    for cl in range(num_clients):
        if cl not in valid_clients:
            print(f"Client {cl:>2} | No data")
            continue

        client_data = client_groups.get_group(cl)
        if is_multilabel:
            distr = (
                client_data["class_target"]
                .explode()
                .value_counts()
                .reindex(range(-1, num_classes), fill_value=0)
                .tolist()
            )
        else:
            distr = (
                client_data["target"]
                .value_counts()
                .reindex(range(num_classes), fill_value=0)
                .tolist()
            )
        x_str = " ".join(f"{num:>5}" for num in distr)
        if cl == 0 and is_multilabel:
            pathology_str = " " * 11 + "  ".join(["Other"] + pathology_names)
            print(pathology_str)
        print(f"Client {cl:>3} | {x_str} | {len(client_data):>5}")
