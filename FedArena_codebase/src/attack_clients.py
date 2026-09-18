import numpy as np

from utils.data_utils import get_dataset_loader


class AttackClient:
    def apply_attack(self, client_instance):
        """Apply attack functionality to existing client instance

        Args:
            client_instance (Client): Client instance that is set for attack

        Returns:
            client_instance (Client): Client instance with added attack functionality
        """
        raise NotImplementedError


class LabelFlipClient(AttackClient):
    def __init__(self, percent_of_changed_labels):
        self.percent_of_changed_labels = percent_of_changed_labels

    def apply_attack(self, client_instance):
        """LabelFlip attack corrupts some percent of client labels. It apply attack functionality in 2 steps:
        1. Change `train_df` client attribute to the same dataframe with corrupted labels
        2. Reinit `train_loader` client attribute with corrupted `train_df` to set up training process with attack.

        Args:
            client_instance (Client): Client instance that is set for attack

        Returns:
            client_instance (Client): Client instance with added LabelFlip attack functionality
        """
        client_instance.train_df = self._change_client_labels(
            client_instance.train_df,
            client_instance.cfg.dataset.data_name,
            client_instance.rank,
        )
        client_instance.train_loader = get_dataset_loader(
            client_instance.train_df, client_instance.cfg
        )
        return client_instance

    def _change_client_labels(self, train_df, data_name, rank):
        # Seed randomization in accordance with client rank. See https://github.com/numpy/numpy/issues/9248
        rng = np.random.RandomState(rank)
        labels = np.array(train_df["target"].tolist())
        attacked_labels = rng.choice(
            np.prod(labels.shape),
            int(self.percent_of_changed_labels * np.prod(labels.shape)),
            replace=False,
        )
        corrupted_labels = rng.randint(0, 10, size=attacked_labels.size)
        labels.flat[attacked_labels] = corrupted_labels
        train_df.loc[train_df.index, "target"] = np.abs(labels)

        return train_df
