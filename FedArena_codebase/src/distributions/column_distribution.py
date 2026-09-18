from utils.data_utils import print_df_distribution


class ColumnDistribution:
    """Splits data to clients based on an existing column in the DataFrame."""

    def __init__(self, client_column, top_n_clients=None, verbose=True):
        self.client_column = client_column
        self.top_n_clients = top_n_clients
        self.verbose = verbose

    def split_to_clients(self, df, amount_of_clients, random_state):
        col = self.client_column
        assert col in df.columns, (
            f"Column '{col}' not found in data. Available: {list(df.columns)}"
        )

        if self.top_n_clients is not None:
            top = df[col].value_counts().nlargest(self.top_n_clients).index
            df = df[df[col].isin(top)].reset_index(drop=True)

        unique_vals = sorted(df[col].dropna().unique())
        client_map = {v: i for i, v in enumerate(unique_vals)}
        df = df.assign(client=df[col].map(client_map))

        df = df[df["client"].notna()].reset_index(drop=True)
        df["client"] = df["client"].astype(int)

        if self.verbose:
            n_clients = df["client"].nunique()
            print(f"\nSplit by column '{col}': {n_clients} clients")
            for cid in sorted(df["client"].unique()):
                count = (df["client"] == cid).sum()
                print(f"  Client {cid}: {count} samples")

        return df
