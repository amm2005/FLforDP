# FedArena paper dataset pipelines

Run every repository command below from the
`Federated-Research-LIB-dev` root. Dataset paths in Hydra configs and launch
scripts are deliberately relative to that directory. The data are not
redistributable as part of this repository; accept each upstream license before
downloading.

## Fannie Mae

Download the original 2012 Q1, Q2, and Q3 single-family loan-performance files
from Fannie Mae Data Dynamics and place the headerless, pipe-delimited files at:

```text
data/fannie_mae/raw/2012Q1.csv
data/fannie_mae/raw/2012Q2.csv
data/fannie_mae/raw/2012Q3.csv
```

Build and launch the paper variant:

```bash
python scripts/prepare_fannie_mae.py
bash scripts/launch/fannie_mae/fedavg.sh
```

The prepare command writes
`data/fannie_mae/fannie_mae_d90_h84_oot100_ex-wells.parquet`. It contains nine
seller clients and a baked out-of-time split: 2012Q1 train, 2012Q2 validation,
and 2012Q3 test.

## Home Credit Bureau-B

Attach the Home Credit Credit Risk Model Stability competition data to a Kaggle
notebook, upload `scripts/kaggle/export_home_credit_bureau_b.py`, and run:

```bash
python export_home_credit_bureau_b.py
```

Download `home_credit_bureau_b.zip` from the notebook and extract its seven
parquet files into `data/home_credit_bureau_b/raw_filtered/`. Then run:

```bash
python scripts/prepare_home_credit_bureau_b.py
bash scripts/launch/home_credit_bureau_b/fedavg.sh
```

The local prepare command writes
`data/home_credit_bureau_b/bureau_b_h6_natural.parquet`: the natural H=6,
top-five-creditor population with a baked per-creditor temporal 70/15/15 split.
The Kaggle exporter refuses to replace an existing export unless
`--overwrite` is passed.

## Expedia Hotel Recommendations

Accept the Kaggle competition rules and extract the complete `train.csv` and
`destinations.csv` into `data/expedia_hotel_recommendations/`. The competition
`test.csv` is not used.

```bash
python scripts/prepare_expedia.py
bash scripts/launch/expedia_hotel_recommendations/fedavg.sh
```

The prepare command keeps the earliest 3,000,000 events, selects sites with
6,000 to 100,000 rows, and writes
`data/expedia_hotel_recommendations/expedia_6000_100000.parquet`. The artifact
has 12 `site_name` clients and is split online per client in timestamp order
70/15/15.

## COVID-19

After configuring Kaggle API credentials:

```bash
mkdir -p data/covid19/raw
kaggle datasets download \
    -d josephassaker/covid19-global-dataset \
    -p data/covid19/raw \
    --unzip
python scripts/prepare_covid.py
bash scripts/launch/covid19/fedavg.sh
```

The prepare command selects the ten most populous countries and writes
`data/covid19/train_df.csv` and `data/covid19/test_df.csv`. The loader combines
the two files, constructs leakage-safe lag/rolling features, and applies a
per-country temporal 60/20/20 experiment split.

## Flight Price Prediction

After configuring Kaggle API credentials:

```bash
mkdir -p data/flight_price_prediction
kaggle datasets download \
    -d shubhambathwal/flight-price-prediction \
    -p data/flight_price_prediction \
    --unzip
bash scripts/launch/flight_price_prediction/fedavg.sh
```

There is no repository prepare step. The loader reads
`data/flight_price_prediction/Clean_Dataset.csv` directly, creates six airline
clients, and applies a deterministic per-client random 70/15/15 split.

## Caltech Camera Traps

The repository prepare script downloads the complete annotation archive and
only the images belonging to the paper's top five non-empty locations. Images
are resized directly into the experiment directory; a full-resolution image
tree is not retained.

```bash
python scripts/prepare_cct.py
bash scripts/launch/cct/fedavg.sh
```

Outputs are `data/cct/caltech_images_20210113.json` and
`data/cct/cct_images_256/`. Use `python scripts/prepare_cct.py --dry-run` to
validate the annotation subset without downloading images (the annotation
archive itself is still downloaded if absent).

## FLAIR

No repository-specific prepare step is required. Clone Apple's official
downloader beside this repository and direct it into the repository data tree:

```bash
git clone https://github.com/apple/ml-flair.git ../ml-flair
python ../ml-flair/download_dataset.py --dataset_dir data/flair
bash scripts/launch/flair/fedavg.sh
```

The experiment reads `data/flair/labels_and_metadata.json` and
`data/flair/small_images/`. Do not pass `--download_raw`; the full-resolution
1.2-TB variant is not used.

## Fed-ISIC2019

Clone and install FLamby beside this repository, then run its official download
and resize scripts from the repository root. The second command reads the
`dataset_location.yaml` created by the first.

```bash
git clone https://github.com/owkin/FLamby.git ../FLamby
python -m pip install -e ../FLamby
python ../FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py \
    --output-folder data/fed_isic2019
python ../FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/resize_images.py
python scripts/prepare_fed_isic2019.py \
    --split-file ../FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/train_test_split
bash scripts/launch/fed_isic2019/fedavg.sh
```

The official scripts write
`data/fed_isic2019/ISIC_2019_Training_Input_preprocessed/`. The repository
prepare command validates FLamby's 23,247-row official split and writes
`data/fed_isic2019/fed_isic2019_map.csv`. The six centers are clients; the
official 4,650-image test fold remains the server test set.
