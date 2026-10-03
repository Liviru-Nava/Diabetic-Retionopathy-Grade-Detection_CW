# data/

This folder is empty on purpose. The `raw/` and `processed/` subfolders only hold a `.gitkeep` file so git keeps the folder structure.

## Why there are no photos here

* **They belong to the competition.** The APTOS 2019 photos are shared under Kaggle's competition rules, so they are not redistributed in this repository.
* **They are not needed locally.** All training ran on Kaggle, where the dataset is attached to the notebook as an input. The web app does not need the dataset either, only the trained models in `models/`.
* **They are large.** The cleaned photo caches the notebook builds are several gigabytes and can be rebuilt in about 15 minutes, so they are never committed.

`.gitignore` blocks everything inside `data/raw/` and `data/processed/`, plus any `.npy` cache files and `kaggle.json` (the Kaggle API key), so none of these can be committed by accident.

## The dataset

**APTOS 2019 Blindness Detection**: https://www.kaggle.com/c/aptos2019-blindness-detection

3,662 labelled colour fundus photos collected by Aravind Eye Hospital in India, each graded on the 0 to 4 scale.

| Grade | Stage | Photos | Share |
|---|---|---|---|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative DR | 295 | 8.1% |

The photos come in 17 different resolutions, from 474 x 358 up to 4288 x 2848 pixels. Only the competition's labelled `train_images` are used. They are split 70/15/15 into training, validation and test sets inside the notebook, and the exact split is saved in `outputs/results/04_splits_and_class_weights/split_membership.csv`.

## Getting the data

**On Kaggle (how this project was run):** in the notebook, click **Add Input**, search for `aptos2019-blindness-detection` and attach it. It appears read-only at `/kaggle/input/competitions/aptos2019-blindness-detection`, which is the path the notebook already uses.

**On your own computer (only if you want a few real photos to try in the app):**

```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c aptos2019-blindness-detection -p data/raw/
```

You need to accept the competition rules on the Kaggle website before the download works. Never commit `kaggle.json`.

## Dataset limitations worth knowing

* All photos come from one hospital network in one country, so models trained on them may perform worse on other cameras and populations.
* The grades are very unbalanced, with Severe the rarest at 193 photos.
* Image quality varies: some photos are blurred, dark, overexposed or have the eye cut off at the top and bottom.
* Each photo has a single grade, and neighbouring grades (especially Mild and Moderate) are hard to separate even for experts, so some labels are likely to be noisy.