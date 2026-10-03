# models/

The three trained CNNs that the DRxVision app loads to grade a photo. Together they make up the soft-vote ensemble.

| File | Architecture | Size | Validation QWK | Test QWK on its own |
|---|---|---|---|---|
| `efficientnet_b3_fp16.pth` | EfficientNet-B3 | about 22 MB | 0.891 | 0.872 |
| `efficientnet_b0_fp16.pth` | EfficientNet-B0 | about 8 MB | 0.887 | 0.877 |
| `resnet50_fp16.pth` | ResNet50 | about 47 MB | 0.897 | 0.874 |

Averaged together, the three reach a test QWK of 0.900, which McNemar's test showed is a real improvement over the best single model (p = 0.003). ResNet50 is also the model the app uses to draw the Grad-CAM heat map, because it scored best on validation.

## Where these files come from

1. Each model was trained in notebook cells 46 to 48 with the final recipe. The best epoch by **validation QWK** was saved as a full-precision checkpoint (`best_<model>_by_validation_qwk.pth`) in Kaggle's `/kaggle/working/models/`.
2. Notebook cell 69 converted every floating point weight to **half precision (fp16)**, which halves the file size with no meaningful loss of accuracy, and saved them to `app_export/models/`.
3. These three files were downloaded from that `app_export/models/` folder on Kaggle and copied here unchanged.

| Model | Full precision | Half precision (this folder) |
|---|---|---|
| EfficientNet-B3 | 43.4 MB | 21.8 MB |
| EfficientNet-B0 | 16.4 MB | 8.2 MB |
| ResNet50 | 94.4 MB | 47.2 MB |

## Proof the smaller files give the same answers

Notebook cells 70 and 71 rebuilt the app's prediction steps using only these exported files, on a CPU, exactly as the app runs them. On 50 test photos they gave **the same grade as the notebook every time**, and the largest difference in any grade probability was 0.006. Tiny differences like this come from half-precision rounding and CPU versus GPU maths. The full check is in [`outputs/results/18_app_export/app_vs_notebook_parity_check.csv`](../outputs/results/18_app_export/app_vs_notebook_parity_check.csv).

## Why they are committed to git

These files are committed on purpose, so that Streamlit Community Cloud can load them straight from the repository. Each is well under GitHub's 100 MB file limit, so Git LFS is not needed.

The full-precision checkpoints are **not** committed (they would add about 154 MB to the repo history and are only needed for further training). `.gitignore` excludes `outputs/models/*.pth` for that reason. To get them, rerun the notebook on Kaggle and download them from the `models/` output folder.

## How the app uses them

`dr_inference.py` reads `app_config.json` from the repo root, which lists the three file names under `architectures`. For each one it builds the matching torchvision architecture with a 5-grade output layer, loads the weights from this folder, converts them back to full precision for the CPU, and switches the model to evaluation mode. An uploaded photo goes through all three, and their grade probabilities are averaged.

## Replacing the models after retraining

1. Download `app_export/` from the Kaggle notebook output.
2. Copy `app_export/models/*.pth` into this folder, overwriting the old files.
3. Copy `app_export/app_config.json` to the repo root, overwriting the old one. This matters, because the config holds the preprocessing method and settings the models were trained with, and the app must clean photos in exactly the same way.
4. Run `streamlit run app.py` and analyse one photo to confirm it loads.

Each file only stores the weights (a PyTorch `state_dict`), not the model code, so it can only be loaded into the matching torchvision architecture.

Copy these three files here from the Kaggle output folder `app_export/models/`:

- efficientnet_b3_fp16.pth (about 22 MB)
- efficientnet_b0_fp16.pth (about 8 MB)
- resnet50_fp16.pth (about 47 MB)

These are committed to the repo so Streamlit Community Cloud can load them.
