# notebooks/

`ComputerVisionCW_LM_Navaratna_002_16114793_1.ipynb` is the whole project in one notebook: data checks, preprocessing, the image size study, augmentation, six controlled experiments, final training of three CNNs, testing, Grad-CAM, error analysis and the export for the web app. It was run end to end on Kaggle with a Tesla T4 GPU.

The outputs inside the notebook are kept, so every printed table, training log and chart can be read on GitHub without running anything.

## How the notebook is laid out

There are 72 code cells grouped into 15 sections. Every code cell starts with a comment such as `# Cell 23: ...` that says what it does, and that number is how cells are referred to in the READMEs and the report. Every cell ends by printing a plain sentence such as `Cell 23 finished: data split into training, validation and test sets.` so a run can be followed and checked step by step.

| Section | Cells | What happens | Saves to `outputs/` folder |
|---|---|---|---|
| 1. Setup | 1 to 4 | Load libraries, check the GPU, fix the random seed, store fixed settings and create output folders | |
| 2. Looking at the data | 5 to 9 | Load labels, data smoke tests, class balance, sample photos, photo sizes | `01_data_exploration` |
| 3. Preprocessing methods | 10 to 17 | Crop, pad to square, circular mask, CLAHE, Ben Graham, one pipeline with a method switch, preprocessing tests, step-by-step figure | `02_preprocessing_methods` |
| 4. Image size check | 18 to 22 | SSIM detail study, GPU time and memory per size, the evidence table, patch comparison, then build the photo caches | `03_image_size_study` (cache times go to `02`) |
| 5. Splits and class weights | 23 and 24 | 70/15/15 stratified split, class weights from training only | `04_splits_and_class_weights` |
| 6. Augmentation policies | 25 to 27 | Four policies, augmentation tests, preview figure | `05_augmentation_policies` |
| 7. Feeding photos to the model | 28 and 29 | Dataset class and data loaders, with one real batch pulled through as a check | |
| 8. Models and training engine | 30 to 37 | Model sizes, model factory, freezing helpers, model smoke tests, loss, training and evaluation loops, early stopping, the recipe-driven training function | `06_models_and_freezing` |
| 9. Experiments | 38 to 44 | The experiment runner and the six experiments | `07` to `12` |
| 10. Final configuration | 45 | Gather every decision into the final recipe | `13_final_configuration` |
| 11. Final training | 46 to 52 | Train B3, B0 and ResNet50, curves, overfitting table, speed test | `14_final_training` |
| 12. Testing the models | 53 to 61 | One-time test, ensemble, confidence intervals, per-grade scores, confusion matrices, screening view, agreement, McNemar's test, QWK detail, ensemble decision | `15_evaluation` |
| 13. Grad-CAM | 62 and 63 | Grad-CAM on ResNet50, one photo per grade | `16_gradcam` |
| 14. Error analysis | 64 to 67 | Mistake distance, common confusions, confidence analysis, per-photo predictions, Grad-CAM on the worst mistakes | `17_error_analysis` |
| 15. Summary and app export | 68 to 72 | Consolidated summary, export models and settings for the app, app parity check, demo photos, list of every saved file | `18_app_export` |

See [`outputs/README.md`](../outputs/README.md) for what every saved file shows and what it found.

## Ideas that run through the whole notebook

* **Tested, not assumed.** Only a few settings are fixed up front, and each has a written reason in the table under Cell 4. Everything else is decided by an experiment on the validation set.
* **The test set stays locked.** It is not touched until Cell 53, after every choice has been made, and it is used once.
* **One change at a time.** Each experiment changes one part of the recipe and reruns with two seeds, so a difference smaller than the seed-to-seed wobble is not mistaken for a real gain.
* **Tie rule fixed in advance.** If a simpler or safer option is within 0.005 QWK of the leader, the simpler option wins.
* **Smoke tests before building.** Data, preprocessing, augmentation and models each get pass or fail tests before the next stage uses them.
* **No repeated work.** The experiment runner remembers runs it has already done. When an option in a later experiment is identical to a run from an earlier one, it reuses the result instead of training again, so only 26 runs were needed.

## Coding style

* One main job per cell, numbered in its first comment line.
* Long, descriptive names such as `diabetic_retinopathy_labels_dataframe` and `run_mcnemar_test_on_paired_predictions`, so the code reads close to plain English.
* Comments explain **why** a choice was made, not only what the line does.
* Every figure, table and JSON file is saved through small helper functions (`save_current_figure`, `save_results_table` and `save_results_json`), so every file lands in its numbered section folder and the cell prints where it went.

## How to run it

1. Upload or open the notebook on **Kaggle Notebooks**.
2. **Add Input**: search for `aptos2019-blindness-detection` and attach the competition data.
3. Session options: **GPU T4** on and **Internet** on (needed to download the pretrained ImageNet weights).
4. **Save Version** with **Save & Run All (Commit)**, so it runs once, top to bottom, in a clean session.
5. It takes roughly **4 to 5 hours**: about 3.5 hours for the 26 experiment runs, about 55 minutes for the three final models, and about 15 minutes to build the preprocessing caches.

Running it on a laptop is not practical, because it trains 29 CNNs in total. To look at the results, reading the notebook on GitHub is enough. To try the models, use the web app in the repo root, which only needs a CPU.

## Reproducibility

The seed is fixed to 42, with 7 used for the second run of each experiment, and the GPU is told to use repeatable maths where it can. The outputs printed in this notebook and the files in `outputs/` came from two separate runs of the same code. Every score, split and decision matches exactly. Only the timings differ a little, which is normal on shared cloud GPUs.