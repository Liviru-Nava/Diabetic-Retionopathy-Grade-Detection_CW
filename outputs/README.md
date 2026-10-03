# outputs/

Everything the notebook saved on Kaggle: charts in `figures/`, tables and JSON summaries in `results/`. Nothing here is edited by hand. Every file was written by a numbered notebook cell, so any number in the report can be traced back to the exact cell that produced it.

## How the folders are organised

`figures/` and `results/` use the **same numbered folder names**, and the numbers follow the order of the notebook. So `figures/07_experiment_preprocessing/` and `results/07_experiment_preprocessing/` both come from the same experiment, and a reader can go from a chart to its raw numbers by switching folders.

Some numbers only exist on one side, which is expected:

| Folder number | In figures/ | In results/ | Why |
|---|---|---|---|
| 04 splits and class weights | no | yes | This step only produces tables |
| 06 models and freezing | no | yes | Smoke tests and parameter counts are tables |
| 16 Grad-CAM | yes | no | Grad-CAM produces images only |
| 18 app export | no | yes | The export step produces files and checks, not charts |

`outputs/models/` is empty on purpose. The full-precision training checkpoints (about 16 MB, 43 MB and 94 MB) are not committed. The smaller half-precision copies the app uses live in the top-level [`models/`](../models/README.md) folder instead.

**A note on Kaggle leftovers.** The Kaggle output folder also contained older folders from earlier versions of the notebook (for example `02_preprocessing`, `05_models_and_training`, `06_evaluation`, `09_ben_graham_experiment` and `10_final_summary`). Those were deliberately **not** copied here, because they came from an earlier pipeline. Only the folders written by the final notebook are in this repo.

**File types.** `.png` files are charts. `.csv` files open in Excel or pandas. `.json` files hold small summaries and decisions. The single `.txt` file is scikit-learn's classification report.

---

## 01_data_exploration (notebook cells 5 to 9)

| File | What it is |
|---|---|
| `figures/.../class_distribution.png` | Bar chart of how many photos each grade has |
| `figures/.../sample_photos_by_grade.png` | Three raw photos from each grade |
| `figures/.../resolution_and_aspect_ratio.png` | How much photo width, height and shape vary |
| `results/.../data_smoke_tests.csv` | Three pass or fail checks on the raw data |
| `results/.../class_distribution.csv` | Photo count and percentage per grade |
| `results/.../resolution_summary.csv` | Summary statistics for width, height and aspect ratio |

**What it found.** All three smoke tests passed. The data is very unbalanced: No DR has 1,805 photos (49.3%), Moderate 999 (27.3%), Mild 370 (10.1%), Proliferative 295 (8.1%) and Severe only 193 (5.3%). The photos come in 17 different resolutions, from 474 x 358 to 4288 x 2848 pixels, so every photo has to be brought to one square size without stretching the eye.

## 02_preprocessing_methods (cells 10 to 17, plus cell 22)

| File | What it is |
|---|---|
| `figures/.../preprocessing_steps_and_methods_per_grade.png` | Every step (crop, pad, mask, contrast) for one photo per grade, with all three contrast methods side by side |
| `figures/.../eye_shape_kept_by_padding.png` | Shows why the crop is padded to a square before resizing, so the eye stays round |
| `results/.../preprocessing_tests_per_photo.csv` and `preprocessing_tests_summary.csv` | Six automatic checks run on 50 photos |
| `results/.../local_contrast_per_photo_per_method.csv` and `local_contrast_summary_per_method.csv` | How much local contrast each method gives, measured on 200 photos |
| `results/.../cache_build_times.csv` | How long it took to clean all 3,662 photos with each method |

**What it found.** Five of the six preprocessing tests passed at 100%. The one that did not is the shape test: 80% of photos kept their eye shape within 3%, against a target of 100%. Of the 50 test photos, 34 had the eye cut off at the top and bottom, and those still showed about 3.2% distortion on average. This is reported as a known limitation rather than hidden.

Local contrast inside the eye went from 5.46 (resize only) to 10.43 with CLAHE (1.9 times) and 17.73 with Ben Graham (3.2 times). Higher contrast makes small spots easier to see, but it does not prove better grading, which is why the method was chosen by experiment in folder 07.

## 03_image_size_study (cells 18 to 21)

| File | What it is |
|---|---|
| `figures/.../image_size_ssim_vs_cost.png` | Detail kept (SSIM) against training time for every size from 224 to 512 |
| `figures/.../image_size_patch_comparison.png` | The same small patch of one eye at every size, so the blur is visible |
| `results/.../ssim_per_photo_per_size.csv` | SSIM for each of the 100 photos at each size |
| `results/.../gpu_cost_per_size.csv` | Seconds per training step and peak GPU memory, per model and size |
| `results/.../image_size_evidence_table.csv` | Both sides of the trade-off in one table |
| `results/.../image_size_study_summary.json` | The key numbers in short form |

**What it found.** SSIM measures how much of the fine structure survives after shrinking a 1000 x 1000 photo and growing it back (1.0 means nothing was lost). At 224 it is 0.963, at 384 it is 0.977 and at 512 it is 0.984. Each step up adds less and less, while training time keeps climbing: 512 would take about 1.75 times longer than 384, and EfficientNet-B3 would no longer fit in the T4's memory with a batch of 16. That makes 384 a sensible middle point. No model had to be trained to reach this conclusion.

## 04_splits_and_class_weights (cells 23 and 24, results only)

| File | What it is |
|---|---|
| `split_membership.csv` | Every photo ID with its split (train, validation or test). This is the exact split used, so anyone can recreate it |
| `split_counts_per_grade.csv` | Number of photos per grade in each split |
| `split_proportions_per_grade.csv` | Share of each grade in each split, to prove the mix is the same |
| `class_weights.csv` | The loss weight each grade would get if class weighting is switched on |

**What it found.** Training has 2,563 photos, validation 549 and test 550. The share of each grade is almost identical in every split (for example, Severe is 5.3% in all three). The class weights range from 0.41 for No DR to 3.80 for Severe, meaning a Severe mistake would count about nine times as much as a No DR mistake. Whether to use them was tested in folder 09.

## 05_augmentation_policies (cells 25 to 27)

| File | What it is |
|---|---|
| `figures/.../augmentation_policies_preview.png` | Five random training versions of the same photo under each of the four policies |
| `results/.../augmentation_tests_per_policy.csv` | Checks that each policy keeps the size, keeps values valid, and actually changes photos when it should |
| `results/.../augmentation_extra_checks.csv` | A flip gives an exact mirror image, and the validation/test transform gives the same output every time |

**What it found.** All checks passed. The four policies range from no augmentation to flips, rotation, zoom, brightness, contrast and a little saturation. The winner was picked in folder 08.

## 06_models_and_freezing (cells 30 to 32, results only)

| File | What it is |
|---|---|
| `parameter_counts_before_changes.csv` | Size of each pretrained model with its original 1000-class ImageNet head |
| `model_smoke_tests.csv` | Four checks per model: output shape, pretrained weights loaded, freezing works, and the model can learn one batch |
| `trainable_parameters_per_freezing_option.csv` | How many weights learn under each freezing option |

**What it found.** All three models passed all four checks, and each one could cut the loss on a single repeated batch by close to 100%, which proves the training loop works. Training only the new last layer means learning just 0.04% to 0.16% of the weights, while unfreezing everything means learning all of them (4.0 M for B0, 10.7 M for B3, 23.5 M for ResNet50).

## 07 to 12: the six controlled experiments (cells 39 to 44)

Each of these folders has the same layout:

| File | What it is |
|---|---|
| `figures/.../experiment_comparison.png` | Validation QWK and the overfitting gap for every option, both seeds shown |
| `results/.../all_runs.csv` | One row per training run: best QWK, best epoch, overfitting gap, macro F1, recall per grade, minutes |
| `results/.../summary_per_option.csv` | The two seeds averaged for each option |
| `results/.../decision.json` | Which option won and the reason, written by the code using the rule fixed before the experiments |

Every experiment trains EfficientNet-B0 twice (seeds 42 and 7) on the validation set only. If a simpler option is within 0.005 QWK of the leader, the simpler option wins.

| Folder | Question | Mean validation QWK per option | Winner |
|---|---|---|---|
| 07_experiment_preprocessing | Which contrast step? | resize only 0.867, **CLAHE 0.874**, Ben Graham 0.872 | CLAHE |
| 08_experiment_augmentation | Is augmentation needed, and how much? | none 0.861, **flips and rotation 0.872**, full 0.874, full with zoom out 0.867 | Flips and rotation (tie rule) |
| 09_experiment_class_balance | Should rare grades count for more? | **plain loss 0.880**, class-weighted 0.872 | Plain loss |
| 10_experiment_transfer_learning | How much of the model should learn? | head only 0.798, partial unfreeze 0.875, **two-phase full unfreeze 0.880**, full fine-tune 0.883 | Two-phase full unfreeze (tie rule) |
| 11_experiment_learning_rate | Which fine-tuning learning rate? | 3e-5 0.871, **1e-4 0.880**, 3e-4 0.883 | 1e-4 (tie rule) |
| 12_experiment_regularisation | Does extra regularisation help? | none 0.880, **regularised 0.883** | Regularised |

**Things worth noticing in these folders:**

* **07:** CLAHE also had the smallest overfitting gap (0.017) and the best Severe recall (0.69).
* **08:** With no augmentation the overfitting gap was 0.073, the worst of any option in any experiment. Zooming out made things worse, probably because it shrinks the eye and hides small lesions.
* **09:** This is the most interesting trade-off. Class weights raised Severe recall from 0.38 to 0.60 but lowered QWK slightly, so by the rule the plain loss won. A screening service that cares most about catching Severe cases might choose differently.
* **10:** Training only the last layer was far behind (0.798). The pretrained ImageNet features must adapt to eye photos to grade them well, which is the core reason transfer learning here means fine-tuning, not just reusing features.
* **12:** This folder has one extra chart, `train_vs_validation_qwk_with_and_without_regularisation.png`, which shows the overfitting gap shrinking from 0.059 to 0.041 when label smoothing, weight decay, higher dropout and a cosine schedule are added together.

## 13_final_configuration (cell 45)

| File | What it is |
|---|---|
| `figures/.../validation_qwk_by_experiment_stage.png` | How the winning validation QWK moved from experiment 1 to experiment 6 |
| `results/.../experiment_decisions_table.csv` | One row per experiment: options, winner, score and reason. This is the justification trail for the report |
| `results/.../final_project_configuration.json` | Every fixed setting, the starting recipe, the final recipe and the full schedule in one file |

**What it found.** Validation QWK climbed from 0.874 after experiment 1 to 0.883 after experiment 6. The final recipe is CLAHE, flips and rotation, no class weights, two-phase full unfreeze, learning rates of 1e-3 then 1e-4, label smoothing 0.1, weight decay 0.01, dropout 0.4 and a cosine schedule.

## 14_final_training (cells 46 to 52)

| File | What it is |
|---|---|
| `figures/.../train_vs_validation_qwk_per_model.png` | Training QWK against validation QWK per epoch for each model, the main overfitting evidence |
| `figures/.../train_vs_validation_loss_and_accuracy_per_model.png` | Loss and accuracy curves for each model |
| `results/.../training_history_<model>.csv` | Every epoch's learning rate, loss, accuracy and QWK for training and validation |
| `results/.../overfitting_gaps_at_kept_epoch.csv` | The train minus validation gap at the epoch that was kept |
| `results/.../architecture_comparison.csv` | Size, epochs, kept epoch, best validation QWK and training time per model |
| `results/.../inference_speed.csv` | Prediction time on the GPU in batches and on the CPU for one photo |

**What it found.**

| Model | Kept epoch | Validation QWK | Train minus validation QWK | Training time |
|---|---|---|---|---|
| EfficientNet-B3 | 12 of 18 | 0.891 | 0.050 | about 19 min |
| EfficientNet-B0 | 16 of 22 | 0.887 | 0.069 | about 12 min |
| ResNet50 | 13 of 19 | 0.897 | 0.051 | about 23 min |

For B3 and ResNet50 the kept epoch is also the epoch with the lowest validation loss, which is a good sign that the checkpoint was taken before overfitting set in. For all three models, validation QWK rises quickly once phase 2 starts and the whole network is unfrozen, which matches what the freezing experiment predicted. ResNet50 was the best single model on validation. On a CPU, one photo takes about 0.06 s for B0, 0.13 s for B3, 0.17 s for ResNet50 and about 0.37 s for the full ensemble.

## 15_evaluation (cells 53 to 61)

This is where the test set was used, once, after every decision was fixed.

| File | What it is |
|---|---|
| `figures/.../headline_metrics_with_confidence_intervals.png` and `results/.../headline_metrics_with_confidence_intervals.csv` | QWK, accuracy and F1 for all four models with 95% bootstrap confidence intervals (1,000 resamples) |
| `figures/.../per_grade_scores.png` and `results/.../per_grade_precision_recall_f1_all_models.csv` | Precision, recall and F1 per grade per model |
| `results/.../classification_report_ensemble.txt` | scikit-learn's full report for the ensemble |
| `figures/.../confusion_matrices_counts.png` and `confusion_matrices_row_normalised.png` | Confusion matrices for all four models, as counts and as percentages of each true grade |
| `results/.../confusion_matrix_ensemble_counts.csv` | The ensemble's confusion matrix as numbers |
| `figures/.../roc_referable_dr_all_models.png` and `results/.../screening_metrics_referable_and_any_dr.csv` | Sensitivity, specificity, predictive values and ROC AUC for the two screening questions |
| `results/.../model_agreement.csv` | How often each pair of models, and all three, agree |
| `results/.../mcnemar_ensemble_vs_best_single.json` | McNemar's test, ensemble against ResNet50 |
| `figures/.../qwk_explained_and_train_val_test.png` and `results/.../qwk_detail_table.csv` | How QWK penalises mistakes, and each model's QWK on training, validation and test |
| `results/.../ensemble_vs_single_model_decision.json` | All the evidence for choosing the ensemble for the app |

**What it found.**

* **Headline:** the ensemble scored QWK 0.900 (95% CI 0.873 to 0.925), accuracy 84.2% and macro F1 0.684. Each single model scored between 0.872 and 0.877 QWK.
* **Is the ensemble really better?** The confidence intervals overlap, so they alone cannot say. McNemar's test can: the ensemble was right where ResNet50 was wrong on 28 photos, and the reverse happened on only 9 (p = 0.003). The gain is real.
* **Per grade:** No DR F1 0.98, Moderate 0.81, Mild 0.62, Proliferative 0.58, Severe 0.42. Severe recall is the weakest number at 0.31.
* **Confusion matrix:** 265 of 271 No DR photos were correct. Of the 29 Severe photos, 9 were correct, 12 were called Proliferative and 8 Moderate, and none was called No DR or Mild.
* **Screening view:** for "does this patient need a specialist?" the ensemble had 92.8% sensitivity and 94.2% specificity (AUC 0.985). For "is there any DR?" both were 97.8% (AUC 0.999).
* **Agreement:** all three models agreed on 84% of photos. When they agreed the ensemble was right 88.1% of the time, and when they disagreed only 63.6%, which is why disagreement is used as a review flag in the app.
* **Validation to test:** each model's QWK dropped by 0.009 to 0.023 from validation to test, a small and expected drop.

## 16_gradcam (cells 62 and 63, figures only)

| File | What it is |
|---|---|
| `gradcam_one_photo_per_grade.png` | ResNet50 Grad-CAM heat maps on the first test photo of each grade |

**What it shows.** Red areas pushed the model hardest towards its answer. Grad-CAM uses ResNet50 because it was the best single model on validation, and the ensemble has no single set of layers to look inside. The photos were chosen automatically, not hand-picked, and ResNet50 alone graded two of the five correctly. In the diseased eyes the heat tends to fall on bright yellow patches and dark red blotches inside the retina rather than the black background, which suggests the model is looking at lesion-like areas. This needs a clinician to confirm.

## 17_error_analysis (cells 64 to 67)

| File | What it is |
|---|---|
| `figures/.../error_distance_all_models.png` and `results/.../error_distance_all_models.csv` | How many predictions were exact, one grade off, or two or more off |
| `results/.../most_common_confusions_ensemble.csv` | Which grade pairs get mixed up most, and whether it was over- or under-grading |
| `figures/.../ensemble_confidence_right_vs_wrong.png` | The ensemble's confidence on right and wrong answers |
| `results/.../ensemble_accuracy_by_confidence.csv` | Accuracy inside each confidence band |
| `results/.../ensemble_confidence_summary.json` | How well the 60% confidence flag catches mistakes |
| `results/.../test_set_predictions_per_photo.csv` | Every test photo with its true grade, each model's prediction, the ensemble's probabilities, and whether it was flagged or correct |
| `figures/.../gradcam_biggest_mistakes.png` | Grad-CAM on the four photos ResNet50 got most wrong |

**What it found.**

* 84.2% of ensemble predictions were exact, 11.3% one grade off and only 4.5% two or more off. 71% of its mistakes are just one grade away.
* The top mix-ups are Mild called Moderate (16), Proliferative called Moderate (13), Moderate called Mild (12) and Severe called Proliferative (12). Mistakes go both ways, so the model is not simply biased up or down.
* Confidence is meaningful. At 90% confidence or above (44.5% of photos) the ensemble was right 99.2% of the time. Under 50% it was right 48% of the time.
* The app's 60% flag sends 19.3% of photos for review and catches 49.4% of all mistakes. Adding the disagreement flag raises that to 52.9%. Accuracy on unflagged photos is 90.1%.
* ResNet50 alone made 106 mistakes, 28 of them two or more grades off. In the Grad-CAM of the four worst, one photo is very pale and washed out, and in another the heat sits on the edge of the eye, which points to photo quality as one cause of the biggest errors. This is why the app also checks for dark, bright or oddly coloured photos.

## 18_app_export (cells 68 to 71, results only)

| File | What it is |
|---|---|
| `consolidated_results_summary.json` | Every key number from every section in one file, for writing the report |
| `exported_app_models.csv` | Each model's size before and after converting to half precision |
| `app_vs_notebook_parity_check.csv` | The app's prediction against the notebook's prediction on 50 test photos |
| `demo_images_for_video.csv` | The five test photos (one per grade) picked for the video demo, with the ensemble's prediction |

**What it found.** Converting to half precision halved the model files (for example ResNet50 went from 94.4 MB to 47.2 MB) so they fit comfortably in the repo for Streamlit Cloud. The app's copy gave the same grade as the notebook on all 50 photos, with the largest probability difference only 0.006, so the app reproduces the notebook faithfully. All five demo photos come from the test set, so the video shows the model on photos it never trained on, and the ensemble graded all five correctly.