# Code Appendix README

Project: **Classifying Pre-Labeled Robot Wall-Following Commands from Ultrasonic Sensor Data**

This appendix documents the code used to run the machine learning analysis. The project uses Python and scikit-learn to load the public UCI Wall-Following Robot Navigation Data Set, train classification models, select hyperparameters with cross-validation, evaluate held-out test performance, and generate the plots/tables used in the report.

## How to Reproduce the Analysis

Run the full project from the repository root:

```bash
python main.py
```

For a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Wrapper Code

`main.py` is the wrapper script. It calls the supporting modules in `src/` in this order:

1. Create required folders.
2. Download or locate the UCI raw data files.
3. Load the 24-sensor dataset and assign feature names `US_1` through `US_24`.
4. Print dataset shape, feature count, class labels, class counts, missing values, and split sizes.
5. Create a fixed stratified 80/20 train/test split.
6. Generate EDA outputs: class distribution, PCA plot, sensor statistics, and correlation heatmap.
7. Train models with scikit-learn `Pipeline` and `GridSearchCV`.
8. Evaluate each model on training data and the held-out test set.
9. Save metrics, best parameters, classification reports, confusion matrices, figures, and trained models.
10. Run the optional reduced-sensor ablation study for the portfolio extension.

The wrapper does **not** generate the final report text. The report is written from the saved results produced by `python main.py`.

## Source File Map

| File | Purpose |
| --- | --- |
| `main.py` | Wrapper script that runs the full experiment end to end. |
| `src/config.py` | Stores paths, random seed, dataset URLs, feature names, labels, and model settings. |
| `src/download_data.py` | Downloads or locates the UCI raw dataset files. |
| `src/load_data.py` | Parses raw `.data` files, assigns feature/target names, checks labels, and saves processed CSVs. |
| `src/train_models.py` | Defines scikit-learn pipelines, model grids, `StratifiedKFold`, and `GridSearchCV`. |
| `src/evaluate_models.py` | Computes accuracy, macro precision, macro recall, macro F1, classification reports, and confusion matrices. |
| `src/plot_results.py` | Generates class distribution, PCA, model comparison, CV, confusion matrix, ablation, and coefficient figures. |
| `src/utils.py` | Provides helper functions for directories, JSON/text output, filename slugs, and metric formatting. |

## Models and Model Selection

The class-report comparison uses:

- Dummy majority-class baseline
- Logistic Regression
- Linear SVM
- RBF Kernel SVM

`GridSearchCV` with stratified 5-fold cross-validation selects model parameters using macro F1:

| Model | Parameters selected with CV |
| --- | --- |
| Logistic Regression | `C` |
| Linear SVM | `C` |
| RBF SVM | `C`, `gamma` |

Preprocessing is inside each scikit-learn `Pipeline`, so scaling is fit only on training folds during cross-validation.

## Verification Target

This is offline supervised classification. The code verifies a prediction by comparing:

```text
predicted_command == true_command
```

The project does not estimate robot position, measure physical standoff error, deploy a robot controller, or use reinforcement learning.

## Main Generated Outputs

| Output | Purpose |
| --- | --- |
| `results/dataset_verification.csv` | Dataset source, file name, feature count, class counts, missing values, split sizes, and verification target. |
| `results/model_metrics.csv` | Train/test accuracy, macro precision, macro recall, and macro F1 for all full models. |
| `results/class_model_metrics.csv` | Class-report model comparison: Dummy, Logistic Regression, Linear SVM, and RBF SVM. |
| `results/cv_results.csv` | Cross-validation scores for each tested hyperparameter setting. |
| `results/best_params.json` | Best hyperparameters selected by `GridSearchCV`. |
| `results/classification_reports/*.txt` | Per-class precision, recall, and F1 reports. |
| `results/confusion_matrices/*.csv` | Held-out test confusion matrices. |
| `figures/*.png` | Plots used in the report and poster. |

## Code Included

The full code listing is included after this README-style summary in:

- `report/code_appendix.pdf`
- `report/final_submission_with_code_appendix.pdf`

Those PDFs include the wrapper code and the supporting `src/` files used to generate the project outputs.
