## Code Appendix Summary

This project is run through `main.py`, which is the main wrapper code. It connects the smaller files in `src/` to load the dataset, train the models, evaluate results, and save the figures and tables used in the report.

The code is included to make the project reproducible, but the most important parts are the data loading, model training, cross-validation, evaluation metrics, and saved outputs.

### How to Run

From the project folder, run:

```bash
python main.py
```

This regenerates the main project outputs, including:

- dataset verification
- train/test split summary
- cross-validation results
- best hyperparameters
- model metrics
- classification reports
- confusion matrices
- figures

### Most Important Code Files

| File | Why it matters |
| --- | --- |
| `main.py` | Main wrapper script that runs the full experiment end to end. |
| `src/load_data.py` | Loads the UCI sensor data, assigns ultrasonic feature names `US_1` through `US_24`, and creates the target command label column. |
| `src/train_models.py` | Defines the scikit-learn pipelines, models, hyperparameter grids, stratified cross-validation, and `GridSearchCV`. |
| `src/evaluate_models.py` | Computes train/test accuracy, macro precision, macro recall, macro F1, classification reports, and confusion matrices. |
| `src/plot_results.py` | Generates the plots used in the report and poster, including class distribution, PCA, model comparison, and confusion matrices. |
| `src/config.py` | Stores project paths, random seed, dataset URLs, feature names, and label order. |

### Key Machine Learning Implementation Details

- This is an **offline supervised classification** project.
- The dataset is the **UCI Wall-Following Robot Navigation Data Set**.
- The main input file is `sensor_readings_24.data`.
- Each sample has 24 ultrasonic sensor readings.
- The target is the dataset-provided command label.
- The four command labels are:
  - `Move-Forward`
  - `Slight-Right-Turn`
  - `Sharp-Right-Turn`
  - `Slight-Left-Turn`

### Models Compared

The class-report models are:

- Dummy majority-class baseline
- Logistic Regression
- Linear SVM
- RBF Kernel SVM

These models compare a simple baseline, linear classifiers, and a nonlinear kernel method.

### Cross-Validation and Model Selection

Hyperparameters are selected using stratified 5-fold `GridSearchCV`.

| Model | Parameters tuned |
| --- | --- |
| Logistic Regression | `C` |
| Linear SVM | `C` |
| RBF SVM | `C`, `gamma` |

Macro F1 is used as the main cross-validation score because the command labels are imbalanced.

Preprocessing is done inside scikit-learn `Pipeline` objects, so scaling is fit only on the training folds during cross-validation. This helps avoid data leakage.

### Multiclass Classification

The dataset has four command labels, not a binary `+1/-1` target.

The implementation uses scikit-learn’s multiclass support:

- `LogisticRegression` handles the four-class problem directly with the `lbfgs` solver.
- `SVC` handles multiclass SVM classification internally using one-vs-one decision functions.
- Precision, recall, and F1 are reported with macro averaging across the four command classes.

### Verification Target

The model is evaluated by comparing:

```text
predicted_command == true_command
```

In other words, the prediction is checked against the dataset-provided command label on the held-out test set.

This project does **not** estimate robot position, measure physical standoff error, deploy a robot controller, or use reinforcement learning.

### Main Output Files

| Output | Purpose |
| --- | --- |
| `results/dataset_verification.csv` | Confirms dataset source, raw file, sample count, feature count, class counts, missing values, train/test sizes, and verification target. |
| `results/model_metrics.csv` | Stores train/test accuracy, macro precision, macro recall, and macro F1 for each model. |
| `results/class_model_metrics.csv` | Stores the class-report model comparison for Dummy, Logistic Regression, Linear SVM, and RBF SVM. |
| `results/cv_results.csv` | Stores cross-validation scores for each tested hyperparameter setting. |
| `results/best_params.json` | Stores the best hyperparameters selected by `GridSearchCV`. |
| `results/classification_reports/` | Contains per-class precision, recall, and F1 reports. |
| `results/confusion_matrices/` | Contains held-out test confusion matrices. |
| `figures/` | Contains plots used in the report and poster. |

### Key Result Connected to the Code

The best class-report model was the RBF SVM.

- Test accuracy: `0.929`
- Test macro F1: `0.925`
- Best parameters: `C=100`, `gamma=scale`

The largest confusion was:

```text
Move-Forward predicted as Sharp-Right-Turn
```

for 24 held-out test samples.

### Code Appendix

The full code appendix is available here:

```text
report/code_appendix.pdf
```

The combined submission PDF with the report and code appendix is:

```text
report/final_submission_with_code_appendix.pdf
```
