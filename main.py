from __future__ import annotations

import json
import os
import shutil
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSIFICATION_REPORT_DIR,
    COMMAND_ORDER,
    CONFUSION_MATRIX_DIR,
    DATASET_FILES,
    DATASET_URLS,
    FIGURES_DIR,
    PRIMARY_SCORING,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    TRAINED_MODEL_DIR,
)
from src.download_data import download_all_datasets
from src.evaluate_models import (
    evaluate_estimator,
    extract_linear_feature_importance,
    most_confused_pair,
)
from src.load_data import RobotDataset, load_all_datasets, summarize_missing_values
from src.plot_results import (
    save_class_distribution,
    save_coefficient_plot,
    save_confusion_matrix,
    save_cv_score_comparison,
    save_metric_comparison,
    save_pca_sensor_space,
    save_sensor_ablation_comparison,
    save_sensor_correlation_heatmap,
)
from src.train_models import (
    cv_results_to_frame,
    fit_grid_search,
    get_ablation_model_specs,
    get_main_model_specs,
)
from src.utils import (
    ensure_project_directories,
    format_metric,
    slugify,
    write_json,
    write_text,
)

CLASS_REPORT_MODELS = [
    "Dummy Majority",
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
]
CLASS_SELECTION_MODELS = ["Logistic Regression", "Linear SVM", "RBF SVM"]


def print_dataset_summary(dataset: RobotDataset) -> None:
    """Print the raw data facts that matter for grading and reproducibility."""
    missing_counts = summarize_missing_values(dataset)
    print("\nDataset summary")
    print("===============")
    print(f"Dataset: {dataset.display_name}")
    print(f"Raw file used: {DATASET_FILES[dataset.key]}")
    print(f"Shape: {dataset.data.shape[0]} samples x {dataset.data.shape[1]} columns")
    print(f"Feature count: {len(dataset.feature_names)}")
    print("Class counts:")
    print(dataset.y.value_counts().reindex(COMMAND_ORDER).to_string())
    print(f"Label names: {', '.join(COMMAND_ORDER)}")
    print(f"Missing values: {int(missing_counts.sum())}")


def make_train_test_indices(y: pd.Series) -> tuple[pd.Index, pd.Index]:
    """Create one stratified train/test split reused across all experiments."""
    indices = pd.Index(range(len(y)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return pd.Index(train_idx), pd.Index(test_idx)


def print_split_summary(dataset: RobotDataset, train_idx: pd.Index, test_idx: pd.Index) -> None:
    """Show that the held-out split preserved the class imbalance."""
    print("\nTrain/test split")
    print("================")
    print(f"Train set size: {len(train_idx)}")
    print(f"Test set size: {len(test_idx)}")
    print("Train class counts:")
    print(dataset.y.iloc[train_idx].value_counts().reindex(COMMAND_ORDER).to_string())
    print("Test class counts:")
    print(dataset.y.iloc[test_idx].value_counts().reindex(COMMAND_ORDER).to_string())


def write_dataset_verification(
    dataset: RobotDataset, train_idx: pd.Index, test_idx: pd.Index
) -> pd.DataFrame:
    """Save a compact audit table describing the dataset and verification target."""
    full_counts = dataset.y.value_counts().reindex(COMMAND_ORDER).fillna(0).astype(int)
    train_counts = dataset.y.iloc[train_idx].value_counts().reindex(COMMAND_ORDER).fillna(0).astype(int)
    test_counts = dataset.y.iloc[test_idx].value_counts().reindex(COMMAND_ORDER).fillna(0).astype(int)
    full_props = full_counts / len(dataset.y)
    train_props = train_counts / len(train_idx)
    test_props = test_counts / len(test_idx)
    max_train_diff = float((train_props - full_props).abs().max())
    max_test_diff = float((test_props - full_props).abs().max())
    stratification_check = (
        "pass: stratified split preserved class proportions within "
        f"{max(max_train_diff, max_test_diff):.4f} absolute proportion"
    )

    rows = [
        ("dataset_source", "UCI Wall-Following Robot Navigation Data Set"),
        ("dataset_url", DATASET_URLS[dataset.key]),
        ("raw_file_name", DATASET_FILES[dataset.key]),
        ("number_of_samples", len(dataset.data)),
        ("number_of_features", len(dataset.feature_names)),
        ("feature_names", ", ".join(dataset.feature_names)),
        ("target_column_name", dataset.target_name),
        ("unique_command_labels", ", ".join(COMMAND_ORDER)),
        ("class_counts", json.dumps(full_counts.to_dict(), sort_keys=True)),
        ("missing_value_count", int(dataset.data.isna().sum().sum())),
        ("train_set_size", len(train_idx)),
        ("test_set_size", len(test_idx)),
        ("train_class_counts", json.dumps(train_counts.to_dict(), sort_keys=True)),
        ("test_class_counts", json.dumps(test_counts.to_dict(), sort_keys=True)),
        ("stratification_check", stratification_check),
        ("random_seed", RANDOM_STATE),
        (
            "preprocessing_statement",
            "StandardScaler preprocessing is fit inside scikit-learn Pipeline during GridSearchCV.",
        ),
        (
            "verification_target",
            "predicted command label compared with dataset-provided true command label",
        ),
    ]
    verification = pd.DataFrame(rows, columns=["field", "value"])
    verification.to_csv(RESULTS_DIR / "dataset_verification.csv", index=False)
    return verification


def run_main_experiment(
    dataset: RobotDataset, train_idx: pd.Index, test_idx: pd.Index
) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame], pd.DataFrame]:
    """Train and evaluate the full 24-sensor model comparison."""
    X_train = dataset.X.iloc[train_idx]
    X_test = dataset.X.iloc[test_idx]
    y_train = dataset.y.iloc[train_idx]
    y_test = dataset.y.iloc[test_idx]

    metric_rows = []
    cv_frames = []
    best_params = {}
    confusion_matrices = {}
    feature_importance_frames = []

    print("\nTraining full 24-sensor models")
    print("==============================")
    for spec in get_main_model_specs():
        print(f"Fitting {spec.name} with GridSearchCV ({PRIMARY_SCORING})")
        search = fit_grid_search(spec, X_train, y_train)
        estimator = search.best_estimator_
        model_slug = slugify(spec.name)

        metrics, report_text, cm_frame = evaluate_estimator(
            spec.name, estimator, X_train, y_train, X_test, y_test
        )
        metrics["dataset"] = dataset.key
        metrics["feature_set"] = dataset.display_name
        metrics["n_features"] = len(dataset.feature_names)
        metrics["cv_best_macro_f1"] = float(search.best_score_)
        metrics["best_params"] = json.dumps(search.best_params_, sort_keys=True)
        metric_rows.append(metrics)

        best_params[spec.name] = {
            "best_params": search.best_params_,
            "best_cv_macro_f1": float(search.best_score_),
        }
        cv_frames.append(cv_results_to_frame(spec.name, search, dataset.key))

        write_text(CLASSIFICATION_REPORT_DIR / f"{model_slug}.txt", report_text)
        cm_frame.to_csv(CONFUSION_MATRIX_DIR / f"{model_slug}.csv")
        save_confusion_matrix(
            cm_frame,
            FIGURES_DIR / f"confusion_matrix_{model_slug}.png",
            f"Held-out Test Confusion Matrix: {spec.name}",
        )
        confusion_matrices[spec.name] = cm_frame
        joblib.dump(estimator, TRAINED_MODEL_DIR / f"{model_slug}.joblib")

        if spec.name in {"Logistic Regression", "Linear SVM"}:
            feature_importance_frames.append(
                extract_linear_feature_importance(
                    spec.name, estimator, dataset.feature_names
                )
            )

    metrics_frame = pd.DataFrame(metric_rows)
    best_model_name = metrics_frame.sort_values(
        ["test_f1_macro", "test_accuracy"], ascending=False
    ).iloc[0]["model"]
    metrics_frame["is_best_model"] = metrics_frame["model"].eq(best_model_name)
    metrics_frame = metrics_frame.sort_values("test_f1_macro", ascending=False)
    metrics_frame.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    class_metrics = metrics_frame[metrics_frame["model"].isin(CLASS_REPORT_MODELS)].copy()
    class_best_model_name = (
        class_metrics[class_metrics["model"].isin(CLASS_SELECTION_MODELS)]
        .sort_values(["test_f1_macro", "test_accuracy"], ascending=False)
        .iloc[0]["model"]
    )
    class_metrics["is_class_report_best_model"] = class_metrics["model"].eq(class_best_model_name)
    class_metrics = class_metrics.sort_values("test_f1_macro", ascending=False)
    class_metrics.to_csv(RESULTS_DIR / "class_model_metrics.csv", index=False)

    cv_results = pd.concat(cv_frames, ignore_index=True)
    cv_results.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    feature_importance = (
        pd.concat(feature_importance_frames, ignore_index=True)
        if feature_importance_frames
        else pd.DataFrame()
    )
    if not feature_importance.empty:
        feature_importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
        for model_name in ["Logistic Regression", "Linear SVM"]:
            save_coefficient_plot(
                feature_importance,
                model_name,
                FIGURES_DIR / f"sensor_coefficients_{slugify(model_name)}.png",
            )

    save_metric_comparison(
        metrics_frame,
        "f1_macro",
        FIGURES_DIR / "model_comparison_f1_all.png",
        "Full Portfolio Model Comparison: Macro F1",
        "Macro F1",
    )
    save_metric_comparison(
        metrics_frame,
        "accuracy",
        FIGURES_DIR / "model_comparison_accuracy_all.png",
        "Full Portfolio Model Comparison: Accuracy",
        "Accuracy",
    )
    save_metric_comparison(
        class_metrics,
        "f1_macro",
        FIGURES_DIR / "model_comparison_f1.png",
        "Class Report Model Comparison: Macro F1",
        "Macro F1",
    )
    save_metric_comparison(
        class_metrics,
        "accuracy",
        FIGURES_DIR / "model_comparison_accuracy.png",
        "Class Report Model Comparison: Accuracy",
        "Accuracy",
    )
    save_cv_score_comparison(metrics_frame, FIGURES_DIR / "cv_score_comparison.png")
    shutil.copyfile(
        FIGURES_DIR / f"confusion_matrix_{slugify(best_model_name)}.png",
        FIGURES_DIR / "confusion_matrix_best_overall_model.png",
    )
    shutil.copyfile(
        FIGURES_DIR / f"confusion_matrix_{slugify(class_best_model_name)}.png",
        FIGURES_DIR / "confusion_matrix_best_model.png",
    )

    return metrics_frame, best_params, confusion_matrices, feature_importance


def run_sensor_ablation(
    datasets: dict[str, RobotDataset],
    train_idx: pd.Index,
    test_idx: pd.Index,
) -> tuple[pd.DataFrame, dict]:
    """Compare the full 24-sensor file with official reduced-sensor UCI files."""
    rows = []
    best_params = {}
    print("\nRunning sensor ablation study")
    print("=============================")
    for dataset_key in ["24_sensors", "4_sensors", "2_sensors"]:
        dataset = datasets[dataset_key]
        X_train = dataset.X.iloc[train_idx]
        X_test = dataset.X.iloc[test_idx]
        y_train = dataset.y.iloc[train_idx]
        y_test = dataset.y.iloc[test_idx]
        best_params[dataset_key] = {}
        for spec in get_ablation_model_specs():
            print(f"Ablation: {dataset.display_name} | {spec.name}")
            search = fit_grid_search(spec, X_train, y_train)
            metrics, _, _ = evaluate_estimator(
                spec.name, search.best_estimator_, X_train, y_train, X_test, y_test
            )
            metrics.update(
                {
                    "dataset": dataset_key,
                    "feature_set": dataset.display_name,
                    "n_features": len(dataset.feature_names),
                    "cv_best_macro_f1": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "source_note": dataset.source_note,
                }
            )
            rows.append(metrics)
            best_params[dataset_key][spec.name] = {
                "best_params": search.best_params_,
                "best_cv_macro_f1": float(search.best_score_),
            }

    ablation = pd.DataFrame(rows).sort_values(
        ["feature_set", "test_f1_macro"], ascending=[True, False]
    )
    ablation.to_csv(RESULTS_DIR / "sensor_ablation_results.csv", index=False)
    save_sensor_ablation_comparison(
        ablation, FIGURES_DIR / "sensor_ablation_comparison.png"
    )
    return ablation, best_params


def generate_eda_outputs(dataset: RobotDataset) -> pd.DataFrame:
    """Create lightweight exploratory tables and figures for the report."""
    print("\nGenerating EDA outputs")
    print("======================")
    summary = dataset.X.agg(["mean", "std", "min", "max"]).T
    summary.index.name = "sensor"
    summary.to_csv(RESULTS_DIR / "sensor_summary_statistics.csv")
    save_class_distribution(dataset.y, FIGURES_DIR / "class_distribution.png")
    save_pca_sensor_space(dataset.X, dataset.y, FIGURES_DIR / "pca_sensor_space.png")
    save_sensor_correlation_heatmap(
        dataset.X, FIGURES_DIR / "sensor_correlation_heatmap.png"
    )
    return summary


def build_context(
    primary: RobotDataset,
    metrics: pd.DataFrame,
    best_params: dict,
    ablation: pd.DataFrame,
    feature_importance: pd.DataFrame,
    confusion_matrices: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Collect headline results used only for terminal summary output."""
    best_row = metrics.sort_values(["test_f1_macro", "test_accuracy"], ascending=False).iloc[0]
    best_model_name = str(best_row["model"])
    confusion = most_confused_pair(confusion_matrices[best_model_name])
    class_metrics = metrics[metrics["model"].isin(CLASS_REPORT_MODELS)].copy()
    class_best_row = (
        class_metrics[class_metrics["model"].isin(CLASS_SELECTION_MODELS)]
        .sort_values(["test_f1_macro", "test_accuracy"], ascending=False)
        .iloc[0]
    )
    class_best_model_name = str(class_best_row["model"])
    class_confusion = most_confused_pair(confusion_matrices[class_best_model_name])

    nonlinear_models = ["RBF SVM", "MLP Neural Network"]
    linear_models = ["Logistic Regression", "Linear SVM"]
    best_nonlinear = metrics[metrics["model"].isin(nonlinear_models)].sort_values(
        "test_f1_macro", ascending=False
    ).iloc[0]
    best_linear = metrics[metrics["model"].isin(linear_models)].sort_values(
        "test_f1_macro", ascending=False
    ).iloc[0]
    nonlinear_beats_linear = bool(best_nonlinear["test_f1_macro"] > best_linear["test_f1_macro"])

    rbf_row = metrics[metrics["model"] == "RBF SVM"].iloc[0]
    mlp_row = metrics[metrics["model"] == "MLP Neural Network"].iloc[0]

    ablation_best = ablation.sort_values(
        ["dataset", "test_f1_macro"], ascending=[True, False]
    ).groupby("dataset", as_index=False).head(1)
    full_ablation_f1 = float(
        ablation_best.loc[ablation_best["dataset"] == "24_sensors", "test_f1_macro"].iloc[0]
    )
    reduced_best_f1 = float(
        ablation_best.loc[
            ablation_best["dataset"].isin(["4_sensors", "2_sensors"]), "test_f1_macro"
        ].max()
    )
    sensor_reduction_hurt = full_ablation_f1 > reduced_best_f1

    top_features = pd.DataFrame()
    if not feature_importance.empty:
        top_features = feature_importance.groupby("model", as_index=False).head(5)

    class_counts = primary.y.value_counts().reindex(COMMAND_ORDER)

    return {
        "best_row": best_row,
        "best_model_name": best_model_name,
        "best_params": best_params["full_24_sensor_experiment"][best_model_name]["best_params"],
        "confusion": confusion,
        "class_best_row": class_best_row,
        "class_best_model_name": class_best_model_name,
        "class_best_params": best_params["full_24_sensor_experiment"][class_best_model_name]["best_params"],
        "class_confusion": class_confusion,
        "nonlinear_beats_linear": nonlinear_beats_linear,
        "best_nonlinear_model": str(best_nonlinear["model"]),
        "best_nonlinear_f1": float(best_nonlinear["test_f1_macro"]),
        "best_linear_model": str(best_linear["model"]),
        "best_linear_f1": float(best_linear["test_f1_macro"]),
        "rbf_test_f1": float(rbf_row["test_f1_macro"]),
        "mlp_test_f1": float(mlp_row["test_f1_macro"]),
        "sensor_reduction_hurt": sensor_reduction_hurt,
        "full_ablation_f1": full_ablation_f1,
        "reduced_best_f1": reduced_best_f1,
        "ablation_best": ablation_best,
        "top_features": top_features,
        "class_counts": class_counts,
        "missing_values": int(primary.data.isna().sum().sum()),
    }


def verify_required_outputs() -> list[str]:
    """Verify only analysis artifacts generated by this script."""
    model_slugs = [slugify(spec.name) for spec in get_main_model_specs()]
    required_paths = [
        RESULTS_DIR / "dataset_verification.csv",
        RESULTS_DIR / "class_model_metrics.csv",
        RESULTS_DIR / "model_metrics.csv",
        RESULTS_DIR / "cv_results.csv",
        RESULTS_DIR / "best_params.json",
        RESULTS_DIR / "feature_importance.csv",
        RESULTS_DIR / "sensor_ablation_results.csv",
        RESULTS_DIR / "sensor_summary_statistics.csv",
        FIGURES_DIR / "class_distribution.png",
        FIGURES_DIR / "pca_sensor_space.png",
        FIGURES_DIR / "model_comparison_f1.png",
        FIGURES_DIR / "model_comparison_accuracy.png",
        FIGURES_DIR / "model_comparison_f1_all.png",
        FIGURES_DIR / "model_comparison_accuracy_all.png",
        FIGURES_DIR / "confusion_matrix_best_model.png",
        FIGURES_DIR / "confusion_matrix_best_overall_model.png",
        FIGURES_DIR / "cv_score_comparison.png",
        FIGURES_DIR / "sensor_ablation_comparison.png",
        FIGURES_DIR / "sensor_coefficients_logistic_regression.png",
        FIGURES_DIR / "sensor_coefficients_linear_svm.png",
    ]
    for model_slug in model_slugs:
        required_paths.extend(
            [
                CLASSIFICATION_REPORT_DIR / f"{model_slug}.txt",
                CONFUSION_MATRIX_DIR / f"{model_slug}.csv",
                TRAINED_MODEL_DIR / f"{model_slug}.joblib",
                FIGURES_DIR / f"confusion_matrix_{model_slug}.png",
            ]
        )
    return [str(path.relative_to(PROJECT_ROOT)) for path in required_paths if not path.exists()]


def main() -> None:
    ensure_project_directories()
    download_all_datasets()
    datasets = load_all_datasets()
    primary = datasets["24_sensors"]
    print_dataset_summary(primary)

    train_idx, test_idx = make_train_test_indices(primary.y)
    print_split_summary(primary, train_idx, test_idx)
    write_dataset_verification(primary, train_idx, test_idx)
    generate_eda_outputs(primary)
    metrics, main_best_params, confusion_matrices, feature_importance = run_main_experiment(
        primary, train_idx, test_idx
    )
    ablation, ablation_best_params = run_sensor_ablation(datasets, train_idx, test_idx)
    best_params = {
        "full_24_sensor_experiment": main_best_params,
        "sensor_ablation": ablation_best_params,
    }
    write_json(RESULTS_DIR / "best_params.json", best_params)

    context = build_context(
        primary, metrics, best_params, ablation, feature_importance, confusion_matrices
    )

    missing_outputs = verify_required_outputs()
    if missing_outputs:
        raise RuntimeError(f"Missing required outputs: {missing_outputs}")

    best_row = context["best_row"]
    print("\nFinal summary")
    print("=============")
    print(f"Dataset shape: {primary.data.shape[0]} samples x {primary.data.shape[1]} columns")
    print("Models trained: Dummy Majority, Logistic Regression, Linear SVM, RBF SVM, MLP Neural Network")
    print(f"Best model: {context['best_model_name']}")
    print(f"Best test accuracy: {format_metric(float(best_row['test_accuracy']))}")
    print(f"Best test macro F1: {format_metric(float(best_row['test_f1_macro']))}")
    print(f"Best hyperparameters: {json.dumps(context['best_params'], sort_keys=True)}")
    print(f"Most confused command pair: {context['confusion']['description']}")
    print(f"Best class-report model: {context['class_best_model_name']}")
    print(f"Class-report test accuracy: {format_metric(float(context['class_best_row']['test_accuracy']))}")
    print(f"Class-report test macro F1: {format_metric(float(context['class_best_row']['test_f1_macro']))}")
    print(f"Class-report confusion pair: {context['class_confusion']['description']}")
    print(
        "Nonlinear beat linear: "
        f"{'yes' if context['nonlinear_beats_linear'] else 'no'} "
        f"({context['best_nonlinear_model']} {format_metric(context['best_nonlinear_f1'])} vs. "
        f"{context['best_linear_model']} {format_metric(context['best_linear_f1'])})"
    )
    print(
        "Sensor reduction hurt performance: "
        f"{'yes' if context['sensor_reduction_hurt'] else 'no'} "
        f"(24-sensor best {format_metric(context['full_ablation_f1'])}, "
        f"best reduced {format_metric(context['reduced_best_f1'])})"
    )
    if not feature_importance.empty:
        top = (
            feature_importance.sort_values("mean_abs_coefficient", ascending=False)
            .head(5)[["model", "feature", "sensor_region", "mean_abs_coefficient"]]
            .to_dict(orient="records")
        )
        print(f"Most important sensors by coefficient magnitude: {top}")
    print("Files created/updated: results CSV/JSON/TXT files, trained models, and figures.")
    print("Reports are static files in report/ and are not generated by main.py.")


if __name__ == "__main__":
    main()
