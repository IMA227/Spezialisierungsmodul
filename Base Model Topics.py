#loading libraries:

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    multilabel_confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)

try:
    from iterstrat.ml_stratifiers import (
        MultilabelStratifiedKFold,
        MultilabelStratifiedShuffleSplit,
    )
except ImportError:
    raise ImportError(
        "Please install iterative-stratification first:\n"
        "!pip install iterative-stratification"
    )

warnings.filterwarnings("ignore")

# basic settings
EXCEL_PATH = r"..."
SHEET_NAME = 0

TEXT_COL = "review_text"
TOPIC_COLS = ["Topic_1", "Topic_2", "Topic_3"]
RELEVANCE_COL = "Relevance Flag"

OUTPUT_DIR = "..."
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.20
N_SPLITS = 5

USE_GENERAL_OTHER_CLEANING = True
GENERAL_OTHER_LABEL = "General/Other"

LABEL_ORDER = [
    "Ambiance",
    "Food",
    "General/Other",
    "Price",
    "Service",
]

BASELINE_PARAM_GRID = [
    {
        "tfidf__ngram_range": (1, 1),
        "tfidf__max_features": 20000,
        "clf__estimator__C": 0.5,
        "clf__estimator__class_weight": "balanced",
    },
    {
        "tfidf__ngram_range": (1, 2),
        "tfidf__max_features": 30000,
        "clf__estimator__C": 0.5,
        "clf__estimator__class_weight": "balanced",
    },
    {
        "tfidf__ngram_range": (1, 2),
        "tfidf__max_features": 50000,
        "clf__estimator__C": 1.0,
        "clf__estimator__class_weight": "balanced",
    },
    {
        "tfidf__ngram_range": (1, 2),
        "tfidf__max_features": 50000,
        "clf__estimator__C": 2.0,
        "clf__estimator__class_weight": "balanced",
    },
    {
        "tfidf__ngram_range": (1, 3),
        "tfidf__max_features": 50000,
        "clf__estimator__C": 1.0,
        "clf__estimator__class_weight": "balanced",
    },
    {
        "tfidf__ngram_range": (1, 2),
        "tfidf__max_features": 50000,
        "clf__estimator__C": 1.0,
        "clf__estimator__class_weight": None,
    },
]

THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50, 0.55]
GENERAL_OTHER_FALLBACK_GRID = [False, True]

METRIC_FOR_SELECTION = "f1_macro"

print("Baseline model: TF-IDF + One-vs-Rest Logistic Regression")
print(f"Output directory: {OUTPUT_DIR}")

# keep the run reproducible
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(RANDOM_SEED)

TOPIC_DEFINITIONS = {
    "Food": (
        "Mentions about the food and drinks themselves. This includes ingredients, "
        "taste, freshness, quality, portion size, cooking level, healthiness, dietary "
        "suitability, specific dishes, drinks, desserts, appetizers, coffee, cocktails, "
        "and complaints about food being cold, burned, spoiled, salty, bland, or "
        "incorrectly prepared."
    ),
    "Service": (
        "Mentions about how the restaurant serves and supports customers. This includes "
        "staff behavior, friendliness, professionalism, attentiveness, availability, "
        "service experience, waiting time, ordering process, wrong or forgotten orders, "
        "reservations, delivery, opening hours, seating process, accessibility, pets "
        "allowed, handling of complaints, and general customer treatment."
    ),
    "Ambiance": (
        "Mentions about the physical environment and atmosphere of the restaurant. This "
        "includes decoration, furniture, tables, doors, music, TV, live performance, "
        "room size, air conditioning, bathroom/restroom, smoking area, buffet area, bar, "
        "patio, dining room, outside view, location/area, cleanliness, noise level, "
        "lighting, and comfort of the space."
    ),
    "Price": (
        "Mentions about cost, value for money, discounts, offers, payment, and perceived "
        "worthiness. This includes prices, expensive or cheap impressions, fair or unfair "
        "pricing, happy hours, buy-one-get-one offers, birthday offers, payment methods, "
        "payment problems, unexpected charges, and whether the portion size or quality "
        "is worth the price."
    ),
    "General/Other": (
        "General opinions that describe the overall restaurant experience without "
        "focusing on a specific aspect such as food, service, ambiance, or price. This "
        "includes overall satisfaction, overall dissatisfaction, general recommendations, "
        "star-rating-style comments, and very short reviews such as 'Alles top', "
        "'Nicht wieder', 'Geht so', or 'Sehr empfehlenswert' when no specific reason "
        "is given."
    ),
}

if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

# load the labelled sample
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

required_cols = [TEXT_COL, RELEVANCE_COL] + TOPIC_COLS
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing columns in Excel file: {missing_cols}")

print(f"\nOriginal rows: {len(df)}")

def is_relevance_zero(value):
    if pd.isna(value):
        return False

    value_str = str(value).strip().lower()

    if value_str in {"0", "0.0"}:
        return True

    try:
        return float(value_str) == 0.0
    except Exception:
        return False

before_filter = len(df)
df = df[~df[RELEVANCE_COL].apply(is_relevance_zero)].copy()
after_filter = len(df)

print(f"Rows removed because {RELEVANCE_COL} == 0: {before_filter - after_filter}")
print(f"Rows after relevance filtering: {after_filter}")

df = df.dropna(subset=[TEXT_COL]).copy()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""].copy()

print(f"Rows after non-empty text filtering: {len(df)}")

def clean_topic(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    invalid_values = {"", "nan", "none", "null", "-", "—"}

    if value.lower() in invalid_values:
        return None

    return value

def collect_topics(row):
    topics = []

    for col in TOPIC_COLS:
        topic = clean_topic(row[col])

        if topic is not None:
            topics.append(topic)

    return list(dict.fromkeys(topics))

def clean_general_other_rule(topics):
    topics = list(topics)

    if GENERAL_OTHER_LABEL in topics and len(topics) > 1:
        topics = [topic for topic in topics if topic != GENERAL_OTHER_LABEL]

    return topics

df["topics_original"] = df.apply(collect_topics, axis=1)
df = df[df["topics_original"].apply(len) > 0].copy()

if USE_GENERAL_OTHER_CLEANING:
    df["topics"] = df["topics_original"].apply(clean_general_other_rule)
else:
    df["topics"] = df["topics_original"]

df = df[df["topics"].apply(len) > 0].copy()

unknown_topics = sorted(
    {
        topic
        for topics in df["topics"]
        for topic in topics
        if topic not in LABEL_ORDER
    }
)

if unknown_topics:
    raise ValueError(
        f"Unknown topic labels found: {unknown_topics}. "
        f"Expected only: {LABEL_ORDER}"
    )

before_general_plus_specific = df[
    df["topics_original"].apply(
        lambda x: GENERAL_OTHER_LABEL in x and len(x) > 1
    )
]

after_general_plus_specific = df[
    df["topics"].apply(
        lambda x: GENERAL_OTHER_LABEL in x and len(x) > 1
    )
]

print("\nGeneral/Other cleaning check:")
print(
    "Rows with General/Other + specific topics before cleaning:",
    len(before_general_plus_specific),
)
print(
    "Rows with General/Other + specific topics after cleaning:",
    len(after_general_plus_specific),
)

mlb = MultiLabelBinarizer(classes=LABEL_ORDER)
Y = mlb.fit_transform(df["topics"]).astype(np.int32)

label_names = list(mlb.classes_)
num_labels = len(label_names)
texts = df[TEXT_COL].tolist()

print(f"\nNumber of usable reviews: {len(df)}")
print(f"Number of topic labels: {num_labels}")

for i, label in enumerate(label_names):
    print(f"{i}: {label} | count: {int(Y[:, i].sum())}")

# keep the final test set separate from the CV process
final_splitter = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=FINAL_TEST_SIZE,
    random_state=RANDOM_SEED,
)

train_pool_idx, final_test_idx = next(
    final_splitter.split(
        np.zeros(len(Y)),
        Y.astype(int),
    )
)

X_train_pool = [texts[i] for i in train_pool_idx]
y_train_pool = Y[train_pool_idx]

X_final_test = [texts[i] for i in final_test_idx]
y_final_test = Y[final_test_idx]

print("\n==============================")
print("Independent holdout split")
print("==============================")
print(f"Train pool size for CV and final training: {len(X_train_pool)}")
print(f"Final untouched test size: {len(X_final_test)}")

print("\nTrain-pool label counts:")
for i, label in enumerate(label_names):
    print(f"{label}: {int(y_train_pool[:, i].sum())}")

print("\nFinal-test label counts:")
for i, label in enumerate(label_names):
    print(f"{label}: {int(y_final_test[:, i].sum())}")

def apply_threshold(probabilities, threshold=0.5):
    return (probabilities >= threshold).astype(int)

def apply_general_other_fallback(predictions, label_names):
    if GENERAL_OTHER_LABEL not in label_names:
        return predictions

    predictions = predictions.copy()
    general_idx = label_names.index(GENERAL_OTHER_LABEL)

    specific_indices = [
        i
        for i, label in enumerate(label_names)
        if label != GENERAL_OTHER_LABEL
    ]

    for row_idx in range(len(predictions)):
        has_specific_topic = predictions[row_idx, specific_indices].sum() > 0

        if has_specific_topic:
            predictions[row_idx, general_idx] = 0
        else:
            predictions[row_idx, general_idx] = 1

    return predictions

def make_predictions(probabilities, threshold, use_fallback):
    predictions = apply_threshold(probabilities, threshold)

    if use_fallback:
        predictions = apply_general_other_fallback(predictions, label_names)

    return predictions

def evaluate_multilabel_predictions(labels, predictions):
    labels = labels.astype(int)
    predictions = predictions.astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "labelwise_accuracy": float((labels == predictions).mean()),
        "precision_macro": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "precision_micro": precision_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "recall_micro": recall_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "f1_micro": f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
    }

    return {key: float(value) for key, value in metrics.items()}

def format_mean_std(mean_value, std_value):
    if pd.isna(std_value):
        std_value = 0.0

    return f"{mean_value:.4f} +- {std_value:.4f}"

# TF-IDF stays inside the pipeline to avoid leakage
def build_baseline_model():
    logistic_regression = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        random_state=RANDOM_SEED,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    analyzer="word",
                    token_pattern=r"(?u)\b\w\w+\b",
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(logistic_regression),
            ),
        ]
    )

    return model

def get_probabilities(model, texts):
    probabilities = model.predict_proba(texts)

    if isinstance(probabilities, list):
        probabilities = np.column_stack(
            [p[:, 1] if p.ndim == 2 else p for p in probabilities]
        )

    return np.asarray(probabilities, dtype=float)

cv = MultilabelStratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

cv_results_rows = []
oof_store = {}

total_training_runs = len(BASELINE_PARAM_GRID) * N_SPLITS
run_counter = 0

print("\n==============================")
print("==============================")
print(f"Training parameter combinations: {len(BASELINE_PARAM_GRID)}")
print(f"Evaluation thresholds: {THRESHOLD_GRID}")
print(f"General/Other fallback options: {GENERAL_OTHER_FALLBACK_GRID}")
print(f"Total model trainings: {total_training_runs}")


for param_idx, params in enumerate(BASELINE_PARAM_GRID, start=1):
    train_key = json.dumps(
        {
            key: str(value)
            for key, value in params.items()
        },
        sort_keys=True,
    )

    oof_store[train_key] = {
        "params": params,
        "folds": [],
    }

 
    print(f"Training parameter set {param_idx}/{len(BASELINE_PARAM_GRID)}")
    print(params)
    

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
        cv.split(np.zeros(len(y_train_pool)), y_train_pool.astype(int)),
        start=1,
    ):
        run_counter += 1

        print("\n---------------------------------------------")
        print(f"Run {run_counter}/{total_training_runs}")
        print(f"Fold {fold_idx}/{N_SPLITS}")
        print("---------------------------------------------")

        X_fold_train = [X_train_pool[i] for i in fold_train_idx]
        y_fold_train = y_train_pool[fold_train_idx]

        X_fold_val = [X_train_pool[i] for i in fold_val_idx]
        y_fold_val = y_train_pool[fold_val_idx]

        print(f"Fold train size: {len(X_fold_train)}")
        print(f"Fold validation size: {len(X_fold_val)}")

        model = build_baseline_model()
        model.set_params(**params)

        model.fit(X_fold_train, y_fold_train)

        val_probabilities = get_probabilities(model, X_fold_val)

        oof_store[train_key]["folds"].append(
            {
                "fold": fold_idx,
                "texts": X_fold_val,
                "labels": y_fold_val.astype(int),
                "probabilities": val_probabilities,
            }
        )

        for threshold in THRESHOLD_GRID:
            for use_fallback in GENERAL_OTHER_FALLBACK_GRID:
                fold_predictions = make_predictions(
                    val_probabilities,
                    threshold=threshold,
                    use_fallback=use_fallback,
                )

                fold_metrics = evaluate_multilabel_predictions(
                    y_fold_val,
                    fold_predictions,
                )

                row = {
                    "fold": fold_idx,
                    "param_idx": param_idx,
                    "ngram_range": str(params["tfidf__ngram_range"]),
                    "max_features": params["tfidf__max_features"],
                    "C": params["clf__estimator__C"],
                    "class_weight": str(params["clf__estimator__class_weight"]),
                    "threshold": threshold,
                    "use_general_other_fallback": use_fallback,
                    "train_key": train_key,
                }

                row.update(fold_metrics)
                cv_results_rows.append(row)

cv_results_df = pd.DataFrame(cv_results_rows)

metric_cols = [
    "accuracy",
    "labelwise_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
]

group_cols = [
    "param_idx",
    "ngram_range",
    "max_features",
    "C",
    "class_weight",
    "threshold",
    "use_general_other_fallback",
    "train_key",
]

summary_records = []

for group_values, group in cv_results_df.groupby(group_cols):
    group_dict = dict(zip(group_cols, group_values))

    record = {
        "param_idx": int(group_dict["param_idx"]),
        "ngram_range": group_dict["ngram_range"],
        "max_features": int(group_dict["max_features"]),
        "C": float(group_dict["C"]),
        "class_weight": group_dict["class_weight"],
        "threshold": float(group_dict["threshold"]),
        "use_general_other_fallback": bool(group_dict["use_general_other_fallback"]),
        "train_key": group_dict["train_key"],
    }

    for metric in metric_cols:
        mean_value = group[metric].mean()
        std_value = group[metric].std(ddof=1)

        record[metric] = format_mean_std(mean_value, std_value)
        record[f"{metric}_mean"] = float(mean_value)
        record[f"{metric}_std"] = float(std_value)

    summary_records.append(record)

summary_df = pd.DataFrame(summary_records)

sort_cols = [
    f"{METRIC_FOR_SELECTION}_mean",
    "f1_micro_mean",
    "accuracy_mean",
]

summary_df = summary_df.sort_values(
    sort_cols,
    ascending=[False, False, False],
).reset_index(drop=True)

display_cols = [
    "param_idx",
    "ngram_range",
    "max_features",
    "C",
    "class_weight",
    "threshold",
    "use_general_other_fallback",
] + metric_cols

summary_display_df = summary_df[display_cols].copy()

print("\n==============================")
print("Grid Search CV Results on train_pool")
print("Format: mean +- std over 5 folds")
print("==============================")

try:
    from IPython.display import display

    display(summary_display_df)
except Exception:
    print(summary_display_df.to_string(index=False))

best_row = summary_df.iloc[0]

best_train_key = best_row["train_key"]
best_model_params = oof_store[best_train_key]["params"]

best_params = {
    "param_idx": int(best_row["param_idx"]),
    "tfidf_ngram_range": best_row["ngram_range"],
    "tfidf_max_features": int(best_row["max_features"]),
    "logreg_C": float(best_row["C"]),
    "logreg_class_weight": best_row["class_weight"],
    "threshold": float(best_row["threshold"]),
    "use_general_other_fallback": bool(best_row["use_general_other_fallback"]),
}

print("\n==============================")
print("Best Parameter Combination from CV")
print("==============================")
print(json.dumps(best_params, indent=2))

print("\nBest CV metrics:")
for metric in metric_cols:
    print(f"{metric}: {best_row[metric]}")

best_folds = oof_store[best_train_key]["folds"]

oof_texts = []
oof_labels = []
oof_probabilities = []
oof_fold_ids = []

for fold_data in best_folds:
    n_fold_rows = len(fold_data["texts"])

    oof_texts.extend(fold_data["texts"])
    oof_labels.append(fold_data["labels"])
    oof_probabilities.append(fold_data["probabilities"])
    oof_fold_ids.extend([fold_data["fold"]] * n_fold_rows)

oof_labels = np.vstack(oof_labels).astype(int)
oof_probabilities = np.vstack(oof_probabilities)

oof_predictions = make_predictions(
    oof_probabilities,
    threshold=best_params["threshold"],
    use_fallback=best_params["use_general_other_fallback"],
)

oof_metrics = evaluate_multilabel_predictions(
    oof_labels,
    oof_predictions,
)

print("\n==============================")
print("Train-Pool Out-of-Fold Metrics for Best CV Config")
print("These are model-selection results, not final test performance.")
print("==============================")

for metric_name, metric_value in oof_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

per_topic_cv_rows = []

for fold_data in best_folds:
    fold_id = fold_data["fold"]
    fold_labels = fold_data["labels"].astype(int)
    fold_probs = fold_data["probabilities"]

    fold_preds = make_predictions(
        fold_probs,
        threshold=best_params["threshold"],
        use_fallback=best_params["use_general_other_fallback"],
    )

    report = classification_report(
        fold_labels,
        fold_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    for label in label_names:
        per_topic_cv_rows.append(
            {
                "fold": fold_id,
                "topic": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
        )

per_topic_cv_df = pd.DataFrame(per_topic_cv_rows)

per_topic_cv_summary_rows = []

for topic, topic_df in per_topic_cv_df.groupby("topic"):
    record = {
        "topic": topic,
        "mean_support": float(topic_df["support"].mean()),
    }

    for metric in ["precision", "recall", "f1"]:
        record[metric] = format_mean_std(
            topic_df[metric].mean(),
            topic_df[metric].std(ddof=1),
        )
        record[f"{metric}_mean"] = float(topic_df[metric].mean())
        record[f"{metric}_std"] = float(topic_df[metric].std(ddof=1))

    per_topic_cv_summary_rows.append(record)

per_topic_cv_summary_df = pd.DataFrame(per_topic_cv_summary_rows)



try:
    display(
        per_topic_cv_summary_df[
            [
                "topic",
                "precision",
                "recall",
                "f1",
                "mean_support",
            ]
        ]
    )
except Exception:
    print(per_topic_cv_summary_df.to_string(index=False))

print("\n==============================")

print("==============================")
print(json.dumps(best_params, indent=2))


# train the selected model on the full training pool
final_model = build_baseline_model()
final_model.set_params(**best_model_params)

final_model.fit(X_train_pool, y_train_pool)

print("\n==============================")


final_test_probabilities = get_probabilities(final_model, X_final_test)

final_test_predictions = make_predictions(
    final_test_probabilities,
    threshold=best_params["threshold"],
    use_fallback=best_params["use_general_other_fallback"],
)

final_test_metrics = evaluate_multilabel_predictions(
    y_final_test,
    final_test_predictions,
)

print("\n==============================")


for metric_name, metric_value in final_test_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

final_test_report_dict = classification_report(
    y_final_test,
    final_test_predictions,
    target_names=label_names,
    zero_division=0,
    output_dict=True,
)

final_test_report_df = pd.DataFrame(final_test_report_dict).transpose()

print("\n==============================")
print("Final-Test Classification Report")


try:
    display(final_test_report_df)
except Exception:
    print(final_test_report_df.to_string())

final_confusion_matrices = multilabel_confusion_matrix(
    y_final_test,
    final_test_predictions,
)

final_confusion_rows = []

for i, label in enumerate(label_names):
    cm = final_confusion_matrices[i]
    tn, fp, fn, tp = cm.ravel()

    final_confusion_rows.append(
        {
            "topic": label,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    )

final_confusion_df = pd.DataFrame(final_confusion_rows)

print("\n==============================")
print("Final-Test Confusion Matrix Values per Topic")


try:
    display(final_confusion_df)
except Exception:
    print(final_confusion_df.to_string(index=False))

fig, axes = plt.subplots(
    1,
    num_labels,
    figsize=(4 * num_labels, 4),
)

if num_labels == 1:
    axes = [axes]

for i, label in enumerate(label_names):
    ax = axes[i]
    cm = final_confusion_matrices[i]

    ax.imshow(cm)
    ax.set_title(label)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for row_idx in range(2):
        for col_idx in range(2):
            ax.text(
                col_idx,
                row_idx,
                str(cm[row_idx, col_idx]),
                ha="center",
                va="center",
            )

plt.tight_layout()

confusion_plot_path = os.path.join(
    OUTPUT_DIR,
    "final_test_confusion_matrices_per_topic.png",
)

plt.savefig(confusion_plot_path, dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))

pr_curve_rows = []

for i, label in enumerate(label_names):
    y_true_i = y_final_test[:, i]
    y_score_i = final_test_probabilities[:, i]

    precision, recall, _ = precision_recall_curve(
        y_true_i,
        y_score_i,
    )

    ap = average_precision_score(
        y_true_i,
        y_score_i,
    )

    plt.plot(
        recall,
        precision,
        label=f"{label} AP={ap:.3f}",
    )

    pr_curve_rows.append(
        {
            "topic": label,
            "average_precision": float(ap),
        }
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("TF-IDF + Logistic Regression Final-Test PR Curves per Topic")
plt.legend()
plt.grid(True)

pr_plot_path = os.path.join(
    OUTPUT_DIR,
    "final_test_precision_recall_curves_per_topic.png",
)

plt.savefig(pr_plot_path, dpi=200, bbox_inches="tight")
plt.show()

pr_scores_df = pd.DataFrame(pr_curve_rows)

print("\n==============================")
print("Final-Test Average Precision per Topic")
print("==============================")

try:
    display(pr_scores_df)
except Exception:
    print(pr_scores_df.to_string(index=False))

plt.figure(figsize=(8, 6))

roc_curve_rows = []

for i, label in enumerate(label_names):
    y_true_i = y_final_test[:, i]
    y_score_i = final_test_probabilities[:, i]

    if len(np.unique(y_true_i)) < 2:
        print(f"Skipping ROC for {label}: only one class present.")
        continue

    fpr, tpr, _ = roc_curve(
        y_true_i,
        y_score_i,
    )

    auc_score = roc_auc_score(
        y_true_i,
        y_score_i,
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{label} AUC={auc_score:.3f}",
    )

    roc_curve_rows.append(
        {
            "topic": label,
            "roc_auc": float(auc_score),
        }
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("TF-IDF + Logistic Regression Final-Test ROC Curves per Topic")
plt.legend()
plt.grid(True)

roc_plot_path = os.path.join(
    OUTPUT_DIR,
    "final_test_roc_curves_per_topic.png",
)

plt.savefig(roc_plot_path, dpi=200, bbox_inches="tight")
plt.show()

roc_scores_df = pd.DataFrame(roc_curve_rows)

print("\n==============================")
print("Final-Test ROC-AUC per Topic")


try:
    display(roc_scores_df)
except Exception:
    print(roc_scores_df.to_string(index=False))

oof_predictions_df = pd.DataFrame(
    {
        "fold": oof_fold_ids,
        "review_text": oof_texts,
    }
)

for i, label in enumerate(label_names):
    oof_predictions_df[f"true_{label}"] = oof_labels[:, i]
    oof_predictions_df[f"pred_{label}"] = oof_predictions[:, i]
    oof_predictions_df[f"prob_{label}"] = oof_probabilities[:, i]

final_test_predictions_df = pd.DataFrame(
    {
        "review_text": X_final_test,
    }
)

for i, label in enumerate(label_names):
    final_test_predictions_df[f"true_{label}"] = y_final_test[:, i]
    final_test_predictions_df[f"pred_{label}"] = final_test_predictions[:, i]
    final_test_predictions_df[f"prob_{label}"] = final_test_probabilities[:, i]

cv_results_path = os.path.join(
    OUTPUT_DIR,
    "cv_results_train_pool_all_folds.xlsx",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "grid_search_summary_train_pool_mean_std.xlsx",
)

best_params_path = os.path.join(
    OUTPUT_DIR,
    "best_params_from_train_pool_cv.json",
)

oof_metrics_path = os.path.join(
    OUTPUT_DIR,
    "best_config_train_pool_oof_metrics.json",
)

per_topic_cv_path = os.path.join(
    OUTPUT_DIR,
    "per_topic_cv_summary_train_pool.xlsx",
)

oof_predictions_path = os.path.join(
    OUTPUT_DIR,
    "best_config_train_pool_oof_predictions.xlsx",
)

final_test_metrics_path = os.path.join(
    OUTPUT_DIR,
    "independent_final_test_metrics.json",
)

final_test_report_path = os.path.join(
    OUTPUT_DIR,
    "independent_final_test_classification_report.xlsx",
)

final_confusion_path = os.path.join(
    OUTPUT_DIR,
    "independent_final_test_confusion_matrices_per_topic.xlsx",
)

pr_scores_path = os.path.join(
    OUTPUT_DIR,
    "independent_final_test_average_precision_per_topic.xlsx",
)

roc_scores_path = os.path.join(
    OUTPUT_DIR,
    "independent_final_test_roc_auc_per_topic.xlsx",
)

final_predictions_path = os.path.join(
    OUTPUT_DIR,
    "independent_final_test_predictions.xlsx",
)

topic_definitions_path = os.path.join(
    OUTPUT_DIR,
    "topic_definitions.json",
)

experiment_config_path = os.path.join(
    OUTPUT_DIR,
    "experiment_config_tfidf_logreg_option1.json",
)

# write the experiment outputs
cv_results_df.to_excel(cv_results_path, index=False)
summary_display_df.to_excel(summary_path, index=False)
per_topic_cv_summary_df.to_excel(per_topic_cv_path, index=False)
oof_predictions_df.to_excel(oof_predictions_path, index=False)

final_test_report_df.to_excel(final_test_report_path)
final_confusion_df.to_excel(final_confusion_path, index=False)
pr_scores_df.to_excel(pr_scores_path, index=False)
roc_scores_df.to_excel(roc_scores_path, index=False)
final_test_predictions_df.to_excel(final_predictions_path, index=False)

with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)

with open(oof_metrics_path, "w", encoding="utf-8") as f:
    json.dump(oof_metrics, f, ensure_ascii=False, indent=2)

with open(final_test_metrics_path, "w", encoding="utf-8") as f:
    json.dump(final_test_metrics, f, ensure_ascii=False, indent=2)

with open(topic_definitions_path, "w", encoding="utf-8") as f:
    json.dump(
        TOPIC_DEFINITIONS,
        f,
        ensure_ascii=False,
        indent=2,
    )

experiment_config = {
    "design": "TF-IDF + One-vs-Rest Logistic Regression baseline with train_pool CV grid search plus independent final holdout test",
    "excel_path": EXCEL_PATH,
    "sheet_name": SHEET_NAME,
    "text_column": TEXT_COL,
    "topic_columns": TOPIC_COLS,
    "relevance_column": RELEVANCE_COL,
    "final_test_size": FINAL_TEST_SIZE,
    "n_splits_cv_on_train_pool": N_SPLITS,
    "label_order": LABEL_ORDER,
    "baseline_param_grid": [
        {
            key: str(value)
            for key, value in params.items()
        }
        for params in BASELINE_PARAM_GRID
    ],
    "threshold_grid": THRESHOLD_GRID,
    "general_other_fallback_grid": GENERAL_OTHER_FALLBACK_GRID,
    "metric_for_selection": METRIC_FOR_SELECTION,
    "best_params": best_params,
    "use_general_other_cleaning": USE_GENERAL_OTHER_CLEANING,
    "random_seed": RANDOM_SEED,
    "relevance_filter": f"{RELEVANCE_COL} != 0",
    "note": (
        "The independent final_test split is created before CV and is not used "
        "for hyperparameter, threshold, or fallback selection. TF-IDF is fitted "
        "inside the Pipeline on each training fold only."
    ),
}

with open(experiment_config_path, "w", encoding="utf-8") as f:
    json.dump(
        experiment_config,
        f,
        ensure_ascii=False,
        indent=2,
    )
