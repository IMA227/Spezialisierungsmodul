#loading libraries
import gc
import inspect
import json
import random
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from iterstrat.ml_stratifiers import (
        MultilabelStratifiedKFold,
        MultilabelStratifiedShuffleSplit,
    )
except ImportError as exc:
    raise ImportError("Install iterative-stratification first.") from exc

warnings.filterwarnings("ignore")

try:
    from IPython.display import display
except Exception:
    display = None


DATA_FILE = Path("...")
SHEET_NAME = 0

TEXT_COL = "review_text"
TOPIC_COLS = ["Topic_1", "Topic_2", "Topic_3"]
RELEVANCE_COL = "Relevance Flag"

MODEL_NAME = "deepset/gbert-large"

OUTPUT_DIR = Path("...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STEM = "..."

RESULTS_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_results.xlsx"
CONFIG_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_config.json"
CONFUSION_PLOT_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_confusion.png"
PR_PLOT_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_pr_curves.png"
ROC_PLOT_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_roc_curves.png"

MAX_LENGTH = 256
RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.20
N_SPLITS = 5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01

USE_CLASS_WEIGHTING = True
POS_WEIGHT_CLIP_MAX = 5.0

USE_GENERAL_OTHER_CLEANING = True
GENERAL_OTHER_LABEL = "General/Other"


LABEL_ORDER = [
    "Ambiance",
    "Food",
    "General/Other",
    "Price",
    "Service",
]

TRAIN_PARAM_GRID = [
    {"learning_rate": 1e-5, "num_epochs": 4},
    {"learning_rate": 1e-5, "num_epochs": 6},
    {"learning_rate": 2e-5, "num_epochs": 4},
    {"learning_rate": 2e-5, "num_epochs": 6},
]

THRESHOLD_GRID = [0.45, 0.50, 0.55]
GENERAL_OTHER_FALLBACK_GRID = [False, True]
METRIC_FOR_SELECTION = "f1_macro"

TOPIC_DEFINITIONS = {
    "Food": "Food, drinks, taste, quality, freshness, portions, and dishes.",
    "Service": "Staff, waiting time, orders, reservations, delivery, and customer handling.",
    "Ambiance": "Atmosphere, location, furniture, music, cleanliness, noise, and comfort.",
    "Price": "Prices, value for money, payment, offers, and unexpected charges.",
    "General/Other": "General overall opinions without a clear food, service, ambiance, or price aspect.",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def show_frame(frame):
    if display is not None:
        display(frame)
    else:
        print(frame.to_string(index=False))


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


def clean_topic(value):
    if pd.isna(value):
        return None

    topic = str(value).strip()

    if topic.lower() in {"", "nan", "none", "null", "-", "—"}:
        return None

    return topic


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def unpack_eval_pred(eval_pred):
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    return logits, labels


def apply_threshold(probabilities, threshold=0.5):
    return (probabilities >= threshold).astype(int)


def apply_general_other_fallback(predictions, label_names):
    if GENERAL_OTHER_LABEL not in label_names:
        return predictions

    predictions = predictions.copy()
    general_idx = label_names.index(GENERAL_OTHER_LABEL)
    specific_indices = [i for i, label in enumerate(label_names) if label != GENERAL_OTHER_LABEL]

    for row_idx in range(len(predictions)):
        has_specific_topic = predictions[row_idx, specific_indices].sum() > 0
        predictions[row_idx, general_idx] = 0 if has_specific_topic else 1

    return predictions


def make_predictions(probabilities, threshold, use_fallback, label_names):
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
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, predictions, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, predictions, average="micro", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
    }

    return {key: float(value) for key, value in metrics.items()}


def compute_metrics_for_trainer(eval_pred):
    logits, labels = unpack_eval_pred(eval_pred)
    probabilities = sigmoid(logits)
    predictions = apply_threshold(probabilities, 0.5)

    return evaluate_multilabel_predictions(labels, predictions)


def format_mean_std(mean_value, std_value):
    if pd.isna(std_value):
        std_value = 0.0

    return f"{mean_value:.4f} +- {std_value:.4f}"


def compute_pos_weight(labels):
    positive_counts = labels.sum(axis=0)
    negative_counts = len(labels) - positive_counts
    pos_weight = negative_counts / np.maximum(positive_counts, 1)
    pos_weight = np.clip(pos_weight, 1.0, POS_WEIGHT_CLIP_MAX)

    return torch.tensor(pos_weight, dtype=torch.float)


class ReviewTopicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        item = {key: torch.tensor(value, dtype=torch.long) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class WeightedMultiLabelTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is not None:
            loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        else:
            loss_func = torch.nn.BCEWithLogitsLoss()

        loss = loss_func(logits, labels)

        return (loss, outputs) if return_outputs else loss


def build_training_args(run_output_dir, learning_rate, num_epochs, seed, do_epoch_eval=False):
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {"output_dir": str(run_output_dir)}

    def add(name, value):
        if name in params:
            kwargs[name] = value

    add("num_train_epochs", num_epochs)
    add("per_device_train_batch_size", TRAIN_BATCH_SIZE)
    add("per_device_eval_batch_size", EVAL_BATCH_SIZE)
    add("learning_rate", learning_rate)
    add("weight_decay", WEIGHT_DECAY)
    add("logging_strategy", "epoch")
    add("save_strategy", "no")
    add("report_to", "none")
    add("seed", seed)
    add("data_seed", seed)

    if do_epoch_eval:
        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in params:
            kwargs["evaluation_strategy"] = "epoch"

    if torch.cuda.is_available():
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

        if bf16_supported and "bf16" in params:
            kwargs["bf16"] = True
        elif "fp16" in params:
            kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def build_trainer(
    model,
    training_args,
    train_dataset,
    tokenizer,
    data_collator,
    pos_weight,
    eval_dataset=None,
    compute_metrics=None,
):
    trainer_params = inspect.signature(Trainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "pos_weight": pos_weight,
    }

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if compute_metrics is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return WeightedMultiLabelTrainer(**trainer_kwargs)


def load_labeled_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
    required_cols = [TEXT_COL, RELEVANCE_COL] + TOPIC_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in Excel file: {missing_cols}")

    before_filter = len(df)
    df = df[~df[RELEVANCE_COL].apply(is_relevance_zero)].copy()
    df = df.dropna(subset=[TEXT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].copy()

    print(f"original_rows: {before_filter}")
    print(f"usable_text_rows: {len(df)}")

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
        raise ValueError(f"Unknown topic labels found: {unknown_topics}")

    before_mixed_general = df[
        df["topics_original"].apply(lambda x: GENERAL_OTHER_LABEL in x and len(x) > 1)
    ]
    after_mixed_general = df[
        df["topics"].apply(lambda x: GENERAL_OTHER_LABEL in x and len(x) > 1)
    ]

    print(f"general_plus_specific_before_cleaning: {len(before_mixed_general)}")
    print(f"general_plus_specific_after_cleaning: {len(after_mixed_general)}")

    mlb = MultiLabelBinarizer(classes=LABEL_ORDER)
    y = mlb.fit_transform(df["topics"]).astype(np.float32)

    label_names = list(mlb.classes_)
    texts = df[TEXT_COL].tolist()

    print(f"usable_reviews: {len(df)}")

    for i, label in enumerate(label_names):
        print(f"{label}: {int(y[:, i].sum())}")

    return df, texts, y, label_names


def make_model(label_names):
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )


def evaluate_thresholds(val_labels, val_probabilities, params, fold_idx, label_names):
    rows = []

    for threshold in THRESHOLD_GRID:
        for use_fallback in GENERAL_OTHER_FALLBACK_GRID:
            fold_predictions = make_predictions(
                val_probabilities,
                threshold=threshold,
                use_fallback=use_fallback,
                label_names=label_names,
            )

            fold_metrics = evaluate_multilabel_predictions(val_labels, fold_predictions)
            row = {
                "fold": fold_idx,
                "learning_rate": params["learning_rate"],
                "num_epochs": params["num_epochs"],
                "threshold": threshold,
                "use_general_other_fallback": use_fallback,
            }
            row.update(fold_metrics)
            rows.append(row)

    return rows


def run_cv_grid_search(X_train_pool, y_train_pool, tokenizer, data_collator, label_names):
    cv = MultilabelStratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    cv_results_rows = []
    oof_store = {}
    total_runs = len(TRAIN_PARAM_GRID) * N_SPLITS
    run_counter = 0

    print(f"cv_training_runs: {total_runs}")

    for param_idx, params in enumerate(TRAIN_PARAM_GRID, start=1):
        train_key = (float(params["learning_rate"]), int(params["num_epochs"]))
        oof_store[train_key] = {"folds": []}

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
            cv.split(np.zeros(len(y_train_pool)), y_train_pool.astype(int)),
            start=1,
        ):
            run_counter += 1
            run_seed = RANDOM_SEED + (param_idx * 100) + fold_idx
            set_seed(run_seed)

            X_fold_train = [X_train_pool[i] for i in fold_train_idx]
            y_fold_train = y_train_pool[fold_train_idx]
            X_fold_val = [X_train_pool[i] for i in fold_val_idx]
            y_fold_val = y_train_pool[fold_val_idx]

            print(f"run: {run_counter}/{total_runs}")
            print(f"fold_train_size: {len(X_fold_train)}")
            print(f"fold_validation_size: {len(X_fold_val)}")

            pos_weight = compute_pos_weight(y_fold_train) if USE_CLASS_WEIGHTING else None

            if pos_weight is not None:
                for label, weight in zip(label_names, pos_weight):
                    print(f"weight_{label}: {float(weight):.3f}")

            model = make_model(label_names)
            train_dataset = ReviewTopicDataset(X_fold_train, y_fold_train, tokenizer, MAX_LENGTH)
            val_dataset = ReviewTopicDataset(X_fold_val, y_fold_val, tokenizer, MAX_LENGTH)

            run_output_dir = OUTPUT_DIR / f"tmp_{param_idx}_{fold_idx}"
            training_args = build_training_args(
                run_output_dir=run_output_dir,
                learning_rate=params["learning_rate"],
                num_epochs=params["num_epochs"],
                seed=run_seed,
            )

            trainer = build_trainer(
                model=model,
                training_args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                pos_weight=pos_weight,
            )

            trainer.train()
            val_output = trainer.predict(val_dataset)

            val_logits = val_output.predictions
            val_labels = val_output.label_ids.astype(int)
            val_probabilities = sigmoid(val_logits)

            oof_store[train_key]["folds"].append(
                {
                    "fold": fold_idx,
                    "texts": X_fold_val,
                    "labels": val_labels,
                    "probabilities": val_probabilities,
                }
            )

            cv_results_rows.extend(
                evaluate_thresholds(
                    val_labels=val_labels,
                    val_probabilities=val_probabilities,
                    params=params,
                    fold_idx=fold_idx,
                    label_names=label_names,
                )
            )

            del trainer
            del model
            torch.cuda.empty_cache()
            gc.collect()
            shutil.rmtree(run_output_dir, ignore_errors=True)

    return pd.DataFrame(cv_results_rows), oof_store


def summarize_cv_results(cv_results_df):
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
        "learning_rate",
        "num_epochs",
        "threshold",
        "use_general_other_fallback",
    ]

    records = []

    for group_values, group in cv_results_df.groupby(group_cols):
        learning_rate, num_epochs, threshold, use_fallback = group_values
        record = {
            "learning_rate": float(learning_rate),
            "num_epochs": int(num_epochs),
            "threshold": float(threshold),
            "use_general_other_fallback": bool(use_fallback),
        }

        for metric in metric_cols:
            mean_value = group[metric].mean()
            std_value = group[metric].std(ddof=1)
            record[metric] = format_mean_std(mean_value, std_value)
            record[f"{metric}_mean"] = float(mean_value)
            record[f"{metric}_std"] = float(std_value)

        records.append(record)

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values(
        [f"{METRIC_FOR_SELECTION}_mean", "f1_micro_mean", "accuracy_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    display_cols = group_cols + metric_cols
    return summary_df, summary_df[display_cols].copy(), metric_cols


def collect_oof_predictions(oof_store, best_params, label_names):
    best_train_key = (best_params["learning_rate"], best_params["num_epochs"])
    best_folds = oof_store[best_train_key]["folds"]

    oof_texts = []
    oof_labels = []
    oof_probabilities = []
    oof_fold_ids = []

    for fold_data in best_folds:
        n_rows = len(fold_data["texts"])
        oof_texts.extend(fold_data["texts"])
        oof_labels.append(fold_data["labels"])
        oof_probabilities.append(fold_data["probabilities"])
        oof_fold_ids.extend([fold_data["fold"]] * n_rows)

    oof_labels = np.vstack(oof_labels).astype(int)
    oof_probabilities = np.vstack(oof_probabilities)
    oof_predictions = make_predictions(
        oof_probabilities,
        threshold=best_params["threshold"],
        use_fallback=best_params["use_general_other_fallback"],
        label_names=label_names,
    )

    return oof_texts, oof_labels, oof_probabilities, oof_predictions, oof_fold_ids, best_folds


def build_per_topic_cv_summary(best_folds, best_params, label_names):
    rows = []

    for fold_data in best_folds:
        fold_labels = fold_data["labels"].astype(int)
        fold_predictions = make_predictions(
            fold_data["probabilities"],
            threshold=best_params["threshold"],
            use_fallback=best_params["use_general_other_fallback"],
            label_names=label_names,
        )

        report = classification_report(
            fold_labels,
            fold_predictions,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )

        for label in label_names:
            rows.append(
                {
                    "fold": fold_data["fold"],
                    "topic": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
            )

    per_topic_cv_df = pd.DataFrame(rows)
    summary_rows = []

    for topic, topic_df in per_topic_cv_df.groupby("topic"):
        record = {"topic": topic, "mean_support": float(topic_df["support"].mean())}

        for metric in ["precision", "recall", "f1"]:
            record[metric] = format_mean_std(
                topic_df[metric].mean(),
                topic_df[metric].std(ddof=1),
            )
            record[f"{metric}_mean"] = float(topic_df[metric].mean())
            record[f"{metric}_std"] = float(topic_df[metric].std(ddof=1))

        summary_rows.append(record)

    return per_topic_cv_df, pd.DataFrame(summary_rows)


def train_final_model(X_train_pool, y_train_pool, tokenizer, data_collator, best_params, label_names):
    final_train_seed = RANDOM_SEED + 999
    set_seed(final_train_seed)

    final_pos_weight = compute_pos_weight(y_train_pool) if USE_CLASS_WEIGHTING else None

    if final_pos_weight is not None:
        for label, weight in zip(label_names, final_pos_weight):
            print(f"final_weight_{label}: {float(weight):.3f}")

    final_model = make_model(label_names)
    final_train_dataset = ReviewTopicDataset(X_train_pool, y_train_pool, tokenizer, MAX_LENGTH)
    final_output_dir = OUTPUT_DIR / "tmp_final"

    final_training_args = build_training_args(
        run_output_dir=final_output_dir,
        learning_rate=best_params["learning_rate"],
        num_epochs=best_params["num_epochs"],
        seed=final_train_seed,
    )

    final_trainer = build_trainer(
        model=final_model,
        training_args=final_training_args,
        train_dataset=final_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        pos_weight=final_pos_weight,
    )

    final_trainer.train()

    return final_trainer, final_model, final_train_seed, final_output_dir


def make_prediction_table(texts, labels, predictions, probabilities, label_names, fold_ids=None):
    data = {"review_text": texts}

    if fold_ids is not None:
        data = {"fold": fold_ids, **data}

    table = pd.DataFrame(data)

    for i, label in enumerate(label_names):
        table[f"true_{label}"] = labels[:, i]
        table[f"pred_{label}"] = predictions[:, i]
        table[f"prob_{label}"] = probabilities[:, i]

    return table


def build_confusion_table(labels, predictions, label_names):
    confusion_matrices = multilabel_confusion_matrix(labels, predictions)
    rows = []

    for i, label in enumerate(label_names):
        tn, fp, fn, tp = confusion_matrices[i].ravel()
        rows.append(
            {
                "topic": label,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    return pd.DataFrame(rows), confusion_matrices


def save_confusion_plot(confusion_matrices, label_names):
    fig, axes = plt.subplots(1, len(label_names), figsize=(4 * len(label_names), 4))

    if len(label_names) == 1:
        axes = [axes]

    for i, label in enumerate(label_names):
        ax = axes[i]
        cm = confusion_matrices[i]
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
                ax.text(col_idx, row_idx, str(cm[row_idx, col_idx]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()


def build_pr_results(labels, probabilities, label_names):
    rows = []
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(label_names):
        y_true = labels[:, i]
        y_score = probabilities[:, i]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        plt.plot(recall, precision, label=f"{label} AP={ap:.3f}")
        rows.append({"topic": label, "average_precision": float(ap)})

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final-test precision-recall curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(PR_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()

    return pd.DataFrame(rows)


def build_roc_results(labels, probabilities, label_names):
    rows = []
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(label_names):
        y_true = labels[:, i]
        y_score = probabilities[:, i]

        if len(np.unique(y_true)) < 2:
            print(f"roc_skipped_{label}: one_class_only")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        plt.plot(fpr, tpr, label=f"{label} AUC={auc_score:.3f}")
        rows.append({"topic": label, "roc_auc": float(auc_score)})

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final-test ROC curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()

    return pd.DataFrame(rows)


def save_results(
    cv_results_df,
    summary_display_df,
    per_topic_cv_df,
    per_topic_cv_summary_df,
    oof_metrics,
    oof_predictions_df,
    final_test_metrics,
    final_test_report_df,
    final_confusion_df,
    pr_scores_df,
    roc_scores_df,
    final_test_predictions_df,
    experiment_config,
):
    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
        cv_results_df.to_excel(writer, sheet_name="cv_folds", index=False)
        summary_display_df.to_excel(writer, sheet_name="cv_summary", index=False)
        per_topic_cv_df.to_excel(writer, sheet_name="cv_topics_folds", index=False)
        per_topic_cv_summary_df.to_excel(writer, sheet_name="cv_topics_summary", index=False)
        pd.DataFrame([oof_metrics]).to_excel(writer, sheet_name="oof_metrics", index=False)
        oof_predictions_df.to_excel(writer, sheet_name="oof_predictions", index=False)
        pd.DataFrame([final_test_metrics]).to_excel(writer, sheet_name="final_metrics", index=False)
        final_test_report_df.to_excel(writer, sheet_name="final_report")
        final_confusion_df.to_excel(writer, sheet_name="final_confusion", index=False)
        pr_scores_df.to_excel(writer, sheet_name="final_pr_scores", index=False)
        roc_scores_df.to_excel(writer, sheet_name="final_roc_scores", index=False)
        final_test_predictions_df.to_excel(writer, sheet_name="final_predictions", index=False)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)

    print(f"results_file: {RESULTS_FILE}")
    print(f"config_file: {CONFIG_FILE}")


def main():
    set_seed(RANDOM_SEED)
    print(f"transformers_version: {transformers.__version__}")
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    _, texts, y, label_names = load_labeled_data()

    final_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    # This split is not touched during model selection.
    train_pool_idx, final_test_idx = next(final_splitter.split(np.zeros(len(y)), y.astype(int)))

    X_train_pool = [texts[i] for i in train_pool_idx]
    y_train_pool = y[train_pool_idx]
    X_final_test = [texts[i] for i in final_test_idx]
    y_final_test = y[final_test_idx]

    print(f"train_pool_size: {len(X_train_pool)}")
    print(f"final_test_size: {len(X_final_test)}")

    for i, label in enumerate(label_names):
        print(f"train_pool_{label}: {int(y_train_pool[:, i].sum())}")
        print(f"final_test_{label}: {int(y_final_test[:, i].sum())}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

 
    cv_results_df, oof_store = run_cv_grid_search(
        X_train_pool=X_train_pool,
        y_train_pool=y_train_pool,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_names=label_names,
    )

    summary_df, summary_display_df, metric_cols = summarize_cv_results(cv_results_df)
    show_frame(summary_display_df)

    best_row = summary_df.iloc[0]
    best_params = {
        "learning_rate": float(best_row["learning_rate"]),
        "num_epochs": int(best_row["num_epochs"]),
        "threshold": float(best_row["threshold"]),
        "use_general_other_fallback": bool(best_row["use_general_other_fallback"]),
    }

    print(json.dumps(best_params, indent=2))

    for metric in metric_cols:
        print(f"best_cv_{metric}: {best_row[metric]}")

    oof_texts, oof_labels, oof_probabilities, oof_predictions, oof_fold_ids, best_folds = collect_oof_predictions(
        oof_store=oof_store,
        best_params=best_params,
        label_names=label_names,
    )

    oof_metrics = evaluate_multilabel_predictions(oof_labels, oof_predictions)

    for metric_name, metric_value in oof_metrics.items():
        print(f"oof_{metric_name}: {metric_value:.4f}")

    per_topic_cv_df, per_topic_cv_summary_df = build_per_topic_cv_summary(
        best_folds=best_folds,
        best_params=best_params,
        label_names=label_names,
    )

    show_frame(per_topic_cv_summary_df[["topic", "precision", "recall", "f1", "mean_support"]])

    final_trainer, final_model, final_train_seed, final_output_dir = train_final_model(
        X_train_pool=X_train_pool,
        y_train_pool=y_train_pool,
        tokenizer=tokenizer,
        data_collator=data_collator,
        best_params=best_params,
        label_names=label_names,
    )

    final_test_dataset = ReviewTopicDataset(X_final_test, y_final_test, tokenizer, MAX_LENGTH)
    final_test_output = final_trainer.predict(final_test_dataset)

    final_test_logits = final_test_output.predictions
    final_test_labels = final_test_output.label_ids.astype(int)
    final_test_probabilities = sigmoid(final_test_logits)
    final_test_predictions = make_predictions(
        final_test_probabilities,
        threshold=best_params["threshold"],
        use_fallback=best_params["use_general_other_fallback"],
        label_names=label_names,
    )

    final_test_metrics = evaluate_multilabel_predictions(final_test_labels, final_test_predictions)

    for metric_name, metric_value in final_test_metrics.items():
        print(f"final_test_{metric_name}: {metric_value:.4f}")

    final_test_report_df = pd.DataFrame(
        classification_report(
            final_test_labels,
            final_test_predictions,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
    ).transpose()

    show_frame(final_test_report_df)

    final_confusion_df, final_confusion_matrices = build_confusion_table(
        final_test_labels,
        final_test_predictions,
        label_names,
    )

    show_frame(final_confusion_df)
    save_confusion_plot(final_confusion_matrices, label_names)

    pr_scores_df = build_pr_results(final_test_labels, final_test_probabilities, label_names)
    roc_scores_df = build_roc_results(final_test_labels, final_test_probabilities, label_names)

    oof_predictions_df = make_prediction_table(
        texts=oof_texts,
        labels=oof_labels,
        predictions=oof_predictions,
        probabilities=oof_probabilities,
        label_names=label_names,
        fold_ids=oof_fold_ids,
    )

    final_test_predictions_df = make_prediction_table(
        texts=X_final_test,
        labels=final_test_labels,
        predictions=final_test_predictions,
        probabilities=final_test_probabilities,
        label_names=label_names,
    )

    experiment_config = {
        "design": "train_pool CV grid search plus independent final holdout test",
        "data_file": str(DATA_FILE),
        "sheet_name": SHEET_NAME,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "final_test_size": FINAL_TEST_SIZE,
        "n_splits": N_SPLITS,
        "label_order": LABEL_ORDER,
        "topic_definitions": TOPIC_DEFINITIONS,
        "train_param_grid": TRAIN_PARAM_GRID,
        "threshold_grid": THRESHOLD_GRID,
        "general_other_fallback_grid": GENERAL_OTHER_FALLBACK_GRID,
        "metric_for_selection": METRIC_FOR_SELECTION,
        "best_params": best_params,
        "use_class_weighting": USE_CLASS_WEIGHTING,
        "use_general_other_cleaning": USE_GENERAL_OTHER_CLEANING,
        "pos_weight_clip_max": POS_WEIGHT_CLIP_MAX,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "final_train_seed": final_train_seed,
        "relevance_filter": f"{RELEVANCE_COL} != 0",
    }


    save_results(
        cv_results_df=cv_results_df,
        summary_display_df=summary_display_df,
        per_topic_cv_df=per_topic_cv_df,
        per_topic_cv_summary_df=per_topic_cv_summary_df,
        oof_metrics=oof_metrics,
        oof_predictions_df=oof_predictions_df,
        final_test_metrics=final_test_metrics,
        final_test_report_df=final_test_report_df,
        final_confusion_df=final_confusion_df,
        pr_scores_df=pr_scores_df,
        roc_scores_df=roc_scores_df,
        final_test_predictions_df=final_test_predictions_df,
        experiment_config=experiment_config,
    )

    del final_trainer
    del final_model
    torch.cuda.empty_cache()
    gc.collect()
    shutil.rmtree(final_output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
