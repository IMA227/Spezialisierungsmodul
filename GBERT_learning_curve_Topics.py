# loading libraries
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
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)



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

RESULTS_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_learning_curve.xlsx"
CONFIG_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_config.json"
OVERALL_PLOT_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_overall_curve.png"
TOPIC_F1_PLOT_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_topic_f1_curve.png"
TOPIC_PRECISION_PLOT_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_topic_precision_curve.png"
TOPIC_RECALL_PLOT_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_topic_recall_curve.png"

MAX_LENGTH = 256
RANDOM_SEED = 42
TEST_SIZE = 0.20

LEARNING_RATE = 1e-5
NUM_EPOCHS = 6
DECISION_THRESHOLD = 0.45

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01

USE_CLASS_WEIGHTING = True
POS_WEIGHT_CLIP_MAX = 5.0

USE_GENERAL_OTHER_CLEANING = True
USE_GENERAL_OTHER_FALLBACK = True
GENERAL_OTHER_LABEL = "General/Other"

LABEL_ORDER = [
    "Ambiance",
    "Food",
    "General/Other",
    "Price",
    "Service",
]

TRAIN_FRACTIONS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
N_REPEATS = 3

RELATIVE_PERFORMANCE_TARGET = 0.95
ABSOLUTE_GOOD_F1_TARGET = 0.85

TOPIC_DEFINITIONS = {
    "Food": "Food, drinks, taste, freshness, quality, portions, and dishes.",
    "Service": "Staff, waiting time, ordering, reservations, delivery, and customer handling.",
    "Ambiance": "Atmosphere, location, furniture, music, cleanliness, noise, and comfort.",
    "Price": "Prices, value for money, offers, payment, and unexpected charges.",
    "General/Other": "General overall comments without a clear specific aspect.",
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


def per_topic_metrics(labels, predictions, label_names):
    report = classification_report(
        labels.astype(int),
        predictions.astype(int),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    rows = []

    for label in label_names:
        rows.append(
            {
                "topic": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
        )

    return rows


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


def get_subset_indices_multilabel(y_pool, fraction, seed):
    if fraction >= 1.0:
        return np.arange(len(y_pool))

    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=fraction,
        test_size=1.0 - fraction,
        random_state=seed,
    )

    subset_idx, _ = next(
        splitter.split(
            np.zeros(len(y_pool)),
            y_pool.astype(int),
        )
    )

    return subset_idx


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


def build_training_args(run_output_dir, seed):
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {"output_dir": str(run_output_dir)}

    def add(name, value):
        if name in params:
            kwargs[name] = value

    add("num_train_epochs", NUM_EPOCHS)
    add("per_device_train_batch_size", TRAIN_BATCH_SIZE)
    add("per_device_eval_batch_size", EVAL_BATCH_SIZE)
    add("learning_rate", LEARNING_RATE)
    add("weight_decay", WEIGHT_DECAY)
    add("logging_strategy", "epoch")
    add("save_strategy", "no")
    add("report_to", "none")
    add("seed", seed)
    add("data_seed", seed)

    if torch.cuda.is_available():
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

        if bf16_supported and "bf16" in params:
            kwargs["bf16"] = True
        elif "fp16" in params:
            kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def build_trainer(model, training_args, train_dataset, tokenizer, data_collator, pos_weight):
    trainer_params = inspect.signature(Trainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "pos_weight": pos_weight,
    }

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

    original_rows = len(df)
    df = df[~df[RELEVANCE_COL].apply(is_relevance_zero)].copy()
    df = df.dropna(subset=[TEXT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].copy()

    print(f"original_rows: {original_rows}")
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

    return texts, y, label_names


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


def split_train_test(texts, y, label_names):
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    train_pool_idx, test_idx = next(splitter.split(np.zeros(len(y)), y.astype(int)))

    X_train_pool = [texts[i] for i in train_pool_idx]
    y_train_pool = y[train_pool_idx]

    X_test = [texts[i] for i in test_idx]
    y_test = y[test_idx]

    print(f"train_pool_size: {len(X_train_pool)}")
    print(f"test_size: {len(X_test)}")

    for i, label in enumerate(label_names):
        print(f"test_{label}: {int(y_test[:, i].sum())}")

    return X_train_pool, y_train_pool, X_test, y_test


def train_one_run(
    fraction,
    repeat_idx,
    X_train_pool,
    y_train_pool,
    test_dataset,
    tokenizer,
    data_collator,
    label_names,
):
    seed = RANDOM_SEED + int(fraction * 1000) + repeat_idx
    set_seed(seed)

    subset_idx = get_subset_indices_multilabel(
        y_train_pool,
        fraction=fraction,
        seed=seed,
    )

    X_train_subset = [X_train_pool[i] for i in subset_idx]
    y_train_subset = y_train_pool[subset_idx]
    train_n = len(X_train_subset)

    print(f"train_fraction: {fraction:.2f}")
    print(f"repeat: {repeat_idx + 1}")
    print(f"train_n: {train_n}")

    for i, label in enumerate(label_names):
        print(f"subset_{label}: {int(y_train_subset[:, i].sum())}")

    pos_weight = compute_pos_weight(y_train_subset) if USE_CLASS_WEIGHTING else None

    if pos_weight is not None:
        for label, weight in zip(label_names, pos_weight):
            print(f"weight_{label}: {float(weight):.3f}")

    model = make_model(label_names)
    train_dataset = ReviewTopicDataset(X_train_subset, y_train_subset, tokenizer, MAX_LENGTH)
    run_output_dir = OUTPUT_DIR / f"tmp_{int(fraction * 100)}_{repeat_idx + 1}"

    training_args = build_training_args(run_output_dir=run_output_dir, seed=seed)

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        pos_weight=pos_weight,
    )

    trainer.train()

    test_output = trainer.predict(test_dataset)
    test_labels = test_output.label_ids.astype(int)
    test_probabilities = sigmoid(test_output.predictions)
    test_predictions = make_predictions(
        test_probabilities,
        threshold=DECISION_THRESHOLD,
        use_fallback=USE_GENERAL_OTHER_FALLBACK,
        label_names=label_names,
    )

    overall_metrics = evaluate_multilabel_predictions(test_labels, test_predictions)

    for metric_name, metric_value in overall_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    topic_metric_rows = per_topic_metrics(test_labels, test_predictions, label_names)

    del trainer
    del model

    torch.cuda.empty_cache()
    gc.collect()
    shutil.rmtree(run_output_dir, ignore_errors=True)

    return train_n, seed, overall_metrics, topic_metric_rows


def run_learning_curve(X_train_pool, y_train_pool, X_test, y_test, tokenizer, data_collator, label_names):
    test_dataset = ReviewTopicDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    learning_rows = []
    topic_rows = []

    total_runs = len(TRAIN_FRACTIONS) * N_REPEATS
    run_counter = 0

    print(f"total_runs: {total_runs}")

    for fraction in TRAIN_FRACTIONS:
        for repeat_idx in range(N_REPEATS):
            run_counter += 1
            print(f"run: {run_counter}/{total_runs}")

            train_n, seed, overall_metrics, topic_metric_rows = train_one_run(
                fraction=fraction,
                repeat_idx=repeat_idx,
                X_train_pool=X_train_pool,
                y_train_pool=y_train_pool,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                label_names=label_names,
            )

            row = {
                "train_fraction": fraction,
                "train_n": train_n,
                "repeat": repeat_idx + 1,
                "seed": seed,
            }
            row.update(overall_metrics)
            learning_rows.append(row)

            for topic_row in topic_metric_rows:
                topic_row["train_fraction"] = fraction
                topic_row["train_n"] = train_n
                topic_row["repeat"] = repeat_idx + 1
                topic_row["seed"] = seed
                topic_rows.append(topic_row)

    return pd.DataFrame(learning_rows), pd.DataFrame(topic_rows)


def summarize_learning_curve(learning_df):
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

    records = []

    for train_fraction, group in learning_df.groupby("train_fraction"):
        record = {
            "train_fraction": float(train_fraction),
            "mean_train_n": float(group["train_n"].mean()),
            "std_train_n": float(group["train_n"].std(ddof=0)),
            "min_train_n": int(group["train_n"].min()),
            "max_train_n": int(group["train_n"].max()),
            "n_repeats": int(len(group)),
        }

        for metric in metric_cols:
            record[metric] = format_mean_std(group[metric].mean(), group[metric].std(ddof=0))
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=0))

        records.append(record)

    return pd.DataFrame(records).sort_values("train_fraction"), metric_cols


def summarize_topic_learning(per_topic_df):
    records = []

    for (train_fraction, topic), group in per_topic_df.groupby(["train_fraction", "topic"]):
        record = {
            "train_fraction": float(train_fraction),
            "mean_train_n": float(group["train_n"].mean()),
            "std_train_n": float(group["train_n"].std(ddof=0)),
            "topic": topic,
            "n_repeats": int(len(group)),
            "mean_support": float(group["support"].mean()),
        }

        for metric in ["precision", "recall", "f1"]:
            record[metric] = format_mean_std(group[metric].mean(), group[metric].std(ddof=0))
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=0))

        records.append(record)

    return pd.DataFrame(records).sort_values(["topic", "train_fraction"])


def find_smallest_fraction_for_target(summary_df, metric_mean_col, target_value):
    eligible = summary_df[summary_df[metric_mean_col] >= target_value].copy()

    if len(eligible) == 0:
        return None, None

    best_row = eligible.sort_values("train_fraction").iloc[0]

    return float(best_row["train_fraction"]), float(best_row["mean_train_n"])


def build_sufficiency_tables(learning_summary_df, topic_learning_summary_df, label_names):
    full_row = learning_summary_df[learning_summary_df["train_fraction"] == 1.0].iloc[0]
    sufficiency_rows = []

    for metric in ["f1_macro", "f1_micro", "accuracy", "labelwise_accuracy"]:
        full_value = float(full_row[f"{metric}_mean"])
        relative_target = RELATIVE_PERFORMANCE_TARGET * full_value

        fraction_relative, n_relative = find_smallest_fraction_for_target(
            learning_summary_df,
            f"{metric}_mean",
            relative_target,
        )

        row = {
            "metric": metric,
            "full_data_mean_value": full_value,
            "relative_target_95pct_of_full": relative_target,
            "smallest_fraction_reaching_95pct_of_full": fraction_relative,
            "mean_train_n_at_95pct_of_full": n_relative,
        }

        if "f1" in metric:
            fraction_absolute, n_absolute = find_smallest_fraction_for_target(
                learning_summary_df,
                f"{metric}_mean",
                ABSOLUTE_GOOD_F1_TARGET,
            )

            row[f"smallest_fraction_reaching_absolute_f1_{ABSOLUTE_GOOD_F1_TARGET}"] = fraction_absolute
            row[f"mean_train_n_reaching_absolute_f1_{ABSOLUTE_GOOD_F1_TARGET}"] = n_absolute

        sufficiency_rows.append(row)

    topic_sufficiency_rows = []

    for topic in label_names:
        topic_df = topic_learning_summary_df[topic_learning_summary_df["topic"] == topic].copy()
        full_topic_row = topic_df[topic_df["train_fraction"] == 1.0].iloc[0]
        full_topic_f1 = float(full_topic_row["f1_mean"])
        relative_target = RELATIVE_PERFORMANCE_TARGET * full_topic_f1

        eligible_relative = topic_df[topic_df["f1_mean"] >= relative_target]
        eligible_absolute = topic_df[topic_df["f1_mean"] >= ABSOLUTE_GOOD_F1_TARGET]

        fraction_relative = None
        n_relative = None
        fraction_absolute = None
        n_absolute = None

        if len(eligible_relative) > 0:
            row_relative = eligible_relative.sort_values("train_fraction").iloc[0]
            fraction_relative = float(row_relative["train_fraction"])
            n_relative = float(row_relative["mean_train_n"])

        if len(eligible_absolute) > 0:
            row_absolute = eligible_absolute.sort_values("train_fraction").iloc[0]
            fraction_absolute = float(row_absolute["train_fraction"])
            n_absolute = float(row_absolute["mean_train_n"])

        topic_sufficiency_rows.append(
            {
                "topic": topic,
                "full_data_mean_f1": full_topic_f1,
                "relative_target_95pct_of_full_f1": relative_target,
                "smallest_fraction_reaching_95pct_of_full_f1": fraction_relative,
                "mean_train_n_reaching_95pct_of_full_f1": n_relative,
                f"smallest_fraction_reaching_absolute_f1_{ABSOLUTE_GOOD_F1_TARGET}": fraction_absolute,
                f"mean_train_n_reaching_absolute_f1_{ABSOLUTE_GOOD_F1_TARGET}": n_absolute,
            }
        )

    return pd.DataFrame(sufficiency_rows), pd.DataFrame(topic_sufficiency_rows)


def plot_overall_curves(learning_summary_df):
    plt.figure(figsize=(9, 6))

    for metric in ["f1_macro", "f1_micro", "accuracy", "labelwise_accuracy"]:
        plt.errorbar(
            learning_summary_df["mean_train_n"].values,
            learning_summary_df[f"{metric}_mean"].values,
            yerr=learning_summary_df[f"{metric}_std"].values,
            marker="o",
            capsize=4,
            label=metric,
        )

    plt.xlabel("Mean number of training samples")
    plt.ylabel("Score")
    plt.title("Overall learning curves")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(OVERALL_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()


def plot_topic_curve(topic_learning_summary_df, label_names, metric, output_file):
    plt.figure(figsize=(10, 6))

    for topic in label_names:
        topic_df = topic_learning_summary_df[
            topic_learning_summary_df["topic"] == topic
        ].sort_values("train_fraction")

        plt.errorbar(
            topic_df["mean_train_n"].values,
            topic_df[f"{metric}_mean"].values,
            yerr=topic_df[f"{metric}_std"].values,
            marker="o",
            capsize=4,
            label=topic,
        )

    plt.xlabel("Mean number of training samples")
    plt.ylabel(metric.title())
    plt.title(f"Per-topic {metric} learning curves")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.show()


def save_outputs(
    learning_df,
    learning_summary_df,
    per_topic_df,
    topic_learning_summary_df,
    sufficiency_df,
    topic_sufficiency_df,
    experiment_config,
):
    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
        learning_df.to_excel(writer, sheet_name="overall_runs", index=False)
        learning_summary_df.to_excel(writer, sheet_name="overall_summary", index=False)
        per_topic_df.to_excel(writer, sheet_name="topic_runs", index=False)
        topic_learning_summary_df.to_excel(writer, sheet_name="topic_summary", index=False)
        sufficiency_df.to_excel(writer, sheet_name="sufficiency_overall", index=False)
        topic_sufficiency_df.to_excel(writer, sheet_name="sufficiency_topics", index=False)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)


def main():
    set_seed(RANDOM_SEED)

    print(f"transformers_version: {transformers.__version__}")
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    texts, y, label_names = load_labeled_data()
    X_train_pool, y_train_pool, X_test, y_test = split_train_test(texts, y, label_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # I use the same fixed test set for every training-size run.
    learning_df, per_topic_df = run_learning_curve(
        X_train_pool=X_train_pool,
        y_train_pool=y_train_pool,
        X_test=X_test,
        y_test=y_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_names=label_names,
    )

    learning_summary_df, _ = summarize_learning_curve(learning_df)
    topic_learning_summary_df = summarize_topic_learning(per_topic_df)

    overall_display_cols = [
        "train_fraction",
        "mean_train_n",
        "std_train_n",
        "min_train_n",
        "max_train_n",
        "n_repeats",
        "accuracy",
        "labelwise_accuracy",
        "f1_macro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
    ]

    show_frame(learning_summary_df[overall_display_cols])
    show_frame(
        topic_learning_summary_df[
            [
                "topic",
                "train_fraction",
                "mean_train_n",
                "std_train_n",
                "n_repeats",
                "precision",
                "recall",
                "f1",
                "mean_support",
            ]
        ]
    )

    sufficiency_df, topic_sufficiency_df = build_sufficiency_tables(
        learning_summary_df=learning_summary_df,
        topic_learning_summary_df=topic_learning_summary_df,
        label_names=label_names,
    )

    show_frame(sufficiency_df)
    show_frame(topic_sufficiency_df)

    plot_overall_curves(learning_summary_df)
    plot_topic_curve(topic_learning_summary_df, label_names, "f1", TOPIC_F1_PLOT_FILE)
    plot_topic_curve(topic_learning_summary_df, label_names, "precision", TOPIC_PRECISION_PLOT_FILE)
    plot_topic_curve(topic_learning_summary_df, label_names, "recall", TOPIC_RECALL_PLOT_FILE)

    experiment_config = {
        "data_file": str(DATA_FILE),
        "sheet_name": SHEET_NAME,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "train_fractions": TRAIN_FRACTIONS,
        "n_repeats": N_REPEATS,
        "full_data_repeated": True,
        "aggregation_level": "train_fraction",
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "decision_threshold": DECISION_THRESHOLD,
        "use_general_other_fallback": USE_GENERAL_OTHER_FALLBACK,
        "use_class_weighting": USE_CLASS_WEIGHTING,
        "pos_weight_clip_max": POS_WEIGHT_CLIP_MAX,
        "weight_decay": WEIGHT_DECAY,
        "use_general_other_cleaning": USE_GENERAL_OTHER_CLEANING,
        "label_order": LABEL_ORDER,
        "topic_definitions": TOPIC_DEFINITIONS,
        "relative_performance_target": RELATIVE_PERFORMANCE_TARGET,
        "absolute_good_f1_target": ABSOLUTE_GOOD_F1_TARGET,
        "relevance_filter": f"{RELEVANCE_COL} != 0",
    }

    
    save_outputs(
        learning_df=learning_df,
        learning_summary_df=learning_summary_df,
        per_topic_df=per_topic_df,
        topic_learning_summary_df=topic_learning_summary_df,
        sufficiency_df=sufficiency_df,
        topic_sufficiency_df=topic_sufficiency_df,
        experiment_config=experiment_config,
    )


if __name__ == "__main__":
    main()
