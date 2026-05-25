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



warnings.filterwarnings("ignore")

try:
    from IPython.display import display
except Exception:
    display = None


LABELED_FILE = Path("...")
POPULATION_FILE = Path("...")
OUTPUT_DIR = Path("...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = "..."
LABELED_SHEET_NAME = 0
POPULATION_SHEET_NAME = 0

TEXT_COL = "review_text"
TOPIC_COLS = ["Topic_1", "Topic_2", "Topic_3"]
RELEVANCE_COL = "Relevance Flag"

MODEL_NAME = "deepset/gbert-large"

MAX_LENGTH = 256
RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.20

BEST_LEARNING_RATE = 1e-5
BEST_NUM_EPOCHS = 6
BEST_THRESHOLD = 0.45
USE_GENERAL_OTHER_FALLBACK = True
MAX_TOPICS_PER_REVIEW = 3

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
WEIGHT_DECAY = 0.01

USE_CLASS_WEIGHTING = True
POS_WEIGHT_CLIP_MAX = 5.0

USE_GENERAL_OTHER_CLEANING = True
GENERAL_OTHER_LABEL = "General/Other"
SOURCE_ROW_ID_COL = "source_row_id"

LABEL_ORDER = [
    "Ambiance",
    "Food",
    "General/Other",
    "Price",
    "Service",
]

EVALUATION_FILE = OUTPUT_DIR / f"{RUN_NAME}_evaluation.xlsx"
POPULATION_XLSX_FILE = OUTPUT_DIR / f"{RUN_NAME}_population_long.xlsx"
POPULATION_CSV_FILE = OUTPUT_DIR / f"{RUN_NAME}_population_long.csv"
SUMMARY_FILE = OUTPUT_DIR / f"{RUN_NAME}_summary.xlsx"
CONFIG_FILE = OUTPUT_DIR / f"{RUN_NAME}_config.json"
MODEL_DIR = OUTPUT_DIR / f"{RUN_NAME}_model"
TMP_EVAL_DIR = OUTPUT_DIR / f"{RUN_NAME}_tmp_eval"

CONFUSION_PLOT_FILE = OUTPUT_DIR / f"{RUN_NAME}_confusion.png"
PR_PLOT_FILE = OUTPUT_DIR / f"{RUN_NAME}_pr_curves.png"
ROC_PLOT_FILE = OUTPUT_DIR / f"{RUN_NAME}_roc_curves.png"
POPULATION_PLOT_FILE = OUTPUT_DIR / f"{RUN_NAME}_topic_distribution.png"


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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_excel_safely(frame, path, index=False):
    excel_limit = 1_048_000

    if len(frame) <= excel_limit:
        frame.to_excel(path, index=index)
        return

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for start in range(0, len(frame), excel_limit):
            end = min(start + excel_limit, len(frame))
            sheet_name = f"rows_{start + 1}_{end}"
            frame.iloc[start:end].to_excel(writer, sheet_name=sheet_name, index=index)


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


def sigmoid(values):
    values = np.clip(values, -50, 50)
    return 1 / (1 + np.exp(-values))


def compute_pos_weight(labels):
    positive_counts = labels.sum(axis=0)
    negative_counts = len(labels) - positive_counts
    pos_weight = negative_counts / np.maximum(positive_counts, 1)
    pos_weight = np.clip(pos_weight, 1.0, POS_WEIGHT_CLIP_MAX)

    return torch.tensor(pos_weight, dtype=torch.float)


def apply_prediction_rules(
    probabilities,
    label_names,
    threshold,
    use_general_other_fallback=True,
    max_topics_per_review=3,
):
    predictions = (probabilities >= threshold).astype(int)
    fallback_inserted = np.zeros(len(predictions), dtype=bool)

    if use_general_other_fallback and GENERAL_OTHER_LABEL in label_names:
        general_idx = label_names.index(GENERAL_OTHER_LABEL)
        specific_indices = [i for i, label in enumerate(label_names) if label != GENERAL_OTHER_LABEL]

        for row_idx in range(len(predictions)):
            has_specific = predictions[row_idx, specific_indices].sum() > 0

            if has_specific:
                predictions[row_idx, general_idx] = 0
            else:
                if predictions[row_idx, general_idx] == 0:
                    fallback_inserted[row_idx] = True

                predictions[row_idx, general_idx] = 1

    if max_topics_per_review is not None:
        for row_idx in range(len(predictions)):
            selected_indices = np.where(predictions[row_idx] == 1)[0]

            if len(selected_indices) > max_topics_per_review:
                selected_probs = probabilities[row_idx, selected_indices]
                top_order = np.argsort(selected_probs)[::-1][:max_topics_per_review]
                keep_indices = selected_indices[top_order]

                predictions[row_idx, :] = 0
                predictions[row_idx, keep_indices] = 1

    return predictions, fallback_inserted


def evaluate_multilabel_predictions(labels, predictions):
    labels = labels.astype(int)
    predictions = predictions.astype(int)

    metrics = {
        "accuracy_subset_exact_match": accuracy_score(labels, predictions),
        "labelwise_accuracy": float((labels == predictions).mean()),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, predictions, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, predictions, average="micro", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "precision_weighted": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision_samples": precision_score(labels, predictions, average="samples", zero_division=0),
        "recall_samples": recall_score(labels, predictions, average="samples", zero_division=0),
        "f1_samples": f1_score(labels, predictions, average="samples", zero_division=0),
    }

    return {key: float(value) for key, value in metrics.items()}


class ReviewTopicDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels.astype(np.float32) if labels is not None else None

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

        if self.labels is not None:
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


def build_training_args(run_output_dir, learning_rate, num_epochs, seed):
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
    add("eval_accumulation_steps", 50)

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
    df_labeled = pd.read_excel(LABELED_FILE, sheet_name=LABELED_SHEET_NAME, engine="openpyxl")
    required_cols = [TEXT_COL, RELEVANCE_COL] + TOPIC_COLS
    missing_cols = [col for col in required_cols if col not in df_labeled.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in labeled file: {missing_cols}")

    original_rows = len(df_labeled)
    df_labeled = df_labeled[~df_labeled[RELEVANCE_COL].apply(is_relevance_zero)].copy()
    df_labeled = df_labeled.dropna(subset=[TEXT_COL]).copy()
    df_labeled[TEXT_COL] = df_labeled[TEXT_COL].astype(str).str.strip()
    df_labeled = df_labeled[df_labeled[TEXT_COL] != ""].copy()

    df_labeled["topics_original"] = df_labeled.apply(collect_topics, axis=1)
    df_labeled = df_labeled[df_labeled["topics_original"].apply(len) > 0].copy()

    if USE_GENERAL_OTHER_CLEANING:
        df_labeled["topics"] = df_labeled["topics_original"].apply(clean_general_other_rule)
    else:
        df_labeled["topics"] = df_labeled["topics_original"]

    df_labeled = df_labeled[df_labeled["topics"].apply(len) > 0].copy()

    unknown_topics = sorted(
        {
            topic
            for topics in df_labeled["topics"]
            for topic in topics
            if topic not in LABEL_ORDER
        }
    )

    if unknown_topics:
        raise ValueError(f"Unknown topic labels found: {unknown_topics}")

    mlb = MultiLabelBinarizer(classes=LABEL_ORDER)
    y = mlb.fit_transform(df_labeled["topics"]).astype(np.float32)
    label_names = list(mlb.classes_)
    texts = df_labeled[TEXT_COL].tolist()

    print(f"original_labeled_rows: {original_rows}")
    print(f"usable_labeled_reviews: {len(df_labeled)}")

    for i, label in enumerate(label_names):
        print(f"labeled_{label}: {int(y[:, i].sum())}")

    return df_labeled, texts, y, label_names


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


def split_labeled_data(texts, y):
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y.astype(int)))

    X_train = [texts[i] for i in train_idx]
    y_train = y[train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = y[test_idx]

    print(f"evaluation_train_size: {len(X_train)}")
    print(f"evaluation_test_size: {len(X_test)}")

    return X_train, y_train, X_test, y_test


def train_model(texts, labels, tokenizer, data_collator, label_names, seed, output_dir):
    set_seed(seed)

    model = make_model(label_names)
    dataset = ReviewTopicDataset(texts, tokenizer, MAX_LENGTH, labels=labels)
    pos_weight = compute_pos_weight(labels) if USE_CLASS_WEIGHTING else None

    if pos_weight is not None:
        for label, weight in zip(label_names, pos_weight):
            print(f"weight_{label}: {float(weight):.3f}")

    training_args = build_training_args(
        run_output_dir=output_dir,
        learning_rate=BEST_LEARNING_RATE,
        num_epochs=BEST_NUM_EPOCHS,
        seed=seed,
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        pos_weight=pos_weight,
    )

    trainer.train()

    return trainer, model


def evaluate_on_test(trainer, tokenizer, X_test, y_test, label_names):
    test_dataset = ReviewTopicDataset(X_test, tokenizer, MAX_LENGTH, labels=y_test)
    test_output = trainer.predict(test_dataset)

    logits = test_output.predictions
    labels = test_output.label_ids.astype(int)
    probabilities = sigmoid(logits)
    predictions, fallback_inserted = apply_prediction_rules(
        probabilities=probabilities,
        label_names=label_names,
        threshold=BEST_THRESHOLD,
        use_general_other_fallback=USE_GENERAL_OTHER_FALLBACK,
        max_topics_per_review=MAX_TOPICS_PER_REVIEW,
    )

    metrics = evaluate_multilabel_predictions(labels, predictions)

    for key, value in metrics.items():
        print(f"test_{key}: {value:.4f}")

    report_df = pd.DataFrame(
        classification_report(
            labels,
            predictions,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
    ).transpose()

    confusion_df, confusion_matrices = build_confusion_table(labels, predictions, label_names)
    pr_scores_df = build_pr_results(labels, probabilities, label_names)
    roc_scores_df = build_roc_results(labels, probabilities, label_names)
    audit_df = make_test_audit_table(X_test, labels, predictions, probabilities, fallback_inserted, label_names)

    show_frame(report_df)
    show_frame(confusion_df)

    return {
        "metrics": metrics,
        "report": report_df,
        "confusion": confusion_df,
        "confusion_matrices": confusion_matrices,
        "pr_scores": pr_scores_df,
        "roc_scores": roc_scores_df,
        "audit": audit_df,
    }


def build_confusion_table(labels, predictions, label_names):
    confusion_matrices = multilabel_confusion_matrix(labels, predictions)
    rows = []

    for i, label in enumerate(label_names):
        tn, fp, fn, tp = confusion_matrices[i].ravel()
        rows.append({"topic": label, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})

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

        if y_true.sum() == 0:
            print(f"pr_skipped_{label}: no_positive_examples")
            continue

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


def make_test_audit_table(texts, labels, predictions, probabilities, fallback_inserted, label_names):
    audit_df = pd.DataFrame({"review_text": texts})

    for i, label in enumerate(label_names):
        audit_df[f"true_{label}"] = labels[:, i]
        audit_df[f"pred_{label}"] = predictions[:, i]
        audit_df[f"prob_{label}"] = probabilities[:, i]

    audit_df["fallback_general_other_inserted"] = fallback_inserted

    return audit_df


def save_evaluation_outputs(eval_results):
    with pd.ExcelWriter(EVALUATION_FILE, engine="openpyxl") as writer:
        pd.DataFrame([eval_results["metrics"]]).to_excel(writer, sheet_name="metrics", index=False)
        eval_results["report"].to_excel(writer, sheet_name="classification_report")
        eval_results["confusion"].to_excel(writer, sheet_name="confusion", index=False)
        eval_results["pr_scores"].to_excel(writer, sheet_name="pr_scores", index=False)
        eval_results["roc_scores"].to_excel(writer, sheet_name="roc_scores", index=False)
        eval_results["audit"].to_excel(writer, sheet_name="audit", index=False)

    save_confusion_plot(eval_results["confusion_matrices"], LABEL_ORDER)


def load_population_data():
    df_population = pd.read_excel(POPULATION_FILE, sheet_name=POPULATION_SHEET_NAME, engine="openpyxl")

    if TEXT_COL not in df_population.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in population file.")

    original_columns = list(df_population.columns)
    row_id_col = SOURCE_ROW_ID_COL

    if row_id_col in df_population.columns:
        row_id_col = f"_{SOURCE_ROW_ID_COL}_internal"

    df_population = df_population.copy()
    df_population[row_id_col] = np.arange(len(df_population))
    df_population[TEXT_COL] = df_population[TEXT_COL].astype(str).str.strip()

    valid_text_mask = (
        df_population[TEXT_COL].notna()
        & (df_population[TEXT_COL].astype(str).str.strip() != "")
        & (df_population[TEXT_COL].astype(str).str.lower() != "nan")
    )

    valid_df = df_population[valid_text_mask].copy()
    invalid_df = df_population[~valid_text_mask].copy()

    print(f"population_rows_original: {len(df_population)}")
    print(f"population_rows_valid_text: {len(valid_df)}")
    print(f"population_rows_invalid_text: {len(invalid_df)}")

    return df_population, valid_df, invalid_df, original_columns, row_id_col


def predict_population(production_trainer, tokenizer, df_population_valid, label_names):
    population_texts = df_population_valid[TEXT_COL].tolist()
    population_dataset = ReviewTopicDataset(population_texts, tokenizer, MAX_LENGTH, labels=None)
    population_output = production_trainer.predict(population_dataset)

    probabilities = sigmoid(population_output.predictions)
    predictions, fallback_inserted = apply_prediction_rules(
        probabilities=probabilities,
        label_names=label_names,
        threshold=BEST_THRESHOLD,
        use_general_other_fallback=USE_GENERAL_OTHER_FALLBACK,
        max_topics_per_review=MAX_TOPICS_PER_REVIEW,
    )

    print(f"population_probability_rows: {probabilities.shape[0]}")
    print(f"population_probability_labels: {probabilities.shape[1]}")

    return probabilities, predictions, fallback_inserted


def build_population_long_output(
    df_population_valid,
    probabilities,
    predictions,
    fallback_inserted,
    label_names,
    original_columns,
    row_id_col,
):
    long_rows = []
    source_ids = df_population_valid[row_id_col].to_numpy()

    for row_pos in range(len(df_population_valid)):
        selected_indices = np.where(predictions[row_pos] == 1)[0]
        selected_indices = selected_indices[np.argsort(probabilities[row_pos, selected_indices])[::-1]]

        for rank, topic_idx in enumerate(selected_indices, start=1):
            long_rows.append(
                {
                    row_id_col: source_ids[row_pos],
                    "predicted_topic": label_names[topic_idx],
                    "predicted_topic_probability": float(probabilities[row_pos, topic_idx]),
                    "predicted_topic_rank": int(rank),
                    "fallback_general_other_inserted": bool(fallback_inserted[row_pos]),
                    "prediction_threshold": BEST_THRESHOLD,
                }
            )

    topic_long_df = pd.DataFrame(long_rows)
    population_long_df = df_population_valid.merge(topic_long_df, on=row_id_col, how="inner")

    final_columns = [row_id_col] + original_columns + [
        "predicted_topic",
        "predicted_topic_probability",
        "predicted_topic_rank",
        "fallback_general_other_inserted",
        "prediction_threshold",
    ]

    population_long_df = population_long_df[final_columns].copy()

    print(f"population_long_rows: {len(population_long_df)}")
    show_frame(population_long_df.head(10))

    return population_long_df


def build_review_level_audit(df_population_valid, probabilities, predictions, fallback_inserted, label_names):
    review_level_df = df_population_valid.copy()

    for i, label in enumerate(label_names):
        review_level_df[f"prob_{label}"] = probabilities[:, i]
        review_level_df[f"pred_{label}"] = predictions[:, i]

    predicted_topics = []
    topic_counts = []

    for row_pos in range(len(df_population_valid)):
        selected_indices = np.where(predictions[row_pos] == 1)[0]
        selected_indices = selected_indices[np.argsort(probabilities[row_pos, selected_indices])[::-1]]
        topics_for_row = [label_names[i] for i in selected_indices]

        predicted_topics.append("; ".join(topics_for_row))
        topic_counts.append(len(topics_for_row))

    review_level_df["predicted_topics_joined_for_audit_only"] = predicted_topics
    review_level_df["n_predicted_topics"] = topic_counts
    review_level_df["fallback_general_other_inserted"] = fallback_inserted

    return review_level_df


def build_population_summaries(population_long_df, review_level_df, row_id_col):
    topic_distribution_df = (
        population_long_df["predicted_topic"]
        .value_counts()
        .rename_axis("predicted_topic")
        .reset_index(name="topic_assignment_count")
    )

    topic_distribution_df["share_of_topic_assignments"] = (
        topic_distribution_df["topic_assignment_count"]
        / topic_distribution_df["topic_assignment_count"].sum()
    )

    reviews_per_topic_df = (
        population_long_df
        .groupby("predicted_topic")[row_id_col]
        .nunique()
        .reset_index(name="number_of_reviews_with_topic")
    )

    topic_distribution_df = topic_distribution_df.merge(
        reviews_per_topic_df,
        on="predicted_topic",
        how="left",
    )

    topic_count_df = (
        review_level_df["n_predicted_topics"]
        .value_counts()
        .sort_index()
        .rename_axis("number_of_predicted_topics")
        .reset_index(name="number_of_reviews")
    )

    show_frame(topic_distribution_df)
    show_frame(topic_count_df)

    plt.figure(figsize=(8, 5))
    plt.bar(topic_distribution_df["predicted_topic"], topic_distribution_df["topic_assignment_count"])
    plt.xlabel("Predicted topic")
    plt.ylabel("Number of topic assignments")
    plt.title("Population topic distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(POPULATION_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()

    return topic_distribution_df, topic_count_df


def save_population_outputs(population_long_df, review_level_df, topic_distribution_df, topic_count_df):
    save_excel_safely(population_long_df, POPULATION_XLSX_FILE, index=False)
    population_long_df.to_csv(POPULATION_CSV_FILE, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(SUMMARY_FILE, engine="openpyxl") as writer:
        topic_distribution_df.to_excel(writer, sheet_name="topic_distribution", index=False)
        topic_count_df.to_excel(writer, sheet_name="topics_per_review", index=False)
        review_level_df.head(10000).to_excel(writer, sheet_name="audit_sample", index=False)

    try:
        parquet_file = OUTPUT_DIR / f"{RUN_NAME}_population_long.parquet"
        population_long_df.to_parquet(parquet_file, index=False)
    except Exception as exc:
        print(f"parquet_skipped: {exc}")

    print(f"population_long_saved_rows: {len(population_long_df)}")


def main():
    set_seed(RANDOM_SEED)

    print(f"transformers_version: {transformers.__version__}")
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    df_labeled, texts, y, label_names = load_labeled_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    X_train, y_train, X_test, y_test = split_labeled_data(texts, y)

    eval_trainer, eval_model = train_model(
        texts=X_train,
        labels=y_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_names=label_names,
        seed=RANDOM_SEED + 999,
        output_dir=TMP_EVAL_DIR,
    )

    eval_results = evaluate_on_test(eval_trainer, tokenizer, X_test, y_test, label_names)
    save_evaluation_outputs(eval_results)

    del eval_trainer
    del eval_model
    torch.cuda.empty_cache()
    gc.collect()
    shutil.rmtree(TMP_EVAL_DIR, ignore_errors=True)

    # Production training uses all usable labeled rows
    production_trainer, production_model = train_model(
        texts=texts,
        labels=y,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_names=label_names,
        seed=RANDOM_SEED + 2026,
        output_dir=MODEL_DIR,
    )

    production_trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    df_population, df_population_valid, df_population_invalid, original_columns, row_id_col = load_population_data()

    probabilities, predictions, fallback_inserted = predict_population(
        production_trainer=production_trainer,
        tokenizer=tokenizer,
        df_population_valid=df_population_valid,
        label_names=label_names,
    )

    population_long_df = build_population_long_output(
        df_population_valid=df_population_valid,
        probabilities=probabilities,
        predictions=predictions,
        fallback_inserted=fallback_inserted,
        label_names=label_names,
        original_columns=original_columns,
        row_id_col=row_id_col,
    )

    review_level_df = build_review_level_audit(
        df_population_valid=df_population_valid,
        probabilities=probabilities,
        predictions=predictions,
        fallback_inserted=fallback_inserted,
        label_names=label_names,
    )

    topic_distribution_df, topic_count_df = build_population_summaries(
        population_long_df=population_long_df,
        review_level_df=review_level_df,
        row_id_col=row_id_col,
    )

    save_population_outputs(
        population_long_df=population_long_df,
        review_level_df=review_level_df,
        topic_distribution_df=topic_distribution_df,
        topic_count_df=topic_count_df,
    )

    run_config = {
        "labeled_file": str(LABELED_FILE),
        "population_file": str(POPULATION_FILE),
        "text_column": TEXT_COL,
        "model_name": MODEL_NAME,
        "label_order": LABEL_ORDER,
        "best_learning_rate": BEST_LEARNING_RATE,
        "best_num_epochs": BEST_NUM_EPOCHS,
        "best_threshold": BEST_THRESHOLD,
        "use_general_other_fallback": USE_GENERAL_OTHER_FALLBACK,
        "max_topics_per_review": MAX_TOPICS_PER_REVIEW,
        "final_test_size": FINAL_TEST_SIZE,
        "usable_labeled_reviews": int(len(df_labeled)),
        "population_rows_original": int(len(df_population)),
        "population_rows_valid_text": int(len(df_population_valid)),
        "population_rows_invalid_text": int(len(df_population_invalid)),
        "population_long_output_rows": int(len(population_long_df)),
        "use_class_weighting": USE_CLASS_WEIGHTING,
        "pos_weight_clip_max": POS_WEIGHT_CLIP_MAX,
        "max_length": MAX_LENGTH,
        "random_seed": RANDOM_SEED,
    }

    save_json(run_config, CONFIG_FILE)

    del production_trainer
    del production_model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
