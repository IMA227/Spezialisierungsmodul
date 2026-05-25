#ä loading libraries
import gc
import json
import random
import inspect
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

warnings.filterwarnings("ignore")

EXCEL_PATH = Path("...")
SHEET_NAME = 0

TEXT_COL = "reviewer_name"
PREFERRED_LABEL_COL = "gender_manual_label"
FALLBACK_LABEL_COLS = ["gender_manual_label", "manual_label"]
RELEVANCE_COL = "Relevance Flag"

MODEL_NAME = "deepset/gbert-large"
RESULTS_FILE = Path("...")

MAX_LENGTH = 256
RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.20
N_SPLITS = 5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01

USE_CLASS_WEIGHTING = True
CLASS_WEIGHT_CLIP_MIN = 0.25
CLASS_WEIGHT_CLIP_MAX = 5.0

LABEL_ORDER = ["Male", "Female", "Unknown"]

TRAIN_PARAM_GRID = [
    {"learning_rate": 1e-5, "num_epochs": 4},
    {"learning_rate": 1e-5, "num_epochs": 6},
    {"learning_rate": 2e-5, "num_epochs": 4},
    {"learning_rate": 2e-5, "num_epochs": 6},
]

METRIC_FOR_SELECTION = "f1_macro"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


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


def clean_gender_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    value_lower = value.lower()

    if value_lower in {"", "nan", "none", "null", "-", "—"}:
        return None

    label_map = {
    "male": "Male",
    "female": "Female",
    "unknown": "Unknown"
}

    return label_map.get(value_lower, value)


if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

available_label_cols = [col for col in FALLBACK_LABEL_COLS if col in df.columns]

if not available_label_cols:
    raise ValueError(
        f"No gender label column found. Expected one of: {FALLBACK_LABEL_COLS}\n"
        f"Available columns are: {list(df.columns)}"
    )

LABEL_COL = available_label_cols[0]

required_cols = [TEXT_COL, LABEL_COL, RELEVANCE_COL]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        f"Missing columns in Excel file: {missing_cols}\n"
        f"Available columns are: {list(df.columns)}"
    )

start_rows = len(df)

df = df[~df[RELEVANCE_COL].apply(is_relevance_zero)].copy()
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""].copy()

df["gender_label"] = df[LABEL_COL].apply(clean_gender_label)
df = df.dropna(subset=["gender_label"]).copy()

unknown_labels = sorted(
    label for label in df["gender_label"].unique()
    if label not in LABEL_ORDER
)

if unknown_labels:
    raise ValueError(
        f"Unknown gender labels found: {unknown_labels}. "
        f"Expected only: {LABEL_ORDER}"
    )

label2id = {label: i for i, label in enumerate(LABEL_ORDER)}
id2label = {i: label for label, i in label2id.items()}

df["label_id"] = df["gender_label"].map(label2id)

texts = df[TEXT_COL].tolist()
Y = df["label_id"].to_numpy(dtype=np.int64)

label_names = LABEL_ORDER
num_labels = len(label_names)

print(f"Usable rows: {len(df)}")
print("Label counts:")
for i, label in enumerate(label_names):
    print(f"{label}: {int((Y == i).sum())}")


splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=FINAL_TEST_SIZE,
    random_state=RANDOM_SEED,
)

train_pool_idx, final_test_idx = next(
    splitter.split(np.zeros(len(Y)), Y)
)

X_train_pool = [texts[i] for i in train_pool_idx]
y_train_pool = Y[train_pool_idx]

X_final_test = [texts[i] for i in final_test_idx]
y_final_test = Y[final_test_idx]

print(f"Train pool: {len(X_train_pool)}")
print(f"Final test: {len(X_final_test)}")


class GenderClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = labels.astype(np.int64)
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

        item = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def unpack_eval_pred(eval_pred):
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    return logits, labels


def make_predictions(probabilities):
    return np.argmax(probabilities, axis=1).astype(int)


def evaluate_predictions(labels, predictions):
    labels = labels.astype(int)
    predictions = predictions.astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "labelwise_accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "precision_micro": precision_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "recall_micro": recall_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(
            labels, predictions, average="micro", zero_division=0
        ),
    }

    return {key: float(value) for key, value in metrics.items()}


def compute_metrics_for_trainer(eval_pred):
    logits, labels = unpack_eval_pred(eval_pred)
    probabilities = softmax(logits)
    predictions = make_predictions(probabilities)
    return evaluate_predictions(labels, predictions)


def format_mean_std(mean_value, std_value):
    if pd.isna(std_value):
        std_value = 0.0

    return f"{mean_value:.4f} +- {std_value:.4f}"


def compute_class_weights(labels):
    counts = np.array(
        [(labels == i).sum() for i in range(num_labels)],
        dtype=np.float32,
    )

    if np.any(counts == 0):
        missing = [label_names[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"Missing class in training split: {missing}")

    total = len(labels)
    weights = total / (num_labels * counts)
    weights = np.clip(weights, CLASS_WEIGHT_CLIP_MIN, CLASS_WEIGHT_CLIP_MAX)

    return torch.tensor(weights, dtype=torch.float)


class WeightedSingleLabelTrainer(Trainer):
    def __init__(self, *args, class_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weight is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weight.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_training_args(
    run_output_dir,
    learning_rate,
    num_epochs,
    seed,
    do_epoch_eval=False,
):
    params = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = {"output_dir": run_output_dir}

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
        bf16_supported = (
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )

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
    class_weight,
    eval_dataset=None,
    compute_metrics=None,
):
    params = inspect.signature(Trainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "class_weight": class_weight,
    }

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if compute_metrics is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics

    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer

    return WeightedSingleLabelTrainer(**trainer_kwargs)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

cv_results_rows = []
oof_store = {}

total_runs = len(TRAIN_PARAM_GRID) * N_SPLITS
run_counter = 0

print(f"CV trainings: {total_runs}")

for param_idx, params in enumerate(TRAIN_PARAM_GRID, start=1):
    learning_rate = params["learning_rate"]
    num_epochs = params["num_epochs"]

    train_key = (float(learning_rate), int(num_epochs))
    oof_store[train_key] = {"folds": []}

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
        cv.split(np.zeros(len(y_train_pool)), y_train_pool),
        start=1,
    ):
        run_counter += 1
        run_seed = RANDOM_SEED + (param_idx * 100) + fold_idx
        set_seed(run_seed)

        X_fold_train = [X_train_pool[i] for i in fold_train_idx]
        y_fold_train = y_train_pool[fold_train_idx]

        X_fold_val = [X_train_pool[i] for i in fold_val_idx]
        y_fold_val = y_train_pool[fold_val_idx]

        class_weight = compute_class_weights(y_fold_train) if USE_CLASS_WEIGHTING else None

        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True,
        )

        train_dataset = GenderClassificationDataset(
            X_fold_train,
            y_fold_train,
            tokenizer,
            MAX_LENGTH,
        )

        val_dataset = GenderClassificationDataset(
            X_fold_val,
            y_fold_val,
            tokenizer,
            MAX_LENGTH,
        )

        with tempfile.TemporaryDirectory() as run_dir:
            training_args = build_training_args(
                run_output_dir=run_dir,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                seed=run_seed,
                do_epoch_eval=False,
            )

            trainer = build_trainer(
                model=model,
                training_args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                class_weight=class_weight,
            )

            trainer.train()
            val_output = trainer.predict(val_dataset)

        val_logits = val_output.predictions
        val_labels = val_output.label_ids.astype(int)
        val_probabilities = softmax(val_logits)
        val_predictions = make_predictions(val_probabilities)

        oof_store[train_key]["folds"].append(
            {
                "fold": fold_idx,
                "texts": X_fold_val,
                "labels": val_labels,
                "probabilities": val_probabilities,
                "predictions": val_predictions,
            }
        )

        fold_metrics = evaluate_predictions(val_labels, val_predictions)

        row = {
            "fold": fold_idx,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        }
        row.update(fold_metrics)
        cv_results_rows.append(row)

        print(
            f"Run {run_counter}/{total_runs} | "
            f"lr={learning_rate} | epochs={num_epochs} | "
            f"fold={fold_idx} | f1_macro={fold_metrics['f1_macro']:.4f}"
        )

        del trainer
        del model

        torch.cuda.empty_cache()
        gc.collect()


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

group_cols = ["learning_rate", "num_epochs"]
summary_records = []

for group_values, group in cv_results_df.groupby(group_cols):
    learning_rate, num_epochs = group_values

    record = {
        "learning_rate": float(learning_rate),
        "num_epochs": int(num_epochs),
    }

    for metric in metric_cols:
        mean_value = group[metric].mean()
        std_value = group[metric].std(ddof=1)

        record[metric] = format_mean_std(mean_value, std_value)
        record[f"{metric}_mean"] = float(mean_value)
        record[f"{metric}_std"] = float(std_value)

    summary_records.append(record)

summary_df = pd.DataFrame(summary_records)

summary_df = summary_df.sort_values(
    [f"{METRIC_FOR_SELECTION}_mean", "f1_micro_mean", "accuracy_mean"],
    ascending=[False, False, False],
).reset_index(drop=True)

summary_display_df = summary_df[group_cols + metric_cols].copy()

best_row = summary_df.iloc[0]

best_params = {
    "learning_rate": float(best_row["learning_rate"]),
    "num_epochs": int(best_row["num_epochs"]),
}

print("Best parameters:")
print(json.dumps(best_params, indent=2))


best_train_key = (
    best_params["learning_rate"],
    best_params["num_epochs"],
)

best_folds = oof_store[best_train_key]["folds"]

oof_texts = []
oof_labels = []
oof_probabilities = []
oof_predictions = []
oof_fold_ids = []

for fold_data in best_folds:
    n_rows = len(fold_data["texts"])

    oof_texts.extend(fold_data["texts"])
    oof_labels.append(fold_data["labels"])
    oof_probabilities.append(fold_data["probabilities"])
    oof_predictions.append(fold_data["predictions"])
    oof_fold_ids.extend([fold_data["fold"]] * n_rows)

oof_labels = np.concatenate(oof_labels).astype(int)
oof_probabilities = np.vstack(oof_probabilities)
oof_predictions = np.concatenate(oof_predictions).astype(int)

oof_metrics = evaluate_predictions(oof_labels, oof_predictions)

print("Best CV out-of-fold metrics:")
for metric_name, metric_value in oof_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


per_gender_cv_rows = []

for fold_data in best_folds:
    fold_id = fold_data["fold"]
    fold_labels = fold_data["labels"].astype(int)
    fold_preds = fold_data["predictions"].astype(int)

    report = classification_report(
        fold_labels,
        fold_preds,
        labels=list(range(num_labels)),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    for label in label_names:
        per_gender_cv_rows.append(
            {
                "fold": fold_id,
                "gender": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
        )

per_gender_cv_df = pd.DataFrame(per_gender_cv_rows)
per_gender_cv_summary_rows = []

for gender, gender_df in per_gender_cv_df.groupby("gender"):
    record = {
        "gender": gender,
        "mean_support": float(gender_df["support"].mean()),
    }

    for metric in ["precision", "recall", "f1"]:
        record[metric] = format_mean_std(
            gender_df[metric].mean(),
            gender_df[metric].std(ddof=1),
        )
        record[f"{metric}_mean"] = float(gender_df[metric].mean())
        record[f"{metric}_std"] = float(gender_df[metric].std(ddof=1))

    per_gender_cv_summary_rows.append(record)

per_gender_cv_summary_df = pd.DataFrame(per_gender_cv_summary_rows)


final_train_seed = RANDOM_SEED + 999
set_seed(final_train_seed)

final_class_weight = compute_class_weights(y_train_pool) if USE_CLASS_WEIGHTING else None

final_config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification",
)

final_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=final_config,
    ignore_mismatched_sizes=True,
)

final_train_dataset = GenderClassificationDataset(
    X_train_pool,
    y_train_pool,
    tokenizer,
    MAX_LENGTH,
)

final_test_dataset = GenderClassificationDataset(
    X_final_test,
    y_final_test,
    tokenizer,
    MAX_LENGTH,
)

with tempfile.TemporaryDirectory() as final_dir:
    final_training_args = build_training_args(
        run_output_dir=final_dir,
        learning_rate=best_params["learning_rate"],
        num_epochs=best_params["num_epochs"],
        seed=final_train_seed,
        do_epoch_eval=False,
    )

    final_trainer = build_trainer(
        model=final_model,
        training_args=final_training_args,
        train_dataset=final_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weight=final_class_weight,
    )

    final_trainer.train()
    final_test_output = final_trainer.predict(final_test_dataset)


final_test_logits = final_test_output.predictions
final_test_labels = final_test_output.label_ids.astype(int)
final_test_probabilities = softmax(final_test_logits)
final_test_predictions = make_predictions(final_test_probabilities)

final_test_metrics = evaluate_predictions(
    final_test_labels,
    final_test_predictions,
)

print("Independent final-test metrics:")
for metric_name, metric_value in final_test_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


final_test_report_dict = classification_report(
    final_test_labels,
    final_test_predictions,
    labels=list(range(num_labels)),
    target_names=label_names,
    zero_division=0,
    output_dict=True,
)

final_test_report_df = pd.DataFrame(final_test_report_dict).transpose()

final_confusion_matrix = confusion_matrix(
    final_test_labels,
    final_test_predictions,
    labels=list(range(num_labels)),
)

final_confusion_df = pd.DataFrame(
    final_confusion_matrix,
    index=[f"true_{label}" for label in label_names],
    columns=[f"pred_{label}" for label in label_names],
)

plt.figure(figsize=(6, 5))
plt.imshow(final_confusion_matrix)
plt.title("Final-Test Confusion Matrix - Gender Classification")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(num_labels), label_names, rotation=45)
plt.yticks(range(num_labels), label_names)

for row_idx in range(num_labels):
    for col_idx in range(num_labels):
        plt.text(
            col_idx,
            row_idx,
            str(final_confusion_matrix[row_idx, col_idx]),
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.show()


final_test_labels_binarized = label_binarize(
    final_test_labels,
    classes=list(range(num_labels)),
)

final_test_predictions_binarized = label_binarize(
    final_test_predictions,
    classes=list(range(num_labels)),
)

final_ovr_confusion_matrices = multilabel_confusion_matrix(
    final_test_labels_binarized,
    final_test_predictions_binarized,
)

final_ovr_confusion_rows = []

for i, label in enumerate(label_names):
    cm = final_ovr_confusion_matrices[i]
    tn, fp, fn, tp = cm.ravel()

    final_ovr_confusion_rows.append(
        {
            "gender": label,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    )

final_ovr_confusion_df = pd.DataFrame(final_ovr_confusion_rows)

fig, axes = plt.subplots(1, num_labels, figsize=(4 * num_labels, 4))

if num_labels == 1:
    axes = [axes]

for i, label in enumerate(label_names):
    ax = axes[i]
    cm = final_ovr_confusion_matrices[i]

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
plt.show()


plt.figure(figsize=(8, 6))
pr_curve_rows = []

for i, label in enumerate(label_names):
    y_true_i = final_test_labels_binarized[:, i]
    y_score_i = final_test_probabilities[:, i]

    precision, recall, _ = precision_recall_curve(y_true_i, y_score_i)
    ap = average_precision_score(y_true_i, y_score_i)

    plt.plot(recall, precision, label=f"{label} AP={ap:.3f}")

    pr_curve_rows.append(
        {
            "gender": label,
            "average_precision": float(ap),
        }
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Final-Test Precision-Recall Curves per Gender")
plt.legend()
plt.grid(True)
plt.show()

pr_scores_df = pd.DataFrame(pr_curve_rows)


plt.figure(figsize=(8, 6))
roc_curve_rows = []

for i, label in enumerate(label_names):
    y_true_i = final_test_labels_binarized[:, i]
    y_score_i = final_test_probabilities[:, i]

    if len(np.unique(y_true_i)) < 2:
        continue

    fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
    auc_score = roc_auc_score(y_true_i, y_score_i)

    plt.plot(fpr, tpr, label=f"{label} AUC={auc_score:.3f}")

    roc_curve_rows.append(
        {
            "gender": label,
            "roc_auc": float(auc_score),
        }
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final-Test ROC Curves per Gender")
plt.legend()
plt.grid(True)
plt.show()

roc_scores_df = pd.DataFrame(roc_curve_rows)


oof_predictions_df = pd.DataFrame(
    {
        "fold": oof_fold_ids,
        "reviewer_name": oof_texts,
        "true_label_id": oof_labels,
        "true_gender": [id2label[int(x)] for x in oof_labels],
        "pred_label_id": oof_predictions,
        "pred_gender": [id2label[int(x)] for x in oof_predictions],
    }
)

for i, label in enumerate(label_names):
    oof_predictions_df[f"prob_{label}"] = oof_probabilities[:, i]

final_test_predictions_df = pd.DataFrame(
    {
        "reviewer_name": X_final_test,
        "true_label_id": final_test_labels,
        "true_gender": [id2label[int(x)] for x in final_test_labels],
        "pred_label_id": final_test_predictions,
        "pred_gender": [id2label[int(x)] for x in final_test_predictions],
    }
)

for i, label in enumerate(label_names):
    final_test_predictions_df[f"prob_{label}"] = final_test_probabilities[:, i]


run_info = {
    "design": "train_pool_cv_plus_independent_final_test",
    "task": "single_label_gender_classification",
    "text_col": TEXT_COL,
    "label_col": LABEL_COL,
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "final_test_size": FINAL_TEST_SIZE,
    "n_splits_cv_on_train_pool": N_SPLITS,
    "label_order": LABEL_ORDER,
    "train_param_grid": TRAIN_PARAM_GRID,
    "metric_for_selection": METRIC_FOR_SELECTION,
    "best_params": best_params,
    "use_class_weighting": USE_CLASS_WEIGHTING,
    "class_weight_clip_min": CLASS_WEIGHT_CLIP_MIN,
    "class_weight_clip_max": CLASS_WEIGHT_CLIP_MAX,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "weight_decay": WEIGHT_DECAY,
    "random_seed": RANDOM_SEED,
    "final_train_seed": final_train_seed,
    "original_rows": int(start_rows),
    "usable_rows": int(len(df)),
    "final_test_metrics": final_test_metrics,
    "oof_metrics": oof_metrics,
}

with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
    cv_results_df.to_excel(writer, sheet_name="cv_folds", index=False)
    summary_display_df.to_excel(writer, sheet_name="cv_summary", index=False)
    per_gender_cv_summary_df.to_excel(writer, sheet_name="cv_gender_summary", index=False)
    oof_predictions_df.to_excel(writer, sheet_name="oof_predictions", index=False)
    final_test_report_df.to_excel(writer, sheet_name="final_report")
    final_confusion_df.to_excel(writer, sheet_name="final_confusion")
    final_ovr_confusion_df.to_excel(writer, sheet_name="final_ovr_confusion", index=False)
    pr_scores_df.to_excel(writer, sheet_name="final_pr_scores", index=False)
    roc_scores_df.to_excel(writer, sheet_name="final_roc_scores", index=False)
    final_test_predictions_df.to_excel(writer, sheet_name="final_predictions", index=False)

    pd.DataFrame(
        [{"key": key, "value": json.dumps(value, ensure_ascii=False)}
         for key, value in run_info.items()]
    ).to_excel(writer, sheet_name="run_info", index=False)


del final_trainer
del final_model

torch.cuda.empty_cache()
gc.collect()

