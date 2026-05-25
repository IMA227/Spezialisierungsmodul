#Import libraries
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
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
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
SENTIMENT_COL = "sentiment_original"
RELEVANCE_COL = "Relevance Flag"
USE_RELEVANCE_FILTER = True

MODEL_NAME = "deepset/gbert-large"

OUTPUT_DIR = Path("...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "..."
CONFIG_FILE = OUTPUT_DIR / "..."
CONFUSION_PLOT_FILE = OUTPUT_DIR / "..."
PR_PLOT_FILE = OUTPUT_DIR / "..."
ROC_PLOT_FILE = OUTPUT_DIR / "..."

RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.20
N_SPLITS = 5

MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
USE_CLASS_WEIGHTING = True

CLASS_ORDER = ["negativ", "neutral", "positiv"]
METRIC_FOR_SELECTION = "f1_macro"

TRAIN_PARAM_GRID = [
    {"learning_rate": 1e-5, "num_epochs": 4},
    {"learning_rate": 1e-5, "num_epochs": 6},
    {"learning_rate": 2e-5, "num_epochs": 4},
    {"learning_rate": 2e-5, "num_epochs": 6},
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def show_frame(frame, index=False):
    if display is not None:
        display(frame)
    else:
        print(frame.to_string(index=index))


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

def normalize_sentiment_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    sentiment_map = {
        "positive": "positiv",
        "positiv": "positiv",
        "negative": "negativ",
        "negativ": "negativ",
        "neutral": "neutral",
    }

    return sentiment_map.get(value)



def softmax(logits):
    logits = np.asarray(logits)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def evaluate_predictions(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "precision_micro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="micro",
            zero_division=0,
        ),
        "recall_micro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="micro",
            zero_division=0,
        ),
        "f1_micro": f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="micro",
            zero_division=0,
        ),
    }

    return {key: float(value) for key, value in metrics.items()}


def evaluate_predictions_from_ids(y_true_ids, y_pred_ids, id2label):
    y_true_labels = np.array([id2label[int(i)] for i in y_true_ids])
    y_pred_labels = np.array([id2label[int(i)] for i in y_pred_ids])
    return evaluate_predictions(y_true_labels, y_pred_labels)


def format_mean_std(mean_value, std_value):
    if pd.isna(std_value):
        std_value = 0.0

    return f"{mean_value:.4f} +- {std_value:.4f}"


def compute_class_weights(y_ids, num_labels):
    counts = np.bincount(y_ids, minlength=num_labels).astype(float)
    total = counts.sum()
    weights = total / (num_labels * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float)


def unpack_eval_pred(eval_pred):
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    return logits, labels


class ReviewSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class WeightedSingleLabelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        loss = loss_func(logits, labels)

        return (loss, outputs) if return_outputs else loss


def make_metrics_function(id2label):
    def compute_metrics(eval_pred):
        logits, labels = unpack_eval_pred(eval_pred)
        predictions = np.argmax(logits, axis=1)
        return evaluate_predictions_from_ids(labels, predictions, id2label=id2label)

    return compute_metrics


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
    class_weights=None,
    eval_dataset=None,
    compute_metrics=None,
):
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "class_weights": class_weights,
    }

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if compute_metrics is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return WeightedSingleLabelTrainer(**trainer_kwargs)


def load_sentiment_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, engine="openpyxl")

    required_cols = [TEXT_COL, SENTIMENT_COL]
    if USE_RELEVANCE_FILTER:
        required_cols.append(RELEVANCE_COL)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in Excel file: {missing_cols}")

    original_rows = len(df)

    if USE_RELEVANCE_FILTER:
        df = df[~df[RELEVANCE_COL].apply(is_relevance_zero)].copy()

    df = df.dropna(subset=[TEXT_COL, SENTIMENT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].copy()
    df["sentiment_clean"] = df[SENTIMENT_COL].apply(normalize_sentiment_label)

    unknown_rows = int(df["sentiment_clean"].isna().sum())
    df = df[df["sentiment_clean"].isin(CLASS_ORDER)].copy()

    texts = df[TEXT_COL].tolist()
    labels = df["sentiment_clean"].values

    label2id = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    id2label = {idx: label for label, idx in label2id.items()}
    y_ids = np.array([label2id[label] for label in labels], dtype=np.int64)

    print(f"original_rows: {original_rows}")
    print(f"unknown_sentiment_rows: {unknown_rows}")
    print(f"usable_rows: {len(df)}")
    print(df["sentiment_clean"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int))

    return df, texts, y_ids, label2id, id2label


def make_model(label2id, id2label):
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_ORDER),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )


def run_cv_grid_search(X_train_pool, y_train_pool, tokenizer, data_collator, label2id, id2label):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv_rows = []
    oof_store = {}
    total_runs = len(TRAIN_PARAM_GRID) * N_SPLITS
    run_counter = 0

    print(f"cv_training_runs: {total_runs}")

    for param_idx, params in enumerate(TRAIN_PARAM_GRID, start=1):
        train_key = (float(params["learning_rate"]), int(params["num_epochs"]))
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

            print(f"run: {run_counter}/{total_runs}")
            print(f"fold_train_size: {len(X_fold_train)}")
            print(f"fold_validation_size: {len(X_fold_val)}")

            class_weights = compute_class_weights(y_fold_train, len(CLASS_ORDER)) if USE_CLASS_WEIGHTING else None

            if class_weights is not None:
                for idx, weight in enumerate(class_weights):
                    print(f"weight_{id2label[idx]}: {float(weight):.3f}")

            model = make_model(label2id, id2label)
            train_dataset = ReviewSentimentDataset(X_fold_train, y_fold_train, tokenizer, MAX_LENGTH)
            val_dataset = ReviewSentimentDataset(X_fold_val, y_fold_val, tokenizer, MAX_LENGTH)

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
                class_weights=class_weights,
            )

            trainer.train()
            val_output = trainer.predict(val_dataset)

            val_logits = val_output.predictions
            val_labels = val_output.label_ids.astype(int)
            val_probabilities = softmax(val_logits)
            val_predictions = np.argmax(val_probabilities, axis=1)
            fold_metrics = evaluate_predictions_from_ids(val_labels, val_predictions, id2label=id2label)

            row = {
                "fold": fold_idx,
                "learning_rate": params["learning_rate"],
                "num_epochs": params["num_epochs"],
            }
            row.update(fold_metrics)
            cv_rows.append(row)

            oof_store[train_key]["folds"].append(
                {
                    "fold": fold_idx,
                    "texts": X_fold_val,
                    "labels": val_labels,
                    "probabilities": val_probabilities,
                    "predictions": val_predictions,
                }
            )

            del trainer
            del model
            torch.cuda.empty_cache()
            gc.collect()
            shutil.rmtree(run_output_dir, ignore_errors=True)

    return pd.DataFrame(cv_rows), oof_store


def summarize_cv_results(cv_results_df):
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
    ]
    group_cols = ["learning_rate", "num_epochs"]
    records = []

    for group_values, group in cv_results_df.groupby(group_cols):
        learning_rate, num_epochs = group_values
        record = {"learning_rate": float(learning_rate), "num_epochs": int(num_epochs)}

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

    return summary_df, summary_df[group_cols + metric_cols].copy(), metric_cols


def collect_oof_predictions(oof_store, best_params):
    best_key = (best_params["learning_rate"], best_params["num_epochs"])
    best_folds = oof_store[best_key]["folds"]

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

    return (
        oof_texts,
        np.concatenate(oof_labels).astype(int),
        np.vstack(oof_probabilities),
        np.concatenate(oof_predictions).astype(int),
        oof_fold_ids,
        best_folds,
    )


def build_per_class_cv_summary(best_folds, id2label):
    rows = []

    for fold_data in best_folds:
        true_labels = np.array([id2label[int(i)] for i in fold_data["labels"]])
        pred_labels = np.array([id2label[int(i)] for i in fold_data["predictions"]])

        report = classification_report(
            true_labels,
            pred_labels,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            zero_division=0,
            output_dict=True,
        )

        for class_name in CLASS_ORDER:
            rows.append(
                {
                    "fold": fold_data["fold"],
                    "class": class_name,
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                }
            )

    per_class_cv_df = pd.DataFrame(rows)
    summary_rows = []

    for class_name, class_df in per_class_cv_df.groupby("class"):
        record = {"class": class_name, "mean_support": float(class_df["support"].mean())}

        for metric in ["precision", "recall", "f1"]:
            record[metric] = format_mean_std(class_df[metric].mean(), class_df[metric].std(ddof=1))
            record[f"{metric}_mean"] = float(class_df[metric].mean())
            record[f"{metric}_std"] = float(class_df[metric].std(ddof=1))

        summary_rows.append(record)

    return per_class_cv_df, pd.DataFrame(summary_rows)


def train_final_model(X_train_pool, y_train_pool, tokenizer, data_collator, best_params, label2id, id2label):
    final_train_seed = RANDOM_SEED + 999
    set_seed(final_train_seed)

    class_weights = compute_class_weights(y_train_pool, len(CLASS_ORDER)) if USE_CLASS_WEIGHTING else None

    if class_weights is not None:
        for idx, weight in enumerate(class_weights):
            print(f"final_weight_{id2label[idx]}: {float(weight):.3f}")

    model = make_model(label2id, id2label)
    train_dataset = ReviewSentimentDataset(X_train_pool, y_train_pool, tokenizer, MAX_LENGTH)
    final_output_dir = OUTPUT_DIR / "tmp_final"

    training_args = build_training_args(
        run_output_dir=final_output_dir,
        learning_rate=best_params["learning_rate"],
        num_epochs=best_params["num_epochs"],
        seed=final_train_seed,
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    trainer.train()

    return trainer, model, final_train_seed, final_output_dir


def build_per_class_results(true_labels, pred_labels, probabilities):
    y_true_bin = label_binarize(true_labels, classes=CLASS_ORDER)
    report = classification_report(
        true_labels,
        pred_labels,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )

    rows = []

    for class_idx, class_name in enumerate(CLASS_ORDER):
        binary_true = y_true_bin[:, class_idx]
        binary_pred = (pred_labels == class_name).astype(int)
        y_score = probabilities[:, class_idx]
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()

        try:
            ap_score = average_precision_score(binary_true, y_score)
        except Exception:
            ap_score = np.nan

        try:
            roc_auc = roc_auc_score(binary_true, y_score)
        except Exception:
            roc_auc = np.nan

        rows.append(
            {
                "class": class_name,
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1": float(report[class_name]["f1-score"]),
                "support": float(report[class_name]["support"]),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "average_precision": float(ap_score) if not pd.isna(ap_score) else np.nan,
                "roc_auc": float(roc_auc) if not pd.isna(roc_auc) else np.nan,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(report).transpose(), y_true_bin


def save_confusion_plot(confusion_matrix_values):
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_matrix_values)
    plt.title("GBERT sentiment confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45)
    plt.yticks(range(len(CLASS_ORDER)), CLASS_ORDER)

    for row_idx in range(len(CLASS_ORDER)):
        for col_idx in range(len(CLASS_ORDER)):
            plt.text(col_idx, row_idx, str(confusion_matrix_values[row_idx, col_idx]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()


def build_pr_scores(y_true_bin, probabilities):
    rows = []
    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(CLASS_ORDER):
        y_true = y_true_bin[:, class_idx]
        y_score = probabilities[:, class_idx]
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        plt.plot(recall_curve, precision_curve, label=f"{class_name} AP={ap:.3f}")
        rows.append({"class": class_name, "average_precision": float(ap)})

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("GBERT sentiment precision-recall curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PR_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()

    return pd.DataFrame(rows)


def build_roc_scores(y_true_bin, probabilities):
    rows = []
    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(CLASS_ORDER):
        y_true = y_true_bin[:, class_idx]
        y_score = probabilities[:, class_idx]

        if len(np.unique(y_true)) < 2:
            print(f"roc_skipped_{class_name}: one_class_only")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{class_name} AUC={auc_score:.3f}")
        rows.append({"class": class_name, "roc_auc": float(auc_score)})

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("GBERT sentiment ROC curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()

    return pd.DataFrame(rows)


def make_prediction_table(texts, true_ids, pred_ids, probabilities, id2label, fold_ids=None):
    data = {
        "review_text": texts,
        "true_sentiment_id": true_ids,
        "true_sentiment": [id2label[int(i)] for i in true_ids],
        "pred_sentiment_id": pred_ids,
        "pred_sentiment": [id2label[int(i)] for i in pred_ids],
    }

    if fold_ids is not None:
        data = {"fold": fold_ids, **data}

    table = pd.DataFrame(data)

    for class_idx, class_name in enumerate(CLASS_ORDER):
        table[f"prob_{class_name}"] = probabilities[:, class_idx]

    return table


def save_results(
    cv_results_df,
    summary_display_df,
    per_class_cv_df,
    per_class_cv_summary_df,
    oof_metrics,
    oof_predictions_df,
    final_test_metrics,
    final_test_report_df,
    final_confusion_df,
    per_class_results_df,
    pr_scores_df,
    roc_scores_df,
    final_predictions_df,
    experiment_config,
):
    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
        cv_results_df.to_excel(writer, sheet_name="cv_folds", index=False)
        summary_display_df.to_excel(writer, sheet_name="cv_summary", index=False)
        per_class_cv_df.to_excel(writer, sheet_name="cv_classes_folds", index=False)
        per_class_cv_summary_df.to_excel(writer, sheet_name="cv_classes_summary", index=False)
        pd.DataFrame([oof_metrics]).to_excel(writer, sheet_name="oof_metrics", index=False)
        oof_predictions_df.to_excel(writer, sheet_name="oof_predictions", index=False)
        pd.DataFrame([final_test_metrics]).to_excel(writer, sheet_name="final_metrics", index=False)
        final_test_report_df.to_excel(writer, sheet_name="final_report")
        final_confusion_df.to_excel(writer, sheet_name="final_confusion")
        per_class_results_df.to_excel(writer, sheet_name="final_classes", index=False)
        pr_scores_df.to_excel(writer, sheet_name="final_pr_scores", index=False)
        roc_scores_df.to_excel(writer, sheet_name="final_roc_scores", index=False)
        final_predictions_df.to_excel(writer, sheet_name="final_predictions", index=False)

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

    _, texts, y_ids, label2id, id2label = load_sentiment_data()

    X_train_pool, X_final_test, y_train_pool, y_final_test = train_test_split(
        texts,
        y_ids,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_ids,
    )

    print(f"train_pool_size: {len(X_train_pool)}")
    print(f"final_test_size: {len(X_final_test)}")

    train_pool_labels = [id2label[int(i)] for i in y_train_pool]
    final_test_labels = [id2label[int(i)] for i in y_final_test]
    print(pd.Series(train_pool_labels).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int))
    print(pd.Series(final_test_labels).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # The test set stays untouched until the final evaluation.
    cv_results_df, oof_store = run_cv_grid_search(
        X_train_pool=X_train_pool,
        y_train_pool=y_train_pool,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label2id=label2id,
        id2label=id2label,
    )

    summary_df, summary_display_df, metric_cols = summarize_cv_results(cv_results_df)
    show_frame(summary_display_df)

    best_row = summary_df.iloc[0]
    best_params = {
        "learning_rate": float(best_row["learning_rate"]),
        "num_epochs": int(best_row["num_epochs"]),
    }

    print(json.dumps(best_params, indent=2))

    for metric in metric_cols:
        print(f"best_cv_{metric}: {best_row[metric]}")

    oof_texts, oof_labels, oof_probabilities, oof_predictions, oof_fold_ids, best_folds = collect_oof_predictions(
        oof_store,
        best_params,
    )

    oof_metrics = evaluate_predictions_from_ids(oof_labels, oof_predictions, id2label=id2label)

    for metric_name, metric_value in oof_metrics.items():
        print(f"oof_{metric_name}: {metric_value:.4f}")

    per_class_cv_df, per_class_cv_summary_df = build_per_class_cv_summary(best_folds, id2label)
    show_frame(per_class_cv_summary_df[["class", "precision", "recall", "f1", "mean_support"]])

    final_trainer, final_model, final_train_seed, final_output_dir = train_final_model(
        X_train_pool=X_train_pool,
        y_train_pool=y_train_pool,
        tokenizer=tokenizer,
        data_collator=data_collator,
        best_params=best_params,
        label2id=label2id,
        id2label=id2label,
    )

    final_test_dataset = ReviewSentimentDataset(X_final_test, y_final_test, tokenizer, MAX_LENGTH)
    final_test_output = final_trainer.predict(final_test_dataset)

    final_test_logits = final_test_output.predictions
    final_test_ids = final_test_output.label_ids.astype(int)
    final_test_probabilities = softmax(final_test_logits)
    final_test_pred_ids = np.argmax(final_test_probabilities, axis=1)

    final_test_true_labels = np.array([id2label[int(i)] for i in final_test_ids])
    final_test_pred_labels = np.array([id2label[int(i)] for i in final_test_pred_ids])
    final_test_metrics = evaluate_predictions(final_test_true_labels, final_test_pred_labels)

    for metric_name, metric_value in final_test_metrics.items():
        print(f"final_test_{metric_name}: {metric_value:.4f}")

    per_class_results_df, final_test_report_df, y_true_bin = build_per_class_results(
        final_test_true_labels,
        final_test_pred_labels,
        final_test_probabilities,
    )

    show_frame(final_test_report_df)
    show_frame(per_class_results_df)

    final_confusion_matrix = confusion_matrix(
        final_test_true_labels,
        final_test_pred_labels,
        labels=CLASS_ORDER,
    )
    final_confusion_df = pd.DataFrame(
        final_confusion_matrix,
        index=[f"true_{c}" for c in CLASS_ORDER],
        columns=[f"pred_{c}" for c in CLASS_ORDER],
    )

    show_frame(final_confusion_df)
    save_confusion_plot(final_confusion_matrix)

    pr_scores_df = build_pr_scores(y_true_bin, final_test_probabilities)
    roc_scores_df = build_roc_scores(y_true_bin, final_test_probabilities)

    oof_predictions_df = make_prediction_table(
        texts=oof_texts,
        true_ids=oof_labels,
        pred_ids=oof_predictions,
        probabilities=oof_probabilities,
        id2label=id2label,
        fold_ids=oof_fold_ids,
    )

    final_predictions_df = make_prediction_table(
        texts=X_final_test,
        true_ids=final_test_ids,
        pred_ids=final_test_pred_ids,
        probabilities=final_test_probabilities,
        id2label=id2label,
    )

    experiment_config = {
        "design": "GBERT sentiment classification with train_pool CV and independent final holdout test",
        "data_file": str(DATA_FILE),
        "sheet_name": SHEET_NAME,
        "text_column": TEXT_COL,
        "sentiment_column": SENTIMENT_COL,
        "relevance_column": RELEVANCE_COL,
        "use_relevance_filter": USE_RELEVANCE_FILTER,
        "model_name": MODEL_NAME,
        "class_order": CLASS_ORDER,
        "label2id": label2id,
        "id2label": id2label,
        "max_length": MAX_LENGTH,
        "final_test_size": FINAL_TEST_SIZE,
        "n_splits": N_SPLITS,
        "train_param_grid": TRAIN_PARAM_GRID,
        "metric_for_selection": METRIC_FOR_SELECTION,
        "best_params": best_params,
        "use_class_weighting": USE_CLASS_WEIGHTING,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "final_train_seed": final_train_seed,
    }

 
    save_results(
        cv_results_df=cv_results_df,
        summary_display_df=summary_display_df,
        per_class_cv_df=per_class_cv_df,
        per_class_cv_summary_df=per_class_cv_summary_df,
        oof_metrics=oof_metrics,
        oof_predictions_df=oof_predictions_df,
        final_test_metrics=final_test_metrics,
        final_test_report_df=final_test_report_df,
        final_confusion_df=final_confusion_df,
        per_class_results_df=per_class_results_df,
        pr_scores_df=pr_scores_df,
        roc_scores_df=roc_scores_df,
        final_predictions_df=final_predictions_df,
        experiment_config=experiment_config,
    )

    del final_trainer
    del final_model
    torch.cuda.empty_cache()
    gc.collect()
    shutil.rmtree(final_output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()