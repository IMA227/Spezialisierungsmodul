#loading libraries
import json
import os
import random
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from IPython.display import display
except Exception:
    display = None


DATA_FILE = Path("...")
SHEET_NAME = 0

TEXT_COL = "review_text"
SENTIMENT_COL = "sentiment_original"
RELEVANCE_COL = "Relevance Flag"

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

OUTPUT_DIR = Path("...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "..._results.xlsx"
CONFIG_FILE = OUTPUT_DIR / "..._config.json"
CONFUSION_PLOT_FILE = OUTPUT_DIR / "..._confusion.png"
PR_PLOT_FILE = OUTPUT_DIR / "..._pr_curves.png"
ROC_PLOT_FILE = OUTPUT_DIR / "..._roc_curves.png"

RANDOM_SEED = 42
TEST_SIZE = 0.20
MAX_LENGTH = 256
BATCH_SIZE = 32

#the three sentiment classes in the labelled data:
LABEL_ORDER = ["negativ", "neutral", "positiv"]
MAP_MIXED_TO_NEUTRAL = False


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


def clean_sentiment_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in ["positive", "negative", "neutral"]:
        return value

    return None


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, engine="openpyxl")

    required_cols = [TEXT_COL, SENTIMENT_COL, RELEVANCE_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in Excel file: {missing_cols}")

    original_rows = len(df)
    df = df[~df[RELEVANCE_COL].apply(is_relevance_zero)].copy()
    df = df.dropna(subset=[TEXT_COL, SENTIMENT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].copy()

    df["sentiment_clean"] = df[SENTIMENT_COL].apply(clean_sentiment_label)
    df = df.dropna(subset=["sentiment_clean"]).copy()

    if MAP_MIXED_TO_NEUTRAL:
        df["sentiment_clean"] = df["sentiment_clean"].replace({"mixed": "neutral"})

    unknown_labels = sorted(set(df["sentiment_clean"]) - set(LABEL_ORDER))

    if unknown_labels:
        raise ValueError(
            f"Unknown sentiment labels found: {unknown_labels}. "
            f"Expected only: {LABEL_ORDER}."
        )

    df = df[df["sentiment_clean"].isin(LABEL_ORDER)].copy()

    texts = df[TEXT_COL].tolist()
    labels = df["sentiment_clean"].tolist()

    label2id = {label: i for i, label in enumerate(LABEL_ORDER)}
    id2label = {i: label for label, i in label2id.items()}
    y = np.array([label2id[label] for label in labels], dtype=int)

    print(f"original_rows: {original_rows}")
    print(f"usable_rows: {len(df)}")
    print(df["sentiment_clean"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int))

    return df, texts, y, label2id, id2label

def normalize_model_label(label):
    label = str(label).strip().lower()

    if label in {"negative", "neutral", "positive"}:
        return label

    return None


def get_model_mapping(config):
    model_id_to_dataset_label = {}

    for model_id, model_label in config.id2label.items():
        model_id_to_dataset_label[int(model_id)] = normalize_model_label(model_label)

    if set(model_id_to_dataset_label.values()) != set(LABEL_ORDER):
        model_id_to_dataset_label = {
            0: "negativ",
            1: "neutral",
            2: "positiv",
        }

    model_idx_for_dataset_label = {
        dataset_label: model_idx
        for model_idx, dataset_label in model_id_to_dataset_label.items()
    }

    missing_labels = [
        label
        for label in LABEL_ORDER
        if label not in model_idx_for_dataset_label
    ]

    if missing_labels:
        raise ValueError(f"Missing mapped model labels: {missing_labels}")

    return model_id_to_dataset_label, model_idx_for_dataset_label


def preprocess_text_for_cardiff(text):
    text = str(text)
    tokens = []

    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            tokens.append("@user")
        elif token.startswith("http"):
            tokens.append("http")
        else:
            tokens.append(token)

    return " ".join(tokens)


def predict_probabilities(
    texts,
    tokenizer,
    model,
    device,
    model_idx_for_dataset_label,
    batch_size=32,
):
    all_probs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="scoring"):
        batch_texts = texts[start:start + batch_size]
        batch_texts = [preprocess_text_for_cardiff(text) for text in batch_texts]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**encoded)
            else:
                outputs = model(**encoded)

        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    reordered_probs = np.zeros((len(texts), len(LABEL_ORDER)), dtype=float)

    for dataset_col_idx, dataset_label in enumerate(LABEL_ORDER):
        model_col_idx = model_idx_for_dataset_label[dataset_label]
        reordered_probs[:, dataset_col_idx] = all_probs[:, model_col_idx]

    return reordered_probs


def evaluate_predictions(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

    return {key: float(value) for key, value in metrics.items()}


def build_class_tables(y_true, y_pred, probabilities):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_ORDER,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    y_true_bin = label_binarize(y_true, classes=list(range(len(LABEL_ORDER))))

    per_class_rows = []
    pr_rows = []
    roc_rows = []

    for class_idx, class_name in enumerate(LABEL_ORDER):
        y_true_class = y_true_bin[:, class_idx]
        y_score_class = probabilities[:, class_idx]
        y_pred_class = (y_pred == class_idx).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true_class,
            y_pred_class,
            labels=[0, 1],
        ).ravel()

        try:
            ap_score = average_precision_score(y_true_class, y_score_class)
        except Exception:
            ap_score = np.nan

        try:
            roc_auc = roc_auc_score(y_true_class, y_score_class)
        except Exception:
            roc_auc = np.nan

        per_class_rows.append(
            {
                "class": class_name,
                "precision": report_dict[class_name]["precision"],
                "recall": report_dict[class_name]["recall"],
                "f1": report_dict[class_name]["f1-score"],
                "support": report_dict[class_name]["support"],
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "average_precision": float(ap_score) if not pd.isna(ap_score) else np.nan,
                "roc_auc": float(roc_auc) if not pd.isna(roc_auc) else np.nan,
            }
        )

        pr_rows.append(
            {
                "class": class_name,
                "average_precision": float(ap_score) if not pd.isna(ap_score) else np.nan,
            }
        )

        roc_rows.append(
            {
                "class": class_name,
                "roc_auc": float(roc_auc) if not pd.isna(roc_auc) else np.nan,
            }
        )

    return report_df, pd.DataFrame(per_class_rows), pd.DataFrame(pr_rows), pd.DataFrame(roc_rows)


def save_confusion_plot(cm):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Final-test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=45, ha="right")
    plt.yticks(range(len(LABEL_ORDER)), LABEL_ORDER)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()


def save_pr_plot(y_true, probabilities):
    y_true_bin = label_binarize(y_true, classes=list(range(len(LABEL_ORDER))))
    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(LABEL_ORDER):
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_bin[:, class_idx],
            probabilities[:, class_idx],
        )
        ap_score = average_precision_score(y_true_bin[:, class_idx], probabilities[:, class_idx])

        plt.plot(recall_curve, precision_curve, label=f"{class_name} AP={ap_score:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final-test PR curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PR_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()


def save_roc_plot(y_true, probabilities):
    y_true_bin = label_binarize(y_true, classes=list(range(len(LABEL_ORDER))))
    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(LABEL_ORDER):
        y_true_class = y_true_bin[:, class_idx]

        if len(np.unique(y_true_class)) < 2:
            print(f"roc_skipped_{class_name}: one_class_only")
            continue

        fpr, tpr, _ = roc_curve(y_true_class, probabilities[:, class_idx])
        auc_score = roc_auc_score(y_true_class, probabilities[:, class_idx])

        plt.plot(fpr, tpr, label=f"{class_name} AUC={auc_score:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final-test ROC curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.show()


def save_results(
    metrics,
    report_df,
    confusion_df,
    per_class_df,
    pr_scores_df,
    roc_scores_df,
    predictions_df,
):
    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="metrics", index=False)
        report_df.to_excel(writer, sheet_name="classification_report")
        confusion_df.to_excel(writer, sheet_name="confusion_matrix")
        per_class_df.to_excel(writer, sheet_name="per_class", index=False)
        pr_scores_df.to_excel(writer, sheet_name="average_precision", index=False)
        roc_scores_df.to_excel(writer, sheet_name="roc_auc", index=False)
        predictions_df.to_excel(writer, sheet_name="predictions", index=False)


def main():
    set_seed(RANDOM_SEED)

    print(f"transformers_version: {transformers.__version__}")
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    _, texts, y, label2id, id2label = load_data()

    all_indices = np.arange(len(texts))
    train_pool_idx, final_test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train_pool = [texts[i] for i in train_pool_idx]
    y_train_pool = y[train_pool_idx]
    X_test = [texts[i] for i in final_test_idx]
    y_test = y[final_test_idx]

    print(f"train_pool_size: {len(X_train_pool)}")
    print(f"final_test_size: {len(X_test)}")
    print(pd.Series([id2label[i] for i in y_train_pool]).value_counts().reindex(LABEL_ORDER).fillna(0).astype(int))
    print(pd.Series([id2label[i] for i in y_test]).value_counts().reindex(LABEL_ORDER).fillna(0).astype(int))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    model_id_to_dataset_label, model_idx_for_dataset_label = get_model_mapping(config)
    print(model_id_to_dataset_label)

    probabilities = predict_probabilities(
        texts=X_test,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_idx_for_dataset_label=model_idx_for_dataset_label,
        batch_size=BATCH_SIZE,
    )

    predictions = np.argmax(probabilities, axis=1)
    metrics = evaluate_predictions(y_test, predictions)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    report_df, per_class_df, pr_scores_df, roc_scores_df = build_class_tables(
        y_true=y_test,
        y_pred=predictions,
        probabilities=probabilities,
    )

    show_frame(report_df)
    show_frame(per_class_df)

    cm = confusion_matrix(y_test, predictions, labels=list(range(len(LABEL_ORDER))))
    confusion_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABEL_ORDER],
        columns=[f"pred_{label}" for label in LABEL_ORDER],
    )

    show_frame(confusion_df)

    save_confusion_plot(cm)
    save_pr_plot(y_test, probabilities)
    save_roc_plot(y_test, probabilities)

    predictions_df = pd.DataFrame(
        {
            "review_text": X_test,
            "true_label": [id2label[i] for i in y_test],
            "pred_label": [id2label[i] for i in predictions],
        }
    )

    for class_idx, class_name in enumerate(LABEL_ORDER):
        predictions_df[f"prob_{class_name}"] = probabilities[:, class_idx]

    save_results(
        metrics=metrics,
        report_df=report_df,
        confusion_df=confusion_df,
        per_class_df=per_class_df,
        pr_scores_df=pr_scores_df,
        roc_scores_df=roc_scores_df,
        predictions_df=predictions_df,
    )

    config_info = {
        "data_file": str(DATA_FILE),
        "sheet_name": SHEET_NAME,
        "text_col": TEXT_COL,
        "sentiment_col": SENTIMENT_COL,
        "relevance_col": RELEVANCE_COL,
        "model_name": MODEL_NAME,
        "label_order": LABEL_ORDER,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "map_mixed_to_neutral": MAP_MIXED_TO_NEUTRAL,
        "method": "zero-training pretrained sentiment inference",
        "relevance_filter": f"{RELEVANCE_COL} != 0",
        "model_id_to_dataset_label": model_id_to_dataset_label,
        "note": (
            "The pretrained CardiffNLP model is used as a zero-training "
            "3-class sentiment baseline."
        ),
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
