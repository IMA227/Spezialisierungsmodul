#loading libraries:
import json
import math
import random
import re
import subprocess
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")

try:
    from IPython.display import display
except Exception:
    display = None

try:
    from textblob_de import TextBlobDE
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "textblob-de"])
    from textblob_de import TextBlobDE


EXCEL_PATH = Path("...")
SHEET_NAME = 0

TEXT_COL = "review_text"
SENTIMENT_COL = "sentiment_original"
RELEVANCE_COL = "Relevance Flag"
USE_RELEVANCE_FILTER = True

OUTPUT_DIR = Path("...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STEM = "..."

LEXICON_DIR = Path("...")
LEXICON_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_results.xlsx"
CONFIG_FILE = OUTPUT_DIR / f"{OUTPUT_STEM}_config.json"

RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.20
N_SPLITS = 5

CLASS_ORDER = ["negativ", "neutral", "positiv"]
METRIC_FOR_SELECTION = "f1_macro"

SENTIWS_POS_PATH = None
SENTIWS_NEG_PATH = None
GERMAN_POLARITY_CLUES_ZIP_PATH = None

SENTIWS_POS_URL = (
    "https://git.informatik.uni-leipzig.de/vp38kaqy/figurennetzwerk/-/raw/"
    "6a1620620a1003abe126a9ba6f7c2c2885c202b0/"
    "Senti%20Net%201.0/SentiWS_v2.0/SentiWS_v2.0_Positive.txt"
)

SENTIWS_NEG_URL = (
    "https://git.informatik.uni-leipzig.de/vp38kaqy/figurennetzwerk/-/raw/"
    "6a1620620a1003abe126a9ba6f7c2c2885c202b0/"
    "Senti%20Net%201.0/SentiWS_v2.0/SentiWS_v2.0_Negative.txt"
)

GERMAN_POLARITY_CLUES_URL = (
    "https://www.ulliwaltinger.de/wp-content/uploads/2025/01/"
    "GermanPolarityClues-2012.zip"
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def show_frame(frame, index=False):
    if display is not None:
        display(frame)
    else:
        print(frame.to_string(index=index))


def download_file(url, output_path):
    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()

    output_path.write_bytes(content)

    if output_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {output_path}")

    return output_path


def read_text_lines(path):
    path = Path(path)

    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def normalize_word(word):
    if word is None:
        return None

    word = str(word).strip().lower()
    word = word.replace("’", "'").replace("`", "'")

    if "|" in word:
        word = word.split("|")[0]

    word = re.sub(r"^[^\wäöüß]+|[^\wäöüß]+$", "", word, flags=re.IGNORECASE)

    if len(word) < 2:
        return None

    return word


def tokenize_german(text):
    return re.findall(r"[a-zäöüß]+", str(text).lower(), flags=re.IGNORECASE)


def parse_float_maybe(value):
    if value is None:
        return None

    value = str(value).replace(",", ".")
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", value)

    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


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


def format_mean_std(mean_value, std_value):
    if pd.isna(std_value):
        std_value = 0.0

    return f"{mean_value:.4f} +- {std_value:.4f}"


def load_clean_data():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

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

    print(f"original_rows: {original_rows}")
    print(f"removed_unknown_sentiment: {unknown_rows}")
    print(f"usable_rows: {len(df)}")
    print(df["sentiment_clean"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int))

    return df, df[TEXT_COL].tolist(), df["sentiment_clean"].values


def ensure_sentiws_files():
    if SENTIWS_POS_PATH is not None and SENTIWS_NEG_PATH is not None:
        return Path(SENTIWS_POS_PATH), Path(SENTIWS_NEG_PATH)

    pos_candidates = list(LEXICON_DIR.rglob("*SentiWS*Positive*.txt"))
    neg_candidates = list(LEXICON_DIR.rglob("*SentiWS*Negative*.txt"))

    if pos_candidates and neg_candidates:
        return pos_candidates[0], neg_candidates[0]

    pos_path = LEXICON_DIR / "SentiWS_v2.0_Positive.txt"
    neg_path = LEXICON_DIR / "SentiWS_v2.0_Negative.txt"

    try:
        download_file(SENTIWS_POS_URL, pos_path)
        download_file(SENTIWS_NEG_URL, neg_path)
    except Exception as exc:
        raise FileNotFoundError("SentiWS files were not found and could not be downloaded.") from exc

    return pos_path, neg_path


def load_sentiws_lexicon():
    pos_path, neg_path = ensure_sentiws_files()
    lexicon = {}

    for path in [pos_path, neg_path]:
        for line in read_text_lines(path):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            score = parse_float_maybe(parts[1])

            if score is None:
                continue

            file_name = path.name.lower()

            if "negative" in file_name and score > 0:
                score = -score

            if "positive" in file_name and score < 0:
                score = abs(score)

            lemma = normalize_word(parts[0])

            if lemma is not None:
                lexicon[lemma] = score

            if len(parts) >= 3:
                for inflection in parts[2].split(","):
                    word = normalize_word(inflection)

                    if word is not None:
                        lexicon[word] = score

    if not lexicon:
        raise ValueError("SentiWS lexicon is empty after parsing.")

    return lexicon


def ensure_gpc_files():
    if GERMAN_POLARITY_CLUES_ZIP_PATH is not None:
        zip_path = Path(GERMAN_POLARITY_CLUES_ZIP_PATH)
    else:
        zip_path = LEXICON_DIR / "GermanPolarityClues-2012.zip"

    existing_tsv_files = list(LEXICON_DIR.rglob("GermanPolarityClues*.tsv"))

    if existing_tsv_files:
        return existing_tsv_files

    try:
        download_file(GERMAN_POLARITY_CLUES_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(LEXICON_DIR)
    except Exception as exc:
        raise FileNotFoundError("GermanPolarityClues files were not found and could not be downloaded.") from exc

    tsv_files = list(LEXICON_DIR.rglob("GermanPolarityClues*.tsv"))

    if not tsv_files:
        raise FileNotFoundError("No GermanPolarityClues TSV files found after extraction.")

    return tsv_files


def load_german_polarity_clues_lexicon():
    tsv_files = ensure_gpc_files()
    lexicon = {}

    for path in tsv_files:
        file_name = path.name.lower()

        if "positive" in file_name:
            file_sign = 1.0
        elif "negative" in file_name:
            file_sign = -1.0
        elif "neutral" in file_name:
            file_sign = 0.0
        else:
            file_sign = None

        for line in read_text_lines(path):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if not parts:
                continue

            feature = normalize_word(parts[0])
            lemma = normalize_word(parts[1]) if len(parts) >= 2 else None
            sign = file_sign

            for part in parts:
                part_lower = str(part).lower()

                if "positive" in part_lower:
                    sign = 1.0
                elif "negative" in part_lower:
                    sign = -1.0
                elif "neutral" in part_lower:
                    sign = 0.0

            if sign is None or sign == 0.0:
                continue

            numeric_values = []

            for part in parts:
                for match in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(part)):
                    try:
                        numeric_values.append(abs(float(match)))
                    except Exception:
                        pass

            weight = max(numeric_values) if numeric_values else 1.0
            weight = weight if weight != 0 else 1.0
            score = sign * weight

            for word in [feature, lemma]:
                if word is not None:
                    lexicon[word] = score

    if not lexicon:
        raise ValueError("GermanPolarityClues lexicon is empty after parsing.")

    return lexicon


def score_text_with_lexicon(text, lexicon):
    tokens = tokenize_german(text)
    total_score = 0.0
    matched_count = 0

    for token in tokens:
        if token in lexicon:
            total_score += lexicon[token]
            matched_count += 1

    if matched_count == 0:
        return 0.0

    return total_score / math.sqrt(matched_count)


def score_textblob_de(text):
    try:
        blob = TextBlobDE(str(text))
        return float(blob.sentiment.polarity)
    except Exception:
        return 0.0


def score_texts_lexicon(texts, lexicon, method_name):
    scores = []

    for idx, text in enumerate(texts, start=1):
        if idx % 500 == 0:
            print(f"{method_name}_scored: {idx}/{len(texts)}")

        scores.append(score_text_with_lexicon(text, lexicon))

    return np.asarray(scores, dtype=float)


def score_texts_textblob(texts):
    scores = []

    for idx, text in enumerate(texts, start=1):
        if idx % 500 == 0:
            print(f"TextBlobDE_scored: {idx}/{len(texts)}")

        scores.append(score_textblob_de(text))

    return np.asarray(scores, dtype=float)


def polarity_scores_to_labels(scores, positive_threshold, negative_threshold):
    predictions = np.array(["neutral"] * len(scores), dtype=object)
    predictions[scores > positive_threshold] = "positiv"
    predictions[scores < -negative_threshold] = "negativ"

    return predictions


def polarity_scores_to_ovr_scores(scores):
    scores = np.asarray(scores, dtype=float)
    return np.column_stack([-scores, -np.abs(scores), scores])


def build_threshold_grid(scores):
    scores = np.asarray(scores, dtype=float)
    abs_scores = np.abs(scores[np.isfinite(scores)])

    base_grid = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    nonzero_scores = abs_scores[abs_scores > 0]

    if len(nonzero_scores) > 0:
        quantile_grid = np.quantile(
            nonzero_scores,
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        ).tolist()
    else:
        quantile_grid = []

    return sorted(set(round(float(x), 6) for x in base_grid + quantile_grid))


def evaluate_predictions(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, labels=CLASS_ORDER, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, labels=CLASS_ORDER, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="micro", zero_division=0),
    }

    return {key: float(value) for key, value in metrics.items()}


def load_methods(X_train_pool, X_final_test):
    sentiws_lexicon = load_sentiws_lexicon()
    gpc_lexicon = load_german_polarity_clues_lexicon()

    print(f"sentiws_entries: {len(sentiws_lexicon)}")
    print(f"german_polarity_clues_entries: {len(gpc_lexicon)}")

    return {
        "SentiWS": {
            "train_scores": score_texts_lexicon(X_train_pool, sentiws_lexicon, "SentiWS_train"),
            "test_scores": score_texts_lexicon(X_final_test, sentiws_lexicon, "SentiWS_test"),
            "entries": len(sentiws_lexicon),
        },
        "GermanPolarityClues": {
            "train_scores": score_texts_lexicon(X_train_pool, gpc_lexicon, "GPC_train"),
            "test_scores": score_texts_lexicon(X_final_test, gpc_lexicon, "GPC_test"),
            "entries": len(gpc_lexicon),
        },
        "TextBlobDE": {
            "train_scores": score_texts_textblob(X_train_pool),
            "test_scores": score_texts_textblob(X_final_test),
            "entries": None,
        },
    }


def tune_thresholds(methods, y_train_pool):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    best_configs = {}

    for method_name, method_data in methods.items():
        train_scores = method_data["train_scores"]
        threshold_grid = build_threshold_grid(train_scores)
        print(f"{method_name}_threshold_candidates: {len(threshold_grid)}")

        for positive_threshold in threshold_grid:
            for negative_threshold in threshold_grid:
                for fold_idx, (_, val_idx) in enumerate(
                    cv.split(np.zeros(len(y_train_pool)), y_train_pool),
                    start=1,
                ):
                    val_predictions = polarity_scores_to_labels(
                        train_scores[val_idx],
                        positive_threshold=positive_threshold,
                        negative_threshold=negative_threshold,
                    )

                    metrics = evaluate_predictions(y_train_pool[val_idx], val_predictions)
                    row = {
                        "method": method_name,
                        "fold": fold_idx,
                        "positive_threshold": positive_threshold,
                        "negative_threshold": negative_threshold,
                    }
                    row.update(metrics)
                    rows.append(row)

        print(f"{method_name}_threshold_tuning_done: True")

    cv_results_df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
    ]
    group_cols = ["method", "positive_threshold", "negative_threshold"]
    summary_rows = []

    for group_values, group in cv_results_df.groupby(group_cols):
        method_name, positive_threshold, negative_threshold = group_values
        record = {
            "method": method_name,
            "positive_threshold": float(positive_threshold),
            "negative_threshold": float(negative_threshold),
        }

        for metric in metric_cols:
            mean_value = group[metric].mean()
            std_value = group[metric].std(ddof=1)
            record[metric] = format_mean_std(mean_value, std_value)
            record[f"{metric}_mean"] = float(mean_value)
            record[f"{metric}_std"] = float(std_value)

        summary_rows.append(record)

    cv_summary_df = pd.DataFrame(summary_rows).sort_values(
        [f"{METRIC_FOR_SELECTION}_mean", "f1_micro_mean", "accuracy_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    for method_name in methods:
        best_row = cv_summary_df[cv_summary_df["method"] == method_name].iloc[0]
        best_configs[method_name] = {
            "positive_threshold": float(best_row["positive_threshold"]),
            "negative_threshold": float(best_row["negative_threshold"]),
        }

    return cv_results_df, cv_summary_df, best_configs


def evaluate_final_test(methods, best_configs, X_final_test, y_final_test):
    overall_rows = []
    per_class_rows = []
    classification_reports = {}
    confusion_matrices = {}
    prediction_tables = {}
    pr_rows = []
    roc_rows = []

    for method_name, method_data in methods.items():
        config = best_configs[method_name]
        test_scores = method_data["test_scores"]
        test_predictions = polarity_scores_to_labels(
            test_scores,
            positive_threshold=config["positive_threshold"],
            negative_threshold=config["negative_threshold"],
        )

        overall_metrics = evaluate_predictions(y_final_test, test_predictions)
        overall_row = {"method": method_name, **config}
        overall_row.update(overall_metrics)
        overall_rows.append(overall_row)

        print(f"method: {method_name}")
        for metric_name, metric_value in overall_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        report_dict = classification_report(
            y_final_test,
            test_predictions,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            zero_division=0,
            output_dict=True,
        )

        report_df = pd.DataFrame(report_dict).transpose()
        classification_reports[method_name] = report_df

        cm = confusion_matrix(y_final_test, test_predictions, labels=CLASS_ORDER)
        confusion_matrices[method_name] = cm
        y_true_bin = label_binarize(y_final_test, classes=CLASS_ORDER)
        ovr_scores = polarity_scores_to_ovr_scores(test_scores)

        for class_idx, class_name in enumerate(CLASS_ORDER):
            binary_true = y_true_bin[:, class_idx]
            binary_pred = (test_predictions == class_name).astype(int)
            tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()
            y_score = ovr_scores[:, class_idx]

            try:
                ap_score = average_precision_score(binary_true, y_score)
            except Exception:
                ap_score = np.nan

            try:
                roc_auc = roc_auc_score(binary_true, y_score)
            except Exception:
                roc_auc = np.nan

            per_class_rows.append(
                {
                    "method": method_name,
                    "class": class_name,
                    "precision": float(report_dict[class_name]["precision"]),
                    "recall": float(report_dict[class_name]["recall"]),
                    "f1": float(report_dict[class_name]["f1-score"]),
                    "support": float(report_dict[class_name]["support"]),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                    "average_precision": float(ap_score) if not pd.isna(ap_score) else np.nan,
                    "roc_auc": float(roc_auc) if not pd.isna(roc_auc) else np.nan,
                }
            )

        prediction_tables[method_name] = pd.DataFrame(
            {
                "review_text": X_final_test,
                "true_sentiment": y_final_test,
                "pred_sentiment": test_predictions,
                "polarity_score": test_scores,
            }
        )

    overall_results_df = pd.DataFrame(overall_rows).sort_values(
        ["f1_macro", "f1_micro", "accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    per_class_results_df = pd.DataFrame(per_class_rows)

    return (
        overall_results_df,
        per_class_results_df,
        classification_reports,
        confusion_matrices,
        prediction_tables,
        pr_rows,
        roc_rows,
    )


def plot_confusion_matrices(confusion_matrices):
    for method_name, cm in confusion_matrices.items():
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"{method_name} - final-test confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45)
        plt.yticks(range(len(CLASS_ORDER)), CLASS_ORDER)

        for row_idx in range(len(CLASS_ORDER)):
            for col_idx in range(len(CLASS_ORDER)):
                plt.text(col_idx, row_idx, str(cm[row_idx, col_idx]), ha="center", va="center")

        plt.tight_layout()
        plot_path = OUTPUT_DIR / f"{OUTPUT_STEM}_{method_name}_confusion.png".replace(" ", "_")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.show()


def plot_pr_curves(methods, y_final_test):
    rows = []
    y_true_bin = label_binarize(y_final_test, classes=CLASS_ORDER)

    for method_name, method_data in methods.items():
        ovr_scores = polarity_scores_to_ovr_scores(method_data["test_scores"])
        plt.figure(figsize=(8, 6))

        for class_idx, class_name in enumerate(CLASS_ORDER):
            y_true_class = y_true_bin[:, class_idx]
            y_score_class = ovr_scores[:, class_idx]
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_class, y_score_class)
            ap = average_precision_score(y_true_class, y_score_class)

            plt.plot(recall_curve, precision_curve, label=f"{class_name} AP={ap:.3f}")
            rows.append({"method": method_name, "class": class_name, "average_precision": float(ap)})

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{method_name} - final-test precision-recall curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / f"{OUTPUT_STEM}_{method_name}_pr.png".replace(" ", "_")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.show()

    return pd.DataFrame(rows)


def plot_roc_curves(methods, y_final_test):
    rows = []
    y_true_bin = label_binarize(y_final_test, classes=CLASS_ORDER)

    for method_name, method_data in methods.items():
        ovr_scores = polarity_scores_to_ovr_scores(method_data["test_scores"])
        plt.figure(figsize=(8, 6))

        for class_idx, class_name in enumerate(CLASS_ORDER):
            y_true_class = y_true_bin[:, class_idx]
            y_score_class = ovr_scores[:, class_idx]

            if len(np.unique(y_true_class)) < 2:
                print(f"roc_skipped_{method_name}_{class_name}: one_class_only")
                continue

            fpr, tpr, _ = roc_curve(y_true_class, y_score_class)
            auc_score = roc_auc_score(y_true_class, y_score_class)

            plt.plot(fpr, tpr, label=f"{class_name} AUC={auc_score:.3f}")
            rows.append({"method": method_name, "class": class_name, "roc_auc": float(auc_score)})

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{method_name} - final-test ROC curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / f"{OUTPUT_STEM}_{method_name}_roc.png".replace(" ", "_")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.show()

    return pd.DataFrame(rows)


def safe_sheet_name(name):
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", str(name))
    return cleaned[:31]


def save_outputs(
    cv_results_df,
    cv_summary_df,
    best_configs,
    overall_results_df,
    per_class_results_df,
    pr_scores_df,
    roc_scores_df,
    classification_reports,
    prediction_tables,
    experiment_config,
):
    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
        cv_results_df.to_excel(writer, sheet_name="cv_folds", index=False)
        cv_summary_df.to_excel(writer, sheet_name="cv_summary", index=False)
        overall_results_df.to_excel(writer, sheet_name="final_overall", index=False)
        per_class_results_df.to_excel(writer, sheet_name="final_per_class", index=False)
        pr_scores_df.to_excel(writer, sheet_name="pr_scores", index=False)
        roc_scores_df.to_excel(writer, sheet_name="roc_scores", index=False)
        pd.DataFrame(best_configs).transpose().to_excel(writer, sheet_name="best_thresholds")

        for method_name, report_df in classification_reports.items():
            report_df.to_excel(writer, sheet_name=safe_sheet_name(f"report_{method_name}"))

        for method_name, prediction_df in prediction_tables.items():
            prediction_df.to_excel(writer, sheet_name=safe_sheet_name(f"pred_{method_name}"), index=False)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)

    print(f"results_file: {RESULTS_FILE}")
    print(f"config_file: {CONFIG_FILE}")


def main():
    set_seed(RANDOM_SEED)

    df, texts, y = load_clean_data()

    X_train_pool, X_final_test, y_train_pool, y_final_test = train_test_split(
        texts,
        y,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"train_pool_size: {len(X_train_pool)}")
    print(f"final_test_size: {len(X_final_test)}")
    print(pd.Series(y_train_pool).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int))
    print(pd.Series(y_final_test).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int))

    # The test split (untoutched till end)
    methods = load_methods(X_train_pool, X_final_test)
    cv_results_df, cv_summary_df, best_configs = tune_thresholds(methods, y_train_pool)

    show_frame(
        cv_summary_df[
            [
                "method",
                "positive_threshold",
                "negative_threshold",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_micro",
                "recall_micro",
                "f1_micro",
            ]
        ]
    )

    print(json.dumps(best_configs, indent=2, ensure_ascii=False))

    (
        overall_results_df,
        per_class_results_df,
        classification_reports,
        confusion_matrices,
        prediction_tables,
        _,
        _,
    ) = evaluate_final_test(methods, best_configs, X_final_test, y_final_test)

    show_frame(overall_results_df)
    show_frame(per_class_results_df)

    plot_confusion_matrices(confusion_matrices)
    pr_scores_df = plot_pr_curves(methods, y_final_test)
    roc_scores_df = plot_roc_curves(methods, y_final_test)

    experiment_config = {
        "design": "lexicon sentiment baselines with train-pool threshold tuning and independent final-test evaluation",
        "excel_path": str(EXCEL_PATH),
        "sheet_name": SHEET_NAME,
        "text_column": TEXT_COL,
        "sentiment_column": SENTIMENT_COL,
        "relevance_column": RELEVANCE_COL,
        "use_relevance_filter": USE_RELEVANCE_FILTER,
        "class_order": CLASS_ORDER,
        "final_test_size": FINAL_TEST_SIZE,
        "n_splits": N_SPLITS,
        "metric_for_selection": METRIC_FOR_SELECTION,
        "methods": list(methods.keys()),
        "best_thresholds": best_configs,
        "random_seed": RANDOM_SEED,
        "usable_rows": int(len(df)),
        "sentiws_entries": methods["SentiWS"]["entries"],
        "german_polarity_clues_entries": methods["GermanPolarityClues"]["entries"],
    }

  
    save_outputs(
        cv_results_df=cv_results_df,
        cv_summary_df=cv_summary_df,
        best_configs=best_configs,
        overall_results_df=overall_results_df,
        per_class_results_df=per_class_results_df,
        pr_scores_df=pr_scores_df,
        roc_scores_df=roc_scores_df,
        classification_reports=classification_reports,
        prediction_tables=prediction_tables,
        experiment_config=experiment_config,
    )


if __name__ == "__main__":
    main()