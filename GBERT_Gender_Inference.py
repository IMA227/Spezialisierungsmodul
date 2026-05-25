#loading libraries
import os
import gc
import json
import random
import inspect
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
LABELED_EXCEL_PATH = '...'
POPULATION_EXCEL_PATH = '...'
LABELED_SHEET_NAME = 0
POPULATION_SHEET_NAME = 0
LABELED_TEXT_COL = 'reviewer_name'
POPULATION_TEXT_COL = 'Reviewer_name'
FALLBACK_LABEL_COLS = ['gender_manual_label', 'manual_label']
RELEVANCE_COL = 'Relevance Flag'
MODEL_NAME = 'deepset/gbert-large'
OUTPUT_DIR = '...'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_LENGTH = 256
RANDOM_SEED = 42
BEST_LEARNING_RATE = 1e-05
BEST_NUM_EPOCHS = 6
FINAL_TEST_SIZE = 0.2
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
WEIGHT_DECAY = 0.01
USE_CLASS_WEIGHTING = True
CLASS_WEIGHT_CLIP_MIN = 0.25
CLASS_WEIGHT_CLIP_MAX = 5.0
LABEL_ORDER = ['Male', 'Female', 'Unknown']

# I keep this order fixed so older reports are still comparable

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(RANDOM_SEED)

def display_or_print(df):
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string())

def is_relevance_zero(value):
    if pd.isna(value):
        return False
    value_str = str(value).strip().lower()
    if value_str in {'0', '0.0'}:
        return True
    try:
        return float(value_str) == 0.0
    except Exception:
        return False

def clean_gender_label(value):
    if pd.isna(value):
        return None

    label = str(value).strip()

    if label == "" or label.lower() in {"nan", "none", "null"}:
        return None

    return label

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def make_predictions(probabilities):
    return np.argmax(probabilities, axis=1).astype(int)

def evaluate_multiclass_predictions(labels, predictions):
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'labelwise_accuracy': accuracy_score(labels, predictions),
        'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'precision_micro': precision_score(labels, predictions, average='micro', zero_division=0),
        'recall_micro': recall_score(labels, predictions, average='micro', zero_division=0),
        'f1_micro': f1_score(labels, predictions, average='micro', zero_division=0),
        'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
    }
    return {key: float(value) for key, value in metrics.items()}

def compute_class_weights(labels, num_labels, label_names):
    counts = np.array([(labels == i).sum() for i in range(num_labels)], dtype=np.float32)
    if np.any(counts == 0):
        missing = [label_names[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f'Some classes are missing in this training split: {missing}. Use a larger labeled dataset.')
    total = len(labels)
    class_weights = total / (num_labels * counts)
    class_weights = np.clip(class_weights, CLASS_WEIGHT_CLIP_MIN, CLASS_WEIGHT_CLIP_MAX)
    return torch.tensor(class_weights, dtype=torch.float)

def build_training_args(run_output_dir, learning_rate, num_epochs, seed):
   

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    training_args_params = training_args_signature.parameters
    kwargs = {'output_dir': run_output_dir}

    def add(name, value):
        if name in training_args_params:
            kwargs[name] = value
    add('num_train_epochs', num_epochs)
    add('per_device_train_batch_size', TRAIN_BATCH_SIZE)
    add('per_device_eval_batch_size', EVAL_BATCH_SIZE)
    add('learning_rate', learning_rate)
    add('weight_decay', WEIGHT_DECAY)
    add('logging_strategy', 'epoch')
    add('save_strategy', 'no')
    add('report_to', 'none')
    add('seed', seed)
    add('data_seed', seed)
    add('eval_accumulation_steps', 50)
    if torch.cuda.is_available():
        bf16_supported = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
        if bf16_supported and 'bf16' in training_args_params:
            kwargs['bf16'] = True
        elif 'fp16' in training_args_params:
            kwargs['fp16'] = True
    return TrainingArguments(**kwargs)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_excel_safely(df, path, index=False):
    # Excel breaks on very large sheets, so I split only when needed.
    excel_max_rows = 1048000
    if len(df) <= excel_max_rows:
        df.to_excel(path, index=index)
    else:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for start in range(0, len(df), excel_max_rows):
                end = min(start + excel_max_rows, len(df))
                sheet_name = f'rows_{start + 1}_{end}'
                df.iloc[start:end].to_excel(writer, sheet_name=sheet_name, index=index)

class GenderClassificationDataset(Dataset):

    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if labels is not None:
            self.labels = labels.astype(np.int64)
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, padding=False)
        item = {key: torch.tensor(value, dtype=torch.long) for key, value in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# This keeps the minority classes from being ignored too easily.
class WeightedSingleLabelTrainer(Trainer):

    def __init__(self, *args, class_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weight is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weight.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def build_trainer(model, training_args, train_dataset, tokenizer, data_collator, class_weight):
    trainer_signature = inspect.signature(Trainer.__init__)
    trainer_params = trainer_signature.parameters
    trainer_kwargs = {'model': model, 'args': training_args, 'train_dataset': train_dataset, 'data_collator': data_collator, 'class_weight': class_weight}
    if 'processing_class' in trainer_params:
        trainer_kwargs['processing_class'] = tokenizer
    elif 'tokenizer' in trainer_params:
        trainer_kwargs['tokenizer'] = tokenizer
    return WeightedSingleLabelTrainer(**trainer_kwargs)
if not os.path.exists(LABELED_EXCEL_PATH):
    raise FileNotFoundError(f'Labeled Excel file not found: {LABELED_EXCEL_PATH}')
df_labeled = pd.read_excel(LABELED_EXCEL_PATH, sheet_name=LABELED_SHEET_NAME, engine='openpyxl')
available_label_cols = [col for col in FALLBACK_LABEL_COLS if col in df_labeled.columns]
if len(available_label_cols) == 0:
    raise ValueError(f'No gender label column found. Expected one of: {FALLBACK_LABEL_COLS}\nAvailable columns are: {list(df_labeled.columns)}')
LABEL_COL = available_label_cols[0]
required_cols = [LABELED_TEXT_COL, LABEL_COL, RELEVANCE_COL]
missing_cols = [col for col in required_cols if col not in df_labeled.columns]
if missing_cols:
    raise ValueError(f'Missing columns in labeled file: {missing_cols}\nAvailable columns are: {list(df_labeled.columns)}')
before_filter = len(df_labeled)
df_labeled = df_labeled[~df_labeled[RELEVANCE_COL].apply(is_relevance_zero)].copy()
after_relevance = len(df_labeled)
df_labeled = df_labeled.dropna(subset=[LABELED_TEXT_COL, LABEL_COL]).copy()
df_labeled[LABELED_TEXT_COL] = df_labeled[LABELED_TEXT_COL].astype(str).str.strip()
df_labeled = df_labeled[df_labeled[LABELED_TEXT_COL] != ''].copy()
df_labeled['gender_label'] = df_labeled[LABEL_COL].apply(clean_gender_label)
df_labeled = df_labeled.dropna(subset=['gender_label']).copy()
unknown_labels = sorted((label for label in df_labeled['gender_label'].unique() if label not in LABEL_ORDER))
if unknown_labels:
    raise ValueError(f'Unknown gender labels found: {unknown_labels}. Expected only: {LABEL_ORDER}')
label_names = LABEL_ORDER
num_labels = len(label_names)
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}
df_labeled['label_id'] = df_labeled['gender_label'].map(label2id)
texts = df_labeled[LABELED_TEXT_COL].tolist()
Y = df_labeled['label_id'].to_numpy(dtype=np.int64)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
final_splitter = StratifiedShuffleSplit(n_splits=1, test_size=FINAL_TEST_SIZE, random_state=RANDOM_SEED)
train_idx, test_idx = next(final_splitter.split(np.zeros(len(Y)), Y))
X_train = [texts[i] for i in train_idx]
y_train = Y[train_idx]
X_test = [texts[i] for i in test_idx]
y_test = Y[test_idx]
eval_seed = RANDOM_SEED + 999
set_seed(eval_seed)
eval_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id, problem_type='single_label_classification')
eval_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=eval_config, ignore_mismatched_sizes=True)
eval_train_dataset = GenderClassificationDataset(X_train, tokenizer, MAX_LENGTH, labels=y_train)
eval_test_dataset = GenderClassificationDataset(X_test, tokenizer, MAX_LENGTH, labels=y_test)
eval_class_weight = compute_class_weights(y_train, num_labels, label_names) if USE_CLASS_WEIGHTING else None
eval_training_args = build_training_args(run_output_dir=os.path.join(OUTPUT_DIR, '...'), learning_rate=BEST_LEARNING_RATE, num_epochs=BEST_NUM_EPOCHS, seed=eval_seed)
eval_trainer = build_trainer(model=eval_model, training_args=eval_training_args, train_dataset=eval_train_dataset, tokenizer=tokenizer, data_collator=data_collator, class_weight=eval_class_weight)
eval_trainer.train()
test_output = eval_trainer.predict(eval_test_dataset)
test_logits = test_output.predictions
test_labels = test_output.label_ids.astype(int)
test_probabilities = softmax(test_logits)
test_predictions = make_predictions(test_probabilities)
test_metrics = evaluate_multiclass_predictions(test_labels, test_predictions)
eval_output_dir = os.path.join(OUTPUT_DIR, '...')
os.makedirs(eval_output_dir, exist_ok=True)
save_json(test_metrics, os.path.join(eval_output_dir, '...'))
test_report_dict = classification_report(test_labels, test_predictions, labels=list(range(num_labels)), target_names=label_names, zero_division=0, output_dict=True)
test_report_df = pd.DataFrame(test_report_dict).transpose()
test_report_df.to_excel(os.path.join(eval_output_dir, '...'))
display_or_print(test_report_df)
test_confusion_matrix = confusion_matrix(test_labels, test_predictions, labels=list(range(num_labels)))
test_confusion_df = pd.DataFrame(test_confusion_matrix, index=[f'true_{label}' for label in label_names], columns=[f'pred_{label}' for label in label_names])
test_confusion_df.to_excel(os.path.join(eval_output_dir, '...'))
display_or_print(test_confusion_df)
plt.figure(figsize=(6, 5))
plt.imshow(test_confusion_matrix)
plt.title('Independent Final-Test Confusion Matrix - Gender')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(range(num_labels), label_names, rotation=45)
plt.yticks(range(num_labels), label_names)
for row_idx in range(num_labels):
    for col_idx in range(num_labels):
        plt.text(col_idx, row_idx, str(test_confusion_matrix[row_idx, col_idx]), ha='center', va='center')
plt.tight_layout()
confusion_plot_path = os.path.join(eval_output_dir, '...')
plt.savefig(confusion_plot_path, dpi=200, bbox_inches='tight')
plt.show()
del eval_trainer
del eval_model
torch.cuda.empty_cache()
gc.collect()
shutil.rmtree(os.path.join(OUTPUT_DIR, '...'), ignore_errors=True)
# After the held-out check, I train once on all usable labeled names.
production_seed = RANDOM_SEED + 2026
set_seed(production_seed)
production_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id, problem_type='single_label_classification')
production_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=production_config, ignore_mismatched_sizes=True)
production_dataset = GenderClassificationDataset(texts, tokenizer, MAX_LENGTH, labels=Y)
production_class_weight = compute_class_weights(Y, num_labels, label_names) if USE_CLASS_WEIGHTING else None
production_model_dir = os.path.join(OUTPUT_DIR, '...')
production_training_args = build_training_args(run_output_dir=production_model_dir, learning_rate=BEST_LEARNING_RATE, num_epochs=BEST_NUM_EPOCHS, seed=production_seed)
production_trainer = build_trainer(model=production_model, training_args=production_training_args, train_dataset=production_dataset, tokenizer=tokenizer, data_collator=data_collator, class_weight=production_class_weight)
production_trainer.train()
production_trainer.save_model(production_model_dir)
tokenizer.save_pretrained(production_model_dir)
production_config_info = {'model_name': MODEL_NAME, 'task': 'single_label_gender_classification', 'input_column_labeled_data': LABELED_TEXT_COL, 'label_column': LABEL_COL, 'trained_on': 'all usable labeled reviewer names after relevance and empty-name filtering', 'number_of_labeled_rows': int(len(df_labeled)), 'label_order': LABEL_ORDER, 'best_learning_rate': BEST_LEARNING_RATE, 'best_num_epochs': BEST_NUM_EPOCHS, 'max_length': MAX_LENGTH, 'use_class_weighting': USE_CLASS_WEIGHTING, 'class_weight_clip_min': CLASS_WEIGHT_CLIP_MIN, 'class_weight_clip_max': CLASS_WEIGHT_CLIP_MAX, 'random_seed': RANDOM_SEED, 'production_seed': production_seed, 'note': 'Predicted labels are reviewer-name-based inferred gender categories. They should be interpreted as aggregate exploratory variables, not certain individual-level demographic facts.'}
save_json(production_config_info, os.path.join(production_model_dir, '...'))
if not os.path.exists(POPULATION_EXCEL_PATH):
    raise FileNotFoundError(f'Population Excel file not found: {POPULATION_EXCEL_PATH}')
df_population = pd.read_excel(POPULATION_EXCEL_PATH, sheet_name=POPULATION_SHEET_NAME, engine='openpyxl')
if POPULATION_TEXT_COL not in df_population.columns:
    raise ValueError(f"Column '{POPULATION_TEXT_COL}' not found in population file. Available columns: {list(df_population.columns)}")
original_population_columns = list(df_population.columns)
# Needed later to restore the original row order.
SOURCE_ROW_ID_COL = 'source_row_id'
if SOURCE_ROW_ID_COL in df_population.columns:
    SOURCE_ROW_ID_COL = '_source_row_id_internal'
df_population = df_population.copy()
df_population[SOURCE_ROW_ID_COL] = np.arange(len(df_population))
raw_names = df_population[POPULATION_TEXT_COL]
valid_name_mask = raw_names.notna() & (raw_names.astype(str).str.strip() != '') & (raw_names.astype(str).str.strip().str.lower() != 'nan')
df_population_valid = df_population[valid_name_mask].copy()
df_population_invalid = df_population[~valid_name_mask].copy()
df_population_valid[POPULATION_TEXT_COL] = df_population_valid[POPULATION_TEXT_COL].astype(str).str.strip()
population_names = df_population_valid[POPULATION_TEXT_COL].tolist()
population_dataset = GenderClassificationDataset(population_names, tokenizer, MAX_LENGTH, labels=None)
population_output = production_trainer.predict(population_dataset)
population_logits = population_output.predictions
population_probabilities = softmax(population_logits)
population_predictions = make_predictions(population_probabilities)
population_confidence = population_probabilities.max(axis=1)
population_pred_labels = [id2label[int(pred_id)] for pred_id in population_predictions]
population_valid_predictions_df = df_population_valid.copy()
population_valid_predictions_df['pred_label_id'] = population_predictions
population_valid_predictions_df['pred_gender'] = population_pred_labels
population_valid_predictions_df['gender_confidence'] = population_confidence
population_valid_predictions_df['inference_status'] = 'predicted'
for i, label in enumerate(label_names):
    population_valid_predictions_df[f'prob_{label}'] = population_probabilities[:, i]
# Missing names stay in the file, but they are clearly marked.
if len(df_population_invalid) > 0:
    population_invalid_predictions_df = df_population_invalid.copy()
    population_invalid_predictions_df['pred_label_id'] = -1
    population_invalid_predictions_df['pred_gender'] = 'Unknown'
    population_invalid_predictions_df['gender_confidence'] = 0.0
    population_invalid_predictions_df['inference_status'] = 'missing_or_empty_reviewer_name'
    for label in label_names:
        population_invalid_predictions_df[f'prob_{label}'] = np.nan
    population_predictions_full_df = pd.concat([population_valid_predictions_df, population_invalid_predictions_df], ignore_index=True)
else:
    population_predictions_full_df = population_valid_predictions_df.copy()
population_predictions_full_df = population_predictions_full_df.sort_values(SOURCE_ROW_ID_COL).reset_index(drop=True)
prediction_columns = ['pred_label_id', 'pred_gender', 'gender_confidence', 'inference_status'] + [f'prob_{label}' for label in label_names]
final_output_columns = [SOURCE_ROW_ID_COL] + original_population_columns + prediction_columns
population_predictions_full_df = population_predictions_full_df[final_output_columns].copy()
display_or_print(population_predictions_full_df.head(10))
gender_distribution_df = population_predictions_full_df['pred_gender'].value_counts(dropna=False).rename_axis('pred_gender').reset_index(name='count')
gender_distribution_df['share'] = gender_distribution_df['count'] / gender_distribution_df['count'].sum()
confidence_summary_df = population_valid_predictions_df.groupby('pred_gender')['gender_confidence'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
display_or_print(gender_distribution_df)
display_or_print(confidence_summary_df)
plt.figure(figsize=(7, 5))
plt.bar(gender_distribution_df['pred_gender'], gender_distribution_df['count'])
plt.xlabel('Predicted gender')
plt.ylabel('Number of rows')
plt.title('Population Predicted Gender Distribution')
plt.tight_layout()
gender_distribution_plot_path = os.path.join(OUTPUT_DIR, '...')
plt.savefig(gender_distribution_plot_path, dpi=200, bbox_inches='tight')
plt.show()

population_predictions_xlsx_path = os.path.join(OUTPUT_DIR, '...')
gender_distribution_xlsx_path = os.path.join(OUTPUT_DIR, '...')
confidence_summary_xlsx_path = os.path.join(OUTPUT_DIR, '...')
save_excel_safely(population_predictions_full_df, population_predictions_xlsx_path, index=False)
gender_distribution_df.to_excel(gender_distribution_xlsx_path, index=False)
confidence_summary_df.to_excel(confidence_summary_xlsx_path, index=False)
run_config = {'task': 'single_label_gender_classification_population_inference', 'labeled_excel_path': LABELED_EXCEL_PATH, 'population_excel_path': POPULATION_EXCEL_PATH, 'labeled_text_column': LABELED_TEXT_COL, 'population_text_column': POPULATION_TEXT_COL, 'label_column': LABEL_COL, 'model_name': MODEL_NAME, 'label_order': LABEL_ORDER, 'best_learning_rate': BEST_LEARNING_RATE, 'best_num_epochs': BEST_NUM_EPOCHS, 'final_test_size_for_evaluation': FINAL_TEST_SIZE, 'usable_labeled_rows': int(len(df_labeled)), 'population_rows_original': int(len(df_population)), 'population_rows_valid_reviewer_name': int(len(df_population_valid)), 'population_rows_missing_or_empty_reviewer_name': int(len(df_population_invalid)), 'population_output_rows': int(len(population_predictions_full_df)), 'max_length': MAX_LENGTH, 'use_class_weighting': USE_CLASS_WEIGHTING, 'class_weight_clip_min': CLASS_WEIGHT_CLIP_MIN, 'class_weight_clip_max': CLASS_WEIGHT_CLIP_MAX, 'random_seed': RANDOM_SEED, 'production_seed': production_seed, 'note': 'Predicted labels are reviewer-name-based inferred gender categories. They should be used only for aggregate exploratory side analysis. The population data is unlabeled, so evaluation metrics are computed only on the labeled final-test split.'}
save_json(run_config, os.path.join(OUTPUT_DIR, '...'))
del production_trainer
del production_model
torch.cuda.empty_cache()
gc.collect()