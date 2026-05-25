#loading libraries
import re
import numpy as np
import pandas as pd
from pathlib import Path


INPUT_FILE = Path(r"...")
OUTPUT_FILE = INPUT_FILE.with_name("...")

TOTAL_SAMPLE_SIZE = 1000
MIN_PER_BUNDESLAND = 15
YEAR_GROUP_SPAN = 5
RANDOM_STATE = 42


def normalize_colname(col):
    col = str(col).strip().lower()
    return re.sub(r"\s+", " ", col)


def find_column(df_columns, aliases):
    norm_map = {normalize_colname(c): c for c in df_columns}

    for alias in aliases:
        alias_norm = normalize_colname(alias)
        if alias_norm in norm_map:
            return norm_map[alias_norm]

    return None


def extract_year(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_year = numeric.where(numeric.between(1900, 2100))

    dates = pd.to_datetime(series, errors="coerce")
    date_year = dates.dt.year

    return numeric_year.fillna(date_year)


def make_year_groups(year_series: pd.Series, span: int = 5) -> pd.Series:
    min_year = int(year_series.min())
    max_year = int(year_series.max())

    def label_year(year):
        start = min_year + ((int(year) - min_year) // span) * span
        end = min(start + span - 1, max_year)
        return f"{start}-{end}"

    return year_series.astype(int).apply(label_year)


def proportional_allocation(capacity: pd.Series, n_to_allocate: int) -> pd.Series:
    allocation_result = pd.Series(0, index=capacity.index, dtype=int)
    eligible = capacity[capacity > 0].copy()

    if n_to_allocate <= 0 or eligible.empty:
        return allocation_result

    raw_quota = n_to_allocate * eligible / eligible.sum()
    allocation = np.floor(raw_quota).astype(int)
    allocation = pd.Series(
        np.minimum(allocation.values, eligible.values),
        index=eligible.index,
        dtype=int
    )

    remaining = int(n_to_allocate - allocation.sum())

    # give the leftover rows to the largest remainders
    if remaining > 0:
        remainders = (raw_quota - np.floor(raw_quota)).sort_values(ascending=False)

        while remaining > 0:
            changed = False

            for key in remainders.index:
                if allocation[key] < eligible[key]:
                    allocation[key] += 1
                    remaining -= 1
                    changed = True

                    if remaining == 0:
                        break

            if not changed:
                break

    allocation_result.loc[allocation.index] = allocation
    return allocation_result


def balanced_sample_within_state(df_state: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df_state):
        return df_state.copy()

    rng = np.random.default_rng(seed)

    grouped = {
        cell: idx.to_numpy()
        for cell, idx in df_state.groupby(["sentiment_original", "year_group"]).groups.items()
    }

    cell_items = list(grouped.items())
    rng.shuffle(cell_items)

    allocation = {cell: 0 for cell, _ in cell_items}
    remaining = n

    
    while remaining > 0:
        changed = False

        for cell, idx in cell_items:
            if allocation[cell] < len(idx):
                allocation[cell] += 1
                remaining -= 1
                changed = True

            if remaining == 0:
                break

        if not changed:
            break

    sampled_indices = []

    for cell, idx in cell_items:
        take = allocation[cell]

        if take > 0:
            selected = rng.choice(idx, size=take, replace=False)
            sampled_indices.extend(selected.tolist())

    return df_state.loc[sampled_indices].copy()


df = pd.read_excel(INPUT_FILE)

bundesland_col = find_column(df.columns, ["Bundesland", "bundesland"])
sentiment_col = find_column(df.columns, ["sentiment_original", "sentiment original", "sentiment"])
review_date_col = find_column(df.columns, ["Review_Date", "review date", "review_date"])

missing_cols = []

if bundesland_col is None:
    missing_cols.append("Bundesland")

if sentiment_col is None:
    missing_cols.append("sentiment_original")

if review_date_col is None:
    missing_cols.append("Review_Date")

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df.copy()
df["_row_id"] = np.arange(len(df))

df["Bundesland"] = df[bundesland_col].astype("string").str.strip()
df["sentiment_original"] = df[sentiment_col].astype("string").str.strip().str.lower()
df["year_num"] = extract_year(df[review_date_col])

before_drop = len(df)

df = df.dropna(subset=["Bundesland", "sentiment_original", "year_num"]).copy()
df = df[
    (df["Bundesland"].str.len() > 0)
    & (df["sentiment_original"].str.len() > 0)
].copy()

df["year_num"] = df["year_num"].astype(int)
df["year_group"] = make_year_groups(df["year_num"], span=YEAR_GROUP_SPAN)

after_drop = len(df)

if len(df) < TOTAL_SAMPLE_SIZE:
    raise ValueError(
        f"Not enough eligible rows after cleaning: {len(df)} available, "
        f"but {TOTAL_SAMPLE_SIZE} requested."
    )

state_counts = df["Bundesland"].value_counts().sort_index()

# first keep a minimum number of rows for each Bundesland
base_quota = state_counts.clip(upper=MIN_PER_BUNDESLAND)
base_total = int(base_quota.sum())

if base_total > TOTAL_SAMPLE_SIZE:
    raise ValueError(
        f"Minimum allocation exceeds total sample size: {base_total} > {TOTAL_SAMPLE_SIZE}"
    )

remaining_n = TOTAL_SAMPLE_SIZE - base_total
state_capacity = state_counts - base_quota
extra_quota = proportional_allocation(state_capacity, remaining_n)

state_quota = (base_quota + extra_quota).astype(int)

if int(state_quota.sum()) != TOTAL_SAMPLE_SIZE:
    raise ValueError(
        f"Final quotas do not sum to {TOTAL_SAMPLE_SIZE}. "
        f"Current sum: {int(state_quota.sum())}"
    )

sample_parts = []

for i, (state, quota) in enumerate(state_quota.items(), start=1):
    df_state = df[df["Bundesland"] == state].copy()

    sampled_state = balanced_sample_within_state(
        df_state=df_state,
        n=int(quota),
        seed=RANDOM_STATE + i
    )

    sample_parts.append(sampled_state)

sample_df = pd.concat(sample_parts, ignore_index=True)


sample_df = sample_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

if len(sample_df) != TOTAL_SAMPLE_SIZE:
    raise ValueError(
        f"Final sample size mismatch: got {len(sample_df)}, expected {TOTAL_SAMPLE_SIZE}"
    )

state_summary = pd.DataFrame({
    "Bundesland": state_counts.index,
    "population_n": state_counts.values,
    "base_quota": base_quota.reindex(state_counts.index).values,
    "extra_quota": extra_quota.reindex(state_counts.index).values,
    "final_quota": state_quota.reindex(state_counts.index).values
})

sample_state_counts = sample_df["Bundesland"].value_counts().sort_index()
state_summary["sampled_n"] = (
    state_summary["Bundesland"]
    .map(sample_state_counts)
    .fillna(0)
    .astype(int)
)

population_cells = (
    df.groupby(["Bundesland", "sentiment_original", "year_group"])
    .size()
    .reset_index(name="population_n")
)

sample_cells = (
    sample_df.groupby(["Bundesland", "sentiment_original", "year_group"])
    .size()
    .reset_index(name="sample_n")
)

cell_summary = population_cells.merge(
    sample_cells,
    on=["Bundesland", "sentiment_original", "year_group"],
    how="left"
)

cell_summary["sample_n"] = cell_summary["sample_n"].fillna(0).astype(int)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    sample_df.to_excel(writer, sheet_name="sample_1000", index=False)
    state_summary.to_excel(writer, sheet_name="state_summary", index=False)
    cell_summary.to_excel(writer, sheet_name="cell_summary", index=False)