# loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path("...")
MIN_REVIEWS = 10

out_dir = DATA_PATH.parent


def load_csv(path):
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    separators = [";", ",", "\t"]
    required = {"Restaurant Name", "Address", "Sentiment", "ID"}

    last_error = None

    for enc in encodings:
        for sep in separators:
            try:
                data = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    engine="python"
                )

                if required.issubset(data.columns):
                    return data

            except Exception as err:
                last_error = err

    raise ValueError(f"Could not load the CSV correctly. Last error: {last_error}")


df = load_csv(DATA_PATH)

required_cols = ["Restaurant Name", "Address", "Sentiment", "ID"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

work = df[required_cols].copy()

for col in required_cols:
    work[col] = work[col].astype(str).str.strip()

work = work[
    (work["Restaurant Name"].str.lower() != "nan") &
    (work["Restaurant Name"] != "") &
    (work["Address"].str.lower() != "nan") &
    (work["Address"] != "") &
    (work["ID"].str.lower() != "nan") &
    (work["ID"] != "")
].copy()

work["Restaurant Entity"] = (
    work["Restaurant Name"].str.lower().str.strip()
    + " | "
    + work["Address"].str.lower().str.strip()
)

work["Sentiment"] = work["Sentiment"].str.lower().str.strip()

sentiment_map = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}

work["Sentiment_clean"] = work["Sentiment"].map(sentiment_map)

work = work[
    work["Sentiment_clean"].isin(["positive", "negative", "neutral"])
].copy()

# My data is in long format, so each review should count only once here.
reviews_unique = work.drop_duplicates(
    subset=["Restaurant Entity", "ID"]
).copy()

restaurant_counts = (
    reviews_unique
    .pivot_table(
        index=["Restaurant Entity", "Restaurant Name", "Address"],
        columns="Sentiment_clean",
        values="ID",
        aggfunc="nunique",
        fill_value=0
    )
    .reset_index()
)

for col in ["positive", "negative", "neutral"]:
    if col not in restaurant_counts.columns:
        restaurant_counts[col] = 0

restaurant_counts = restaurant_counts.rename(columns={
    "positive": "Positive Reviews",
    "negative": "Negative Reviews",
    "neutral": "Neutral Reviews",
})

restaurant_counts["Total Reviews"] = (
    restaurant_counts["Positive Reviews"]
    + restaurant_counts["Negative Reviews"]
    + restaurant_counts["Neutral Reviews"]
)

restaurant_counts["Positive + Negative Reviews"] = (
    restaurant_counts["Positive Reviews"]
    + restaurant_counts["Negative Reviews"]
)

restaurant_counts["Net Sentiment Balance"] = np.where(
    restaurant_counts["Positive + Negative Reviews"] > 0,
    (
        restaurant_counts["Positive Reviews"]
        - restaurant_counts["Negative Reviews"]
    ) / restaurant_counts["Positive + Negative Reviews"],
    np.nan
)

restaurant_scores = restaurant_counts[
    restaurant_counts["Total Reviews"] >= MIN_REVIEWS
].copy()

restaurant_scores = restaurant_scores.sort_values(
    by="Net Sentiment Balance",
    ascending=False
)

summary_table = pd.DataFrame({
    "Measure": [
        "Restaurants included",
        "Mean net sentiment balance",
        "Median net sentiment balance",
        "25th percentile",
        "75th percentile",
        "Minimum net sentiment balance",
        "Maximum net sentiment balance",
        "Share of restaurants with positive balance",
        "Share of restaurants with negative balance",
        "Share of restaurants with neutral balance",
    ],
    "Value": [
        len(restaurant_scores),
        restaurant_scores["Net Sentiment Balance"].mean(),
        restaurant_scores["Net Sentiment Balance"].median(),
        restaurant_scores["Net Sentiment Balance"].quantile(0.25),
        restaurant_scores["Net Sentiment Balance"].quantile(0.75),
        restaurant_scores["Net Sentiment Balance"].min(),
        restaurant_scores["Net Sentiment Balance"].max(),
        (restaurant_scores["Net Sentiment Balance"] > 0).mean(),
        (restaurant_scores["Net Sentiment Balance"] < 0).mean(),
        (restaurant_scores["Net Sentiment Balance"] == 0).mean(),
    ]
})

summary_display = summary_table.copy()
summary_display["Value"] = summary_display["Value"].apply(
    lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x
)

print(summary_display.to_string(index=False))

output_path = out_dir / "restaurant_level_net_sentiment_balance.csv"
restaurant_scores.to_csv(output_path, index=False, encoding="utf-8-sig")

# One simple plot is enough for this part of the analysis.
plt.figure(figsize=(8, 5))
plt.hist(
    restaurant_scores["Net Sentiment Balance"].dropna(),
    bins=30
)
plt.xlabel("Net Sentiment Balance")
plt.ylabel("Number of restaurant entities")
plt.title("Distribution of Restaurant-Level Net Sentiment Balance")
plt.tight_layout()
plt.show()

