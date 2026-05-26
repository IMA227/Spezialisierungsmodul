# A Multi-Method Analysis of Restaurant Reviews in Germany: Sentiment, Topic Classification, and Spatio-Temporal Patterns

This project analyzes German restaurant reviews collected from Speisekarte.de using web scraping, sentiment analysis, topic classification, and exploratory data analysis.  
The workflow combines classical Python-based NLP methods with transformer-based models such as Cardiff XLM-RoBERTa and German BERT/GBERT.  
Additional analysis includes reviewer-name-based gender indicators, regional enrichment using postal codes, temporal patterns, and restaurant-level sentiment aggregation.

---

## Scripts and Notebooks

### `Scrapper.py`
Scrapes restaurant review data from Speisekarte.de

### `PLZ API.py`
Retrieves German postal-code and federal-state information from OpenPLZ API and prepares the regional mapping used for spatial analysis.

### `Sampling_1000.py`
Creates a stratified sample of 1,000 reviews (based on Bundesland, sentiment, and review year) for manual annotation and model evaluation.

### `Sentiment_lexicon_Based_3 Models.py`
Applies three classical lexicon-based sentiment approaches: TextBlobDE, SentiWS, and GermanPolarityClues.

### `Cardiff_xlm_Roberta_Sentiment.py`
Runs sentiment classification using the pretrained Cardiff XLM-RoBERTa sentiment model.

### `GBERT_Sentiment_Analysis.py`
Trains and evaluates the GBERT-based sentiment classification model on the labelled sample.

### `Base Model Topics.py`
Runs the initial baseline model (TF-IDF) for multi-label topic classification.

### `GBERT_GridSearch_Topics.py`
Performs grid search experiments for the GBERT-based topic classification model.

### `GBERT_learning_curve_Topics.py`
Analyzes how topic-classification performance changes with different training sample sizes.

### `GBERT_Topics_Inference.py`
Applies the trained topic classification model to the full review dataset.

### `GBERT_Gender_Grid Search.py`
Performs grid search experiments for reviewer-name-based gender-indicator classification.

### `GBERT_Gender_Inference.py`
Applies the trained gender-indicator model to the full dataset.

### `Net Sentiment Balance.py`
Calculates restaurant-level net sentiment balance to summarize positive and negative review tendencies per restaurant.

### `EDAs.ipynb`
Contains the exploratory data analysis, visualizations, and descriptive analysis of sentiment, topics, regions, time patterns, and gender-indicator groups.

---

## Order of Operation

The project workflow starts with scraping the restaurant reviews and enriching them with regional information. After that, a labelled sample is created for model development and evaluation. The sentiment models are then tested, followed by the topic classification models and the reviewer-name-based gender-indicator model. Finally, the restaurant-level sentiment balance is calculated and the results are explored in the EDA notebook.

`Scrapper.py`  
→ `PLZ API.py`  
→ `Sampling_1000.py`  
→ Sentiment scripts:  
`Sentiment_lexicon_Based_3 Models.py` → `Cardiff_xlm_Roberta_Sentiment.py` → `GBERT_Sentiment_Analysis.py`  
→ Topic scripts:  
`Base Model Topics.py` → `GBERT_GridSearch_Topics.py` → `GBERT_learning_curve_Topics.py` → `GBERT_Topics_Inference.py`  
→ Gender-indicator scripts:  
`GBERT_Gender_Grid Search.py` → `GBERT_Gender_Inference.py`  
→ `Net Sentiment Balance.py`  
→ `EDAs.ipynb`

---

## Notes

The scripts originally used local file paths. These paths were removed before publishing, so users need to adjust the input and output paths before running the scripts.
