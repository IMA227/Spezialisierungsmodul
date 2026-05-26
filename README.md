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

### `Sentiment_lexicon_Based_3_Models.py`
Applies 3 classical lexicon-based sentiment approaches (TextBlobDE, SentiWS, GermanPolarityClues) as baseline sentiment models.

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

1. **Data collection**  
   Run `Scrapper.py` to collect restaurant and review data from Speisekarte.de.

2. **Regional enrichment**  
   Run `PLZ API.py` to prepare postal-code and federal-state information.

3. **Sampling for manual annotation**  
   Run `Sampling_1000.py` to create the labelled sample used for model development and evaluation.

4. **Sentiment modelling**  
   Run the sentiment baseline and model scripts:
   - `Sentiment_lexicon_Based_3_Models.py`
   - `Cardiff_xlm_Roberta_Sentiment.py`
   - `GBERT_Sentiment_Analysis.py`

5. **Topic modelling**  
   Run the topic classification scripts:
   - `Base Model Topics.py`
   - `GBERT_GridSearch_Topics.py`
   - `GBERT_learning_curve_Topics.py`
   - `GBERT_Topics_Inference.py`

6. **Reviewer-name-based gender-indicator modelling**  
   Run:
   - `GBERT_Gender_Grid Search.py`
   - `GBERT_Gender_Inference.py`

7. **Restaurant-level sentiment aggregation**  
   Run `Net Sentiment Balance.py` to calculate restaurant-level net sentiment balance.

8. **Exploratory data analysis and final visualizations**  
   Open `EDAs.ipynb` to generate and review the final descriptive analyses and figures.

---

## Notes

The scripts use local file paths (which had been removed) and may need to be adjusted before running on another machine.  
