# Financial News Sentiment Classifier

A sentence-level sentiment classifier for financial news, built with classical NLP and machine learning. Compares **Multinomial Naive Bayes** (primary algorithm) against **Logistic Regression** and **Linear SVM** on TF-IDF features.

## Overview

Financial news sentiment is a real signal used in equity research, algorithmic trading, and market monitoring. This project classifies financial news headlines/sentences into **positive**, **negative**, or **neutral** sentiment, with a focus on understanding *why* different classical ML algorithms perform differently on imbalanced, domain-specific text — not just which one scores highest.

## Dataset

- **Source:** [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) (Kaggle)
- **Underlying data:** Financial PhraseBank (Malo et al., 2014), 50%-agreement subset
- **Size:** 4,846 sentences, 2 columns (`Sentiment`, `News Headline`)
- **Class distribution:** neutral 59.4% · positive 28.1% · negative 12.5% (notably imbalanced)

## Project Structure

```
financial-news-sentiment/
├── data/
│   ├── raw/              # all-data.csv + original FinancialPhraseBank source files
│   └── processed/        # cleaned_data.csv (post text-preprocessing)
├── notebooks/
│   ├── 01_data_loading_eda.ipynb   # Tasks 1-3: loading, EDA, text cleaning
│   └── 02_modeling.ipynb           # Tasks 4-10: split, TF-IDF, models, evaluation
├── src/
├── requirements.txt
└── README.md
```

## Approach

1. **Data loading & EDA** — class distribution, sentence length analysis, raw word-frequency inspection
2. **Text preprocessing** — lowercasing, punctuation removal (preserving decimals for numeric comparability), tokenization, stopword removal, lemmatization
3. **Stratified 80/20 train-test split** to preserve class proportions given the imbalance
4. **TF-IDF vectorization** — unigrams, top 5,000 features, fit on train only
5. **Model training** — Multinomial Naive Bayes, Logistic Regression, Linear SVM, all on identical TF-IDF features for a fair comparison
6. **Evaluation** — precision/recall/F1 per class, with macro-F1 prioritized over raw accuracy given the class imbalance
7. **Error analysis** — manual review of misclassified sentences
8. **Interpretability** — extraction of Naive Bayes' most indicative words per class

## Results

| Model | Accuracy | Macro F1 | Negative Recall |
|---|---|---|---|
| Naive Bayes | 68.7% | 0.47 | 0.07 |
| Logistic Regression | 73.4% | 0.64 | 0.43 |
| **SVM** | **74.3%** | **0.68** | **0.57** |

SVM performed best overall. The negative-class recall trend (0.07 → 0.43 → 0.57) is the most telling result: Naive Bayes' reliance on class priors made it strongly biased toward the majority (neutral) class given only 12.5% negative examples in training, while LR and SVM's margin/boundary-based approaches were noticeably more resilient to the imbalance, even without any explicit rebalancing technique applied.

## Key Findings

- **Error analysis:** Sentences where a true negative was misclassified as neutral tend to use soft, factual financial phrasing ("cut to," "down from," "eaten by") rather than strongly negative words ("loss," "plunge"). These sentences are structurally similar to neutral training examples (company name + number + change verb), which a bag-of-words model can't reliably distinguish without understanding directional context.
- **Interpretability:** Naive Bayes' top words per class show heavy vocabulary overlap (`eur`, `mn`, `million`, `company`, `sale` appear across multiple classes). Only a small set of words are genuinely discriminative — `decreased`/`loss` for negative, `rose`/`increased` for positive — which independently confirms and explains the error analysis findings above.

## Limitations & Future Work

- TF-IDF + classical ML cannot capture negation, word order, or context — a transformer-based model (e.g. FinBERT) would likely handle the soft-language negative cases identified in error analysis more effectively.
- No explicit class-imbalance handling (e.g. `class_weight='balanced'`, SMOTE) was applied — a natural next extension to test against the current results.
- Numeric magnitude ("13.1%" vs "42%") is preserved in the cleaned text for readability but isn't leveraged by TF-IDF as a quantitative feature.

## Setup

```bash
conda create -n fin-sentiment python=3.12
conda activate fin-sentiment
pip install -r requirements.txt
python -m ipykernel install --user --name=fin-sentiment --display-name "Python (fin-sentiment)"
```

## How to Run

Run `notebooks/01_data_loading_eda.ipynb` first (data loading, EDA, cleaning), then `notebooks/02_modeling.ipynb` (split, vectorization, model training, evaluation, error analysis, interpretability).

## Citation

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782-796.

## License

Dataset licensed under CC BY-NC-SA 4.0, per the Kaggle listing.
