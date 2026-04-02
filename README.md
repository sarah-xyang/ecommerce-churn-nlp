# E-Commerce Churn Analysis with NLP & LLM Insights

**Domain:** E-Commerce | **Stakeholder:** Growth Team | **Dataset:** Olist Brazilian E-Commerce

## Business Problem

Olist's growth team knows customers churn — but not why. This project goes beyond
predicting *who* will churn to diagnosing *why*, using NLP sentiment analysis on
customer reviews combined with behavioral and transactional features.

The output is an actionable churn driver report with LLM-generated summaries
designed for non-technical stakeholders.

## Technical Approach

| Layer | Method |
|---|---|
| Churn prediction | XGBoost classifier with SHAP explainability |
| Sentiment analysis | TextBlob on Portuguese→English review text |
| Feature engineering | Multi-table join across 9 relational CSV files |
| Stakeholder communication | Anthropic Claude API–generated insight summaries |

## Project Structure
```
├── notebooks/
│   ├── 01_eda.ipynb               # Churn rate, order patterns, review── 02_preprocessing.ipynb     # Table joins, NLP pipeline, feature engineering
│   ├── 03_modeling.ipynb          # XGBoost, SHAP, model comparison
│   └── 04_business_impact.ipynb   # ROI model, LLM insight summaries
├── src/
│   ├── data_loader.py             # Load and join 9 CSV files
│   ├── feature_engineering.py     # Churn label, order-level features
│   ├── nlp_pipeline.py            # Sentiment scoring on review text
│   ├── model_utils.py             # Training helpers, evaluation metrics
│   └── llm_insights.py            # Anthropic API integration
├── tests/                         # pytest unit tests
├── data/
│   ├── raw/                       # Olist CSV files (gitignored)
│   └── processed/                 # Engineered features (gitignored)
├── .env.example                   # API key template
└── requirements.txt
```

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https:// Results

## Key Results

| Metric | Value |
|--------|-------|
| Customer-level churn rate | 97.0% (observation-window adjusted) |
| Customers analysed | 68,169 (orders placed before April 2018) |
| Best model | XGBoost with scale_pos_weight |
| ROC-AUC | 0.633 |
| F1 score | 0.803 |
| Retention campaign ROI | 236% (at R$25 cost/customer, 15% conversion) |

### Top Churn Predictors (SHAP)
1. `payment_installments` — payment friction is the strongest churn signal
2. `payment_value` — higher-value first orders do not predict return behaviour
3. `freight_ratio` — freight cost as a share of order value drives churn
4. `delivery_delay_days` — late deliveries compound churn risk
5. `days_to_delivery` — absolute wait time matters independently of delay

### NLP & LLM Layer
- `has_review_text` ranks 16th of 57 features — whether a leaves a review is more predictive than sentiment score
- `sentiment_polarity` ranks 39th — TextBlob on Portuguese text adds 
  marginal signal only
- Anthropic API analysis of 50 negative churned reviews identified 5 
  qualitative churn themes: non-delivery, delivery delays, wrong products 
  received, counterfeit goods, and incomplete orders

### Honest Model Assessment
ROC-AUC of 0.633 reflects a genuine data limitation: 97% platform-wide 
churn means there is no single dominant predictor that cleanly separates 
churners from returners. The model is useful for ranking customers by 
retention likelihood, not for binary classification of individuals.

### Observation Window Methodology
29.6% of raw dataset customers were excluded due to right-censoring bias 
— they ordered in the final 6 months of the dataset and had insufficient 
time to demonstrate return behaviour. Only customers with orders before 
April 2018 are included, ensuring every churn label reflects a complete 
observation window.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Portfolio Context

Project 3 of 5 in a senior data science portfolio.
- Project 1: [Healthcare Readmission Prediction](https://github.com/sarah-xyang/healthcare-readmission-prediction)
- Project 2: [AV Fleet Predictive Maintenance](https://github.com/sarah-xyang/av-fleet-predictive-maintenance)
