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

*To be updated after modeling is complete.*

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
