🚗 Craigslist Car Price Prediction
LLM-Enhanced ETL + Production Modeling Pipeline

This project builds an end-to-end machine learning pipeline to predict car prices using Craigslist listings. It combines web scraping, LLM-based feature extraction, and a production-style modeling workflow to continuously generate predictions and track model performance over time.

📌 Project Overview

This pipeline simulates a real-world data science system that:

Scrapes Craigslist car listings (fixed dataset)
Extracts structured features using RegEx + LLMs (Gemini)
Stores processed data in Google Cloud Storage (GCS)
Trains a machine learning model using historical data
Predicts prices for new listings (daily/hourly)
Tracks performance metrics over time
Syncs outputs to GitHub using GitHub Actions
⚙️ Pipeline Architecture

Data Flow:

Craigslist Scraper → LLM + RegEx ETL → GCS (structured data)
→ Model Training (Cloud Function) → Predictions + Metrics
→ GitHub Actions → Results stored in /results/all-files/

📊 Outputs (Generated Every Run)

Each run produces a folder with timestamp:

results/all-files/<run_id>/

Containing:

preds-llm.csv → Model predictions
metrics.json → Model performance metrics
permutation_importance.csv → Feature importance
pdp_*.png → Partial Dependence Plots
📈 Model Evaluation Metrics

The pipeline tracks:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)
Bias

These metrics are analyzed over time to evaluate model performance and stability.

🔍 Key Features
LLM-enhanced feature extraction (color, city, state, etc.)
Time-based train/holdout split (predicting today's listings)
Hyperparameter tuning using GridSearchCV
Permutation feature importance tracking
Partial Dependence Plot (PDP) generation
Automated pipeline using GitHub Actions + GCP
📊 Notebook Analysis

A Jupyter/Colab notebook is included to analyze:

Model performance trends over time
Feature importance stability
Partial dependence plots (model behavior)
Dataset growth and impact on predictions
🧠 Key Insights
Core features like year, mileage, and mileage per year consistently drive predictions
Model performance remains stable as more data is collected
PDPs confirm real-world pricing relationships
Hyperparameters converge, indicating a stable model
⚠️ Limitations
Some metrics (RMSE, MAPE, Bias) were introduced later, limiting long-term trends
Certain features show variability depending on incoming data
Model may slightly underpredict higher-value vehicles
🚀 Future Improvements
Add text-based features (description embeddings)
Improve outlier handling
Experiment with advanced models (XGBoost, LightGBM)
Continue collecting data for stronger trends
🛠️ Tech Stack
Python (Pandas, NumPy, Scikit-learn)
Google Cloud Platform (Cloud Functions, GCS)
GitHub Actions (automation)
Matplotlib (visualization)
LLM (Gemini) for feature extraction

👤 Author
Sachin Chahal
MS Business Analytics & Project Management
University of Connecticut
