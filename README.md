# RainWise-Machine-Learning-Based-Rainfall-Prediction-System


# Overview

This project focuses on predicting whether it will rain tomorrow in Australia using historical weather data. The goal is to support weather forecasting and agricultural planning by developing a reliable model for rainfall prediction. Accurate rainfall forecasts can assist farmers, meteorologists, and water management systems in decision-making and preparedness.

The study explores data preprocessing, exploratory data analysis (EDA), and comparison of various machine learning models. The final model demonstrates robust predictive performance using an optimized Random Forest classifier.

# Dataset Source

Dataset: Australian Weather Dataset (weatherAUS.csv)

Size: ~145,000 rows and 23 columns

Features: Includes daily weather observations such as temperature, humidity, wind speed, pressure, and evaporation.

Target Variable: RainTomorrow (Yes/No)

Preprocessing Steps:

Removed irrelevant or high-missing-value columns (Evaporation, Sunshine, Cloud9am, Cloud3pm, Date, Location, RISK_MM).

Encoded categorical features using LabelEncoder.

Imputed missing numerical values with mean imputation.

Split the dataset into training (80%) and testing (20%) subsets.

# Methods and Approach

The project followed a structured machine learning workflow:

Exploratory Data Analysis (EDA):

Identified patterns between humidity, wind, and rainfall.

Visualized correlations using heatmaps and pair plots.

Model Development:

Compared algorithms such as Logistic Regression, XGBoost, LightGBM, CatBoost, and Random Forest.

Used Random Forest for final deployment due to its high accuracy and stability.

Model Optimization:

Tuned hyperparameters (max_depth, n_estimators) for improved generalization.

Model Persistence:

Stored model and preprocessing artifacts (rainfall_rf_model.pkl, imputer.pkl, label_encoders.pkl, feature_order.pkl) using joblib.

# Method Comparison (Illustrative Summary)
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	78%	74%	70%	72%
XGBoost	84%	81%	79%	80%
LightGBM	85%	83%	81%	82%
Random Forest (Final)	86%	84%	83%	83.5%
=> Steps to Run the Code

Clone the repository:

git clone https://github.com/<your-username>/rainfall-prediction.git
cd rainfall-prediction


Install dependencies:

pip install -r requirements.txt


Run the model training script:

python rainfall_model.py


(Optional) Launch the Streamlit app for interactive prediction (if included later):

streamlit run app.py

# Experiments and Results

Conducted multiple trials with different ML models.

Performed EDA to understand feature significance—humidity and pressure were the strongest predictors.

Visualized feature importance for interpretability.

The Random Forest model achieved the best trade-off between accuracy and overfitting.

Example Visualization

Feature Importance Plot: Demonstrates which factors most influence rainfall.

Confusion Matrix: Shows classification performance on test data.

# Conclusion

This project highlights that ensemble-based methods, particularly Random Forest, provide reliable and interpretable predictions for rainfall forecasting. Through careful preprocessing, encoding, and model tuning, the model achieved over 85% accuracy. Future work could include integrating time-series analysis, deep learning (LSTM), and real-time API deployment.

# References

Bureau of Meteorology, Australia – https://www.bom.gov.au

Kaggle Dataset – “Weather Dataset: Rattle Package”

Scikit-learn Documentation – https://scikit-learn.org

LightGBM and CatBoost official documentation

# Requirements

The project uses the following Python packages (from requirements.txt):

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
lightgbm
catboost
xgboost
mlxtend
joblib
requests
