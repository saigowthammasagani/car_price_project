# Project Report: Used Car Price Analytics & Prediction

## 1. Problem Statement
Used-car pricing in India is opaque; buyers and sellers lack a data-driven reference for a fair price.

## 2. Objectives
- Identify factors that drive resale price.
- Build and compare ML models that predict selling price.
- Deliver results through an interactive web application.

## 3. Dataset
CarDekho Vehicle Dataset (Kaggle), ~8,100 records, 13 attributes: name, year, selling_price, km_driven, fuel,
seller_type, transmission, owner, mileage, engine, max_power, torque, seats.

## 4. Data Preprocessing & Wrangling
Duplicate removal; extraction of numeric values from mileage/engine/max_power text; brand extracted from name
(rare brands grouped as "Other"); car_age = 2026 - year; median/mode imputation; 3xIQR outlier removal on
price and km_driven; dropped torque (inconsistent units) and name. Log transform applied to price.

## 5. Exploratory Data Analysis
Price distribution, price by brand/fuel/transmission, price vs age and power, correlation heatmap.
(Add your observed insights and screenshots here.)

## 6. Models
| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Linear Regression | (fill) | (fill) | (fill) |
| Random Forest | (fill) | (fill) | (fill) |
| Gradient Boosting | (fill) | (fill) | (fill) |
Discuss: best model, why ensemble methods capture non-linear depreciation, feature importance.

## 7. Web Application
Streamlit app with five tabs: Overview, Preprocessing, EDA, Model Comparison, Predict Price. (Add screenshots.)

## 8. Conclusion & Future Work
Summarise findings. Future: hyperparameter tuning, SHAP explainability, cloud deployment, more recent data.
