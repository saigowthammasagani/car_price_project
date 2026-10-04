# Used Car Price Analytics & Prediction

A Data Analytics & Visualization web app (Streamlit) with data wrangling, EDA, three ML models and a price predictor.

## Folder contents
| File | Purpose |
|---|---|
| `app.py` | Main web application |
| `requirements.txt` | Python dependencies |
| `sample_car_data.csv` | SYNTHETIC demo data (5,060 rows) so the app runs offline |
| `generate_sample_data.py` | Script that created the sample data |
| `test_app.py` | Optional headless smoke test (`python test_app.py`) |
| `REPORT.md` | Report template for submission |

## Run
1. `pip install -r requirements.txt`
2. **For real results:** download `Car details v3.csv` from
   https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho and place it in this folder
   (the app uses it automatically; otherwise it falls back to the synthetic sample).
3. `streamlit run app.py`  ->  opens at http://localhost:8501

## Pipeline
Raw CSV -> dedupe, text-to-number parsing, feature engineering (brand, car_age), imputation, IQR outlier removal
-> EDA (6 charts + correlation heatmap) -> Linear Regression / Random Forest / Gradient Boosting
(log-price target, 80/20 split, R2 / MAE / RMSE) -> interactive predictor.
