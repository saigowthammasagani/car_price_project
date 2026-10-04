"""Used Car Price Analytics & Prediction  |  Streamlit web app
Dataset: CarDekho 'Car details v3.csv' (Kaggle, ~8,100 records)
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Used Car Price Analytics", page_icon="🚗", layout="wide")
CURRENT_YEAR = 2026
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]
NUM = ["car_age", "km_driven", "mileage", "engine", "max_power", "seats"]


# ---------------------------------------------------------------- DATA
@st.cache_data
def load_raw(file):
    return pd.read_csv(file)


@st.cache_data
def preprocess(raw: pd.DataFrame):
    """Wrangling: returns clean df + a log of every step taken."""
    log, df = [], raw.copy()
    log.append(f"Raw data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    n = len(df); df = df.drop_duplicates()
    log.append(f"Removed {n - len(df):,} duplicate rows")

    for col in ["mileage", "engine", "max_power"]:   # '23.4 kmpl' -> 23.4
        df[col] = df[col].astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    log.append("Converted mileage / engine / max_power from text to numbers")

    df["brand"] = df["name"].str.split().str[0]
    rare = df["brand"].value_counts()[lambda s: s < 10].index
    df["brand"] = df["brand"].replace(list(rare), "Other")
    df["car_age"] = CURRENT_YEAR - df["year"]
    log.append("Feature engineering: brand (from name), car_age (from year)")

    miss = df[NUM].isna().sum().sum()
    df["seats"] = df["seats"].fillna(df["seats"].mode()[0])
    for c in NUM:
        df[c] = df[c].fillna(df[c].median())
    log.append(f"Imputed {miss:,} missing numeric values (median / mode)")

    n = len(df)
    for c in ["selling_price", "km_driven"]:         # IQR outlier removal
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        df = df[df[c].between(q1 - 3 * iqr, q3 + 3 * iqr)]
    log.append(f"Removed {n - len(df):,} extreme outliers (3xIQR on price & km_driven)")

    df = df.drop(columns=["name", "torque", "year"], errors="ignore").reset_index(drop=True)
    log.append(f"Clean data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, log


@st.cache_resource
def train_models(df: pd.DataFrame):
    X, y = df[CAT + NUM], np.log1p(df["selling_price"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM)])
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, max_depth=4,
                                                       learning_rate=0.08, random_state=42),
    }
    fitted, rows, preds = {}, [], {}
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("model", m)]).fit(Xtr, ytr)
        p = np.expm1(pipe.predict(Xte)); actual = np.expm1(yte)
        rows.append({"Model": name,
                     "R²": r2_score(actual, p),
                     "MAE (₹)": mean_absolute_error(actual, p),
                     "RMSE (₹)": np.sqrt(mean_squared_error(actual, p))})
        fitted[name], preds[name] = pipe, p
    return fitted, pd.DataFrame(rows), np.expm1(yte).values, preds


# ---------------------------------------------------------------- UI
st.title("🚗 Used Car Price Analytics & Prediction")
st.caption("Data wrangling • EDA • Machine Learning • Interactive prediction")

up = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv")
st.sidebar.markdown("Default file: `Car details v3.csv` in this folder "
                    "([Kaggle: CarDekho dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho))")
import os
if up is not None:
    raw = load_raw(up)
elif os.path.exists("Car details v3.csv"):
    raw = load_raw("Car details v3.csv")
else:
    raw = load_raw("sample_car_data.csv")
    st.warning("Running on SYNTHETIC sample data. Place the real Kaggle file "
               "'Car details v3.csv' in this folder (or upload it in the sidebar) for real results.")

df, log = preprocess(raw)
models, results, y_true, preds = train_models(df)

t1, t2, t3, t4, t5 = st.tabs(["📋 Overview", "🧹 Preprocessing", "📊 EDA",
                              "🤖 Model Comparison", "🎯 Predict Price"])

with t1:
    c = st.columns(4)
    c[0].metric("Records", f"{len(df):,}")
    c[1].metric("Avg price", f"₹{df.selling_price.mean():,.0f}")
    c[2].metric("Brands", df.brand.nunique())
    c[3].metric("Avg car age", f"{df.car_age.mean():.1f} yrs")
    st.subheader("Problem statement")
    st.write("Buying or selling a used car in India lacks transparent pricing. This app analyses "
             "~8,000 listings to uncover what drives resale value and predicts a fair price "
             "from a car's specifications using machine learning.")
    st.dataframe(df.head(50), use_container_width=True)

with t2:
    st.subheader("Wrangling steps")
    for i, s in enumerate(log, 1):
        st.markdown(f"**{i}.** {s}")
    a, b = st.columns(2)
    a.write("Missing values in raw data"); a.dataframe(raw.isna().sum().rename("missing"))
    b.write("Summary statistics (clean)"); b.dataframe(df.describe().T)

with t3:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="selling_price", nbins=50,
                    title="Distribution of selling price"), use_container_width=True)
    top = df.groupby("brand").selling_price.mean().sort_values(ascending=False).head(12).reset_index()
    c2.plotly_chart(px.bar(top, x="brand", y="selling_price",
                    title="Average price by brand (top 12)"), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.box(df, x="fuel", y="selling_price", title="Price by fuel type"),
                    use_container_width=True)
    c4.plotly_chart(px.box(df, x="transmission", y="selling_price", color="transmission",
                    title="Price by transmission"), use_container_width=True)
    c5, c6 = st.columns(2)
    c5.plotly_chart(px.scatter(df.sample(min(2000, len(df)), random_state=1), x="car_age",
                    y="selling_price", color="fuel", opacity=.6, title="Price vs car age"),
                    use_container_width=True)
    c6.plotly_chart(px.scatter(df.sample(min(2000, len(df)), random_state=1), x="max_power",
                    y="selling_price", color="transmission", opacity=.6,
                    title="Price vs engine power (bhp)"), use_container_width=True)
    corr = df[NUM + ["selling_price"]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Correlation heatmap"), use_container_width=True)
    st.info("Insights: newer cars and higher max power command higher prices; "
            "price falls steadily with age and km driven; automatics and diesel cost more on average.")

with t4:
    st.subheader("Performance on 20% hold-out test set")
    st.dataframe(results.style.format({"R²": "{:.3f}", "MAE (₹)": "{:,.0f}", "RMSE (₹)": "{:,.0f}"})
                 .highlight_max(subset="R²", color="#c6f6d5"), use_container_width=True)
    best = results.loc[results["R²"].idxmax(), "Model"]
    st.success(f"Best model: **{best}**")
    m = st.selectbox("Actual vs predicted for:", list(models), index=list(models).index(best))
    fig = px.scatter(x=y_true, y=preds[m], labels={"x": "Actual ₹", "y": "Predicted ₹"},
                     opacity=.5, title=f"{m}: actual vs predicted")
    fig.add_shape(type="line", x0=0, y0=0, x1=y_true.max(), y1=y_true.max(),
                  line=dict(dash="dash", color="red"))
    st.plotly_chart(fig, use_container_width=True)
    tree = models["Random Forest"]
    names = tree.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(tree.named_steps["model"].feature_importances_, index=names)\
            .sort_values(ascending=False).head(12).reset_index()
    imp.columns = ["feature", "importance"]
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                    title="Top features (Random Forest)").update_yaxes(autorange="reversed"),
                    use_container_width=True)

with t5:
    st.subheader("Estimate a fair resale price")
    a, b, c = st.columns(3)
    brand = a.selectbox("Brand", sorted(df.brand.unique()))
    fuel = a.selectbox("Fuel", sorted(df.fuel.unique()))
    trans = a.selectbox("Transmission", sorted(df.transmission.unique()))
    seller = b.selectbox("Seller type", sorted(df.seller_type.unique()))
    owner = b.selectbox("Owner", sorted(df.owner.unique()))
    seats = b.selectbox("Seats", sorted(df.seats.astype(int).unique()), index=2)
    age = c.slider("Car age (years)", 0, 25, 6)
    km = c.number_input("Km driven", 0, 500000, 60000, step=5000)
    mil = c.slider("Mileage (kmpl)", 5.0, 35.0, 19.0)
    eng = c.slider("Engine (CC)", 600, 3500, 1200, step=50)
    pw = c.slider("Max power (bhp)", 30, 300, 85)
    model_name = st.radio("Model", list(models), index=list(models).index(best), horizontal=True)
    if st.button("Predict price", type="primary"):
        X = pd.DataFrame([dict(brand=brand, fuel=fuel, seller_type=seller, transmission=trans,
                               owner=owner, car_age=age, km_driven=km, mileage=mil,
                               engine=eng, max_power=pw, seats=seats)])
        price = np.expm1(models[model_name].predict(X)[0])
        st.metric("Predicted selling price", f"₹{price:,.0f}")
        st.caption("All three models' estimates: " + " | ".join(
            f"{k}: ₹{np.expm1(v.predict(X)[0]):,.0f}" for k, v in models.items()))
