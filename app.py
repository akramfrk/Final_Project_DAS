import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diabetes Analysis", page_icon="📊", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Diabetes Dataset Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>Complete Pipeline: Collection, Preprocessing, EDA, and Linear Regression Modeling</h4>", unsafe_allow_html=True)
st.divider()

@st.cache_data
def load_data():
    df = pd.read_csv('./diabetes.csv')
    df_clean = df.copy()
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df, df_clean

df, df_clean = load_data()

TARGET   = 'Glucose'
FEATURES = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

@st.cache_resource
def train_model(df_clean):
    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    metrics = {
        'r2':    r2_score(y_test, y_pred),
        'mse':   mean_squared_error(y_test, y_pred),
        'mae':   mean_absolute_error(y_test, y_pred),
        'rmse':  np.sqrt(mean_squared_error(y_test, y_pred)),
        'y_test': y_test.values,
        'y_pred': y_pred,
        'residuals': y_test.values - y_pred,
    }
    return model, scaler, metrics

model, scaler, metrics = train_model(df_clean)

page = st.sidebar.radio("Navigation", [
    "1. Data Collection & Preprocessing",
    "2. Exploratory Data Analysis (EDA)",
    "3. Linear Regression Modeling",
    "4. Make a Prediction",
])

if page == "1. Data Collection & Preprocessing":
    st.header("Data Collection & Preprocessing")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data (with 0s)")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Clean Data (Medians imputed)")
        st.dataframe(df_clean.head(10))

    st.subheader("Dataset Statistics")
    st.dataframe(df_clean.describe().round(2))

    st.subheader("Zero values per column (treated as missing)")
    zero_counts = (df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum()
    fig, ax = plt.subplots(figsize=(7, 3))
    zero_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_ylabel("Count of zeros")
    ax.set_title("Impossible zero values replaced by column median")
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig)

    st.info("Zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are "
            "physiologically impossible and were replaced by the column median before modeling.")

elif page == "2. Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.caption("Glucose has notable correlations with Insulin, BMI, Age, and Outcome — "
               "supporting its use as the regression target.")

    st.subheader("Glucose Distribution (target variable)")
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df_clean[TARGET], bins=30, color='steelblue', edgecolor='white')
    axes[0].set_xlabel("Glucose")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram")
    axes[1].boxplot(df_clean[TARGET], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[1].set_ylabel("Glucose")
    axes[1].set_title("Boxplot")
    st.pyplot(fig2)

    st.subheader("Glucose vs Feature (scatter)")
    selected = st.selectbox("Select feature", FEATURES)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.scatter(df_clean[selected], df_clean[TARGET], alpha=0.4, s=15, color='steelblue')
    m, b = np.polyfit(df_clean[selected], df_clean[TARGET], 1)
    xs = np.linspace(df_clean[selected].min(), df_clean[selected].max(), 100)
    ax3.plot(xs, m*xs+b, color='tomato', linewidth=1.5, label='Trend')
    ax3.set_xlabel(selected)
    ax3.set_ylabel("Glucose")
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Feature Distributions")
    fig4, axes4 = plt.subplots(2, 4, figsize=(14, 6))
    for i, col in enumerate(FEATURES):
        ax = axes4[i//4][i%4]
        ax.hist(df_clean[col], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(col, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig4)

elif page == "3. Linear Regression Modeling":
    st.header("Linear Regression Results")
    st.markdown(
        "We apply **Linear Regression** to predict **Glucose level** (a continuous variable) "
        "from the other patient measurements. This is a proper use of Linear Regression "
        "since the target is continuous."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score",  f"{metrics['r2']:.4f}")
    col2.metric("MSE",       f"{metrics['mse']:.2f}")
    col3.metric("MAE",       f"{metrics['mae']:.2f}")
    col4.metric("RMSE",      f"{metrics['rmse']:.2f}")

    st.markdown(f"**Interpretation:** The model explains **{metrics['r2']*100:.1f}%** of the "
                f"variance in Glucose. On average, predictions are off by **{metrics['mae']:.1f} mg/dL**.")

    st.subheader("Predicted vs Actual Glucose")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(metrics['y_test'], metrics['y_pred'], alpha=0.5, s=20, color='steelblue')
    mn = min(metrics['y_test'].min(), metrics['y_pred'].min())
    mx = max(metrics['y_test'].max(), metrics['y_pred'].max())
    ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect prediction')
    ax1.set_xlabel("Actual Glucose")
    ax1.set_ylabel("Predicted Glucose")
    ax1.set_title("Predicted vs Actual")
    ax1.legend()
    st.pyplot(fig1)
    st.caption("Points close to the red dashed line = accurate predictions. "
               "Scatter around the line shows remaining prediction error.")

    st.subheader("Residuals Plot")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(metrics['y_pred'], metrics['residuals'], alpha=0.5, s=20, color='steelblue')
    ax2.axhline(0, color='tomato', linestyle='--', linewidth=1.5)
    ax2.set_xlabel("Predicted Glucose")
    ax2.set_ylabel("Residual (Actual − Predicted)")
    ax2.set_title("Residuals vs Predicted")
    st.pyplot(fig2)
    st.caption("A good model has residuals scattered randomly around 0 with no clear pattern.")

    st.subheader("Feature Coefficients")
    coef_df = pd.DataFrame({
        'Feature': FEATURES,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax3, palette="coolwarm")
    ax3.axvline(0, color='black', linestyle='--')
    ax3.set_title("Feature Coefficients (standardized)")
    st.pyplot(fig3)

    st.success(
        "**Interpretation:** Insulin and SkinThickness are the strongest positive predictors "
        "of Glucose level. Being diabetic (Outcome=1) is also positively associated with higher "
        "Glucose. BloodPressure shows a slight negative relationship."
    )

elif page == "4. Make a Prediction":
    st.header("Predict Glucose Level")
    st.markdown("Adjust the sliders to simulate a patient profile. "
                "The Linear Regression model will predict their **Glucose level**.")

    col1, col2 = st.columns(2)
    with col1:
        preg    = st.slider("Pregnancies",                  0,    17,   3)
        bp      = st.slider("Blood Pressure (mm Hg)",       0,   122,  70)
        skin    = st.slider("Skin Thickness (mm)",          0,    99,  20)
        insulin = st.slider("Insulin (µU/ml)",              0,   846,  79)
    with col2:
        bmi     = st.slider("BMI",                          0.0, 67.1, 32.0)
        dpf     = st.slider("Diabetes Pedigree Function",   0.078, 2.42, 0.47)
        age     = st.slider("Age",                         21,    81,  33)
        outcome = st.selectbox("Diabetic?", options=[0, 1],
                               format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    input_data   = np.array([[preg, bp, skin, insulin, bmi, dpf, age, outcome]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]

    st.divider()
    st.subheader(f"Predicted Glucose Level: {prediction:.1f} mg/dL")

    if prediction < 100:
        st.success("Normal glucose range (< 100 mg/dL)")
    elif prediction < 126:
        st.warning("Pre-diabetic range (100–125 mg/dL)")
    else:
        st.error("Diabetic range (≥ 126 mg/dL)")