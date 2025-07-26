import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.set_page_config(layout="wide")


def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
    df = df.rename(columns={"ConvertedCompYearly": "Salary"})
    df = df.dropna()

    df = df[df["Employment"] == "Employed, full-time"]
    df = df.drop("Employment", axis=1)

    def clean_experience(x):
        if x == "More than 50 years":
            return 50
        if x == "Less than 1 year":
            return 0.5
        return float(x)

    def clean_education(x):
        if "Bachelor’s degree" in x:
            return "Bachelor’s degree"
        if "Master’s degree" in x:
            return "Master’s degree"
        if "Professional degree" in x or "Other doctoral" in x:
            return "Post grad"
        return "Less than a Bachelors"

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)

    st.write("✅ Data loaded:", df.shape)
    return df


def load_model():
    with open("salary_model_gb2.pkl", "rb") as file:
        gbmod = pickle.load(file)
    return gbmod


def show_explore_page(df):
    st.title("Explore Software Engineer Salaries")
    st.write("### Based on 2023 Stack Overflow Developer Survey")

    data = df["Country"].value_counts()
    top_countries = data[:15].index
    df_filtered = df[df["Country"].isin(top_countries)]

    fig1, ax1 = plt.subplots()
    df_filtered["Country"].value_counts().plot(kind="barh", ax=ax1)
    ax1.set_xlabel("Number of Respondents")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    df_filtered.groupby("Country")["Salary"].mean().sort_values().plot(kind="barh", ax=ax2)
    ax2.set.xlabel("Mean Salary (USD)")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    df_filtered.groupby("EdLevel")["Salary"].mean().sort_values().plot(kind="barh", ax=ax3)
    ax3.set_xlabel("Mean Salary (USD)")
    st.pyplot(fig3)


def show_predict_page():
    st.title("Software Engineer Salary Prediction")

    countries = [
        "United States", "India", "United Kingdom", "Germany", "Canada",
        "France", "Brazil", "Spain", "Australia", "Netherlands"
    ]

    education = [
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad"
    ]

    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    model = load_model()

    ok = st.button("Calculate Salary")
    if ok:
        X = pd.DataFrame([[country, education_level, experience]],
                         columns=["Country", "EdLevel", "YearsCodePro"])
        salary = model.predict(X)[0]
        st.subheader(f"Estimated Salary: ${salary:,.2f}")


# --- Main App Logic ---
df = load_data()

page = st.sidebar.selectbox("Select Page", ("Explore", "Predict"))

if page == "Predict":
    st.write("🔮 Loading Predict Page...")
    show_predict_page()
else:
    st.write("🔍 Loading Explore Page...")
    show_explore_page(df)