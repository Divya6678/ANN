import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data


def show_dashboard():

    df = load_data()

    st.markdown(
        "<h1 style='text-align:center;color:#1f77b4;'>"
        "🧠 AI Breast Cancer Diagnostic Dashboard</h1>",
        unsafe_allow_html=True
    )


    total = len(df)
    malignant = len(df[df["diagnosis"]=="M"])
    benign = len(df[df["diagnosis"]=="B"])


    col1, col2, col3 = st.columns(3)

    col1.metric("Total Patients", total)
    col2.metric("Malignant Cases", malignant)
    col3.metric("Benign Cases", benign)


    st.divider()


    # Pie Chart
    pie_data = pd.DataFrame({
        "Type":["Malignant","Benign"],
        "Count":[malignant,benign]
    })


    fig = px.pie(
        pie_data,
        names="Type",
        values="Count",
        title="Cancer Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.divider()


    # Feature Distribution

    feature = st.selectbox(
        "Select Feature",
        df.drop("diagnosis",axis=1).columns
    )


    fig2 = px.histogram(
        df,
        x=feature,
        color="diagnosis",
        title=f"{feature} Distribution",
        marginal="box"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    st.divider()


    # Correlation Heatmap

    numeric_df = df.copy()

    numeric_df["diagnosis"] = numeric_df["diagnosis"].map(
        {
            "M":1,
            "B":0
        }
    )


    corr = numeric_df.corr()


    fig3 = px.imshow(
        corr,
        title="Feature Correlation Heatmap",
        aspect="auto"
    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    with st.expander("View Dataset"):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )