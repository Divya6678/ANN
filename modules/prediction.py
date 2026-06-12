import streamlit as st
import pandas as pd

from utils import (
    load_data,
    predict_patient,
    predict_csv,
    get_risk_level
)


def show_prediction():

    df = load_data()

    st.markdown(
        """
        <h1 style='text-align:center;color:#1f77b4;'>
        🔬 AI Cancer Prediction
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Analyze a single patient or upload a CSV file for bulk prediction."
    )

    tab1, tab2 = st.tabs(
        [
            "Single Patient",
            "Bulk CSV Prediction"
        ]
    )


    # =========================
    # Single Patient Prediction
    # =========================

    with tab1:

        st.subheader(
            "Patient Tumor Measurements"
        )

        features = (
            df.drop(
                "diagnosis",
                axis=1
            ).columns
        )


        user_data = {}

        col1, col2 = st.columns(2)


        for i, feature in enumerate(features):

            with col1 if i % 2 == 0 else col2:

                user_data[feature] = st.number_input(
                    feature,
                    value=float(df[feature].mean()),
                    format="%.4f"
                )


        if st.button(
            "🧠 Analyze Patient",
            use_container_width=True
        ):

            input_df = pd.DataFrame(
                [user_data]
            )


            prediction, confidence = predict_patient(
                input_df
            )


            risk = get_risk_level(
                confidence
            )


            st.divider()


            if prediction == "Malignant":

                st.error(
                    f"""
                    ## 🔴 Malignant Tumor Detected

                    Prediction: **{prediction}**

                    Confidence: **{confidence:.2f}%**
                    """
                )

            else:

                st.success(
                    f"""
                    ## 🟢 Benign Tumor Detected

                    Prediction: **{prediction}**

                    Confidence: **{confidence:.2f}%**
                    """
                )


            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


            st.progress(
                float(confidence / 100)
            )


            st.info(
                f"Confidence Level: {risk}"
            )


    # =========================
    # Bulk CSV Prediction
    # =========================

    with tab2:

        st.subheader(
            "Upload CSV Patient Data"
        )


        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"]
        )


        if uploaded_file:

            csv_data = pd.read_csv(
                uploaded_file
            )


            st.write(
                "Preview of Uploaded Data"
            )


            st.dataframe(
                csv_data.head(),
                use_container_width=True
            )


            if st.button(
                "🚀 Predict All Patients",
                use_container_width=True
            ):


                result = predict_csv(
                    csv_data
                )


                st.success(
                    "Prediction Completed Successfully!"
                )


                st.dataframe(
                    result,
                    use_container_width=True
                )


                csv = result.to_csv(
                    index=False
                )


                st.download_button(
                    "📥 Download Results",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )