import streamlit as st
import pandas as pd
import plotly.express as px
import pickle


def show_performance():

    st.markdown(
        """
        <h1 style='text-align:center;color:#1f77b4;'>
        📈 ANN Model Performance Dashboard
        </h1>
        """,
        unsafe_allow_html=True
    )


    try:

        with open("metrics.pkl", "rb") as file:
            metrics = pickle.load(file)


        with open("history.pkl", "rb") as file:
            history = pickle.load(file)


        col1, col2 = st.columns(2)


        col1.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )


        col2.metric(
            "Model",
            "Deep ANN"
        )


        st.divider()


        # Accuracy Curve
        accuracy_df = pd.DataFrame({
            "Epoch": range(
                1,
                len(history["accuracy"]) + 1
            ),
            "Training Accuracy": history["accuracy"],
            "Validation Accuracy": history["val_accuracy"]
        })


        fig_acc = px.line(
            accuracy_df,
            x="Epoch",
            y=[
                "Training Accuracy",
                "Validation Accuracy"
            ],
            title="Training vs Validation Accuracy"
        )


        st.plotly_chart(
            fig_acc,
            use_container_width=True
        )


        # Loss Curve
        loss_df = pd.DataFrame({
            "Epoch": range(
                1,
                len(history["loss"]) + 1
            ),
            "Training Loss": history["loss"],
            "Validation Loss": history["val_loss"]
        })


        fig_loss = px.line(
            loss_df,
            x="Epoch",
            y=[
                "Training Loss",
                "Validation Loss"
            ],
            title="Training vs Validation Loss"
        )


        st.plotly_chart(
            fig_loss,
            use_container_width=True
        )


        # Confusion Matrix

        st.subheader(
            "Confusion Matrix"
        )


        cm = metrics["confusion_matrix"]


        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Benign", "Malignant"],
            y=["Benign", "Malignant"],
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count"
            }
        )


        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )


        st.subheader(
            "Classification Report"
        )


        report = pd.DataFrame(
            metrics["report"]
        ).transpose()


        st.dataframe(
            report,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            "Could not load model performance data."
        )

        st.code(
            str(e)
        )