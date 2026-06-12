import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os


# ==========================
# Project Path
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================
# Load Model
# ==========================

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model.keras")
)


# ==========================
# Load Scaler
# ==========================

with open(
    os.path.join(BASE_DIR, "scaler.pkl"),
    "rb"
) as file:

    scaler = pickle.load(file)


# ==========================
# Load Dataset
# ==========================

def load_data():

    file_path = os.path.join(
        BASE_DIR,
        "breast-cancer.csv"
    )

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()


    if "id" in df.columns:
        df.drop(
            "id",
            axis=1,
            inplace=True
        )


    return df


# ==========================
# Preprocess Data
# ==========================

def preprocess_data(df):

    return scaler.transform(df)


# ==========================
# Single Patient Prediction
# ==========================

def predict_patient(df):

    data = preprocess_data(df)


    probability = model.predict(
        data,
        verbose=0
    )[0][0]


    if probability >= 0.5:

        prediction = "Malignant"

        confidence = probability * 100


    else:

        prediction = "Benign"

        confidence = (
            1 - probability
        ) * 100


    return (
        prediction,
        float(confidence)
    )


# ==========================
# Confidence Level
# ==========================

def get_risk_level(confidence):

    if confidence >= 95:
        return "Very High Confidence"

    elif confidence >= 85:
        return "High Confidence"

    elif confidence >= 70:
        return "Medium Confidence"

    else:
        return "Low Confidence"


# ==========================
# CSV Batch Prediction
# ==========================

def predict_csv(df):

    patient_ids = None


    if "id" in df.columns:

        patient_ids = df["id"]

        df = df.drop(
            "id",
            axis=1
        )


    data = preprocess_data(df)


    probabilities = model.predict(
        data,
        verbose=0
    )


    results = []


    for index, prob in enumerate(probabilities):

        probability = prob[0]


        if probability >= 0.5:

            diagnosis = "Malignant"

            confidence = probability * 100


        else:

            diagnosis = "Benign"

            confidence = (
                1 - probability
            ) * 100


        results.append(
            {
                "Patient ID":
                patient_ids.iloc[index]
                if patient_ids is not None
                else index + 1,

                "Prediction":
                diagnosis,

                "Confidence (%)":
                round(
                    float(confidence),
                    2
                )
            }
        )


    return pd.DataFrame(results)