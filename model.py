import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("breast-cancer.csv")

df.columns = df.columns.str.strip()

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# ==========================
# Features and Target
# ==========================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


# ==========================
# ANN Architecture
# ==========================

model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    BatchNormalization(),

    Dropout(0.3),


    Dense(
        64,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.2),


    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================
# Callbacks
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10
)


# ==========================
# Training
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        early_stop,
        reduce_lr
    ],
    verbose=1
)


# ==========================
# Evaluation
# ==========================

probabilities = model.predict(X_test)

predictions = (
    probabilities > 0.5
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions
)


cm = confusion_matrix(
    y_test,
    predictions
)


report = classification_report(
    y_test,
    predictions,
    output_dict=True
)


print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)


# ==========================
# Save Files
# ==========================

model.save("model.keras")


with open(
    "history.pkl",
    "wb"
) as file:
    pickle.dump(
        history.history,
        file
    )


with open(
    "metrics.pkl",
    "wb"
) as file:
    pickle.dump(
        {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "report": report
        },
        file
    )


print(
    "Model, scaler and metrics saved successfully!"
)