#!/usr/bin/env python3
"""1D CNN experiment for binary network anomaly detection on CICIDS 2017."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical

RANDOM_STATE = 42
DATA_PATH = Path("preprocessed_data.csv")
EPOCHS = 10
BATCH_SIZE = 32


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessing workflow used in the original CNN notebook."""
    data = data.drop_duplicates().copy()

    threshold = 1e12
    large_value_mask = data.apply(
        lambda column: column.map(
            lambda value: np.nan
            if isinstance(value, (int, float)) and abs(value) > threshold
            else value
        )
    )
    data = data.dropna(axis=1, thresh=large_value_mask.notna().sum())

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    data.columns = data.columns.str.strip()
    data = data.map(lambda value: value.strip() if isinstance(value, str) else value)

    all_zero_columns = data.columns[(data == 0).all()]
    data = data.drop(columns=all_zero_columns)

    label_columns = [column for column in data.columns if column.startswith("Label_")]

    selected_labels = [
        "Label_BENIGN",
        "Label_DoS Slowhttptest",
        "Label_DoS slowloris",
        "Label_FTP-Patator",
        "Label_SSH-Patator",
    ]
    columns_to_drop = [
        column for column in label_columns if column not in selected_labels
    ]
    data = data.drop(columns=columns_to_drop)

    return data


def build_model(input_shape: tuple[int, int]) -> Sequential:
    """Build the 1D CNN architecture used in the dissertation."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=6, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=3, strides=2, padding="same"),
            Conv1D(filters=64, kernel_size=6, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=3, strides=2, padding="same"),
            Conv1D(filters=64, kernel_size=6, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=3, strides=2, padding="same"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    np.random.seed(RANDOM_STATE)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}\n"
            "Place preprocessed_data.csv in the same directory as this script."
        )

    data = pd.read_csv(DATA_PATH)
    data = clean_data(data)

    label_columns = [column for column in data.columns if column.startswith("Label_")]
    selected_labels = [
        "Label_BENIGN",
        "Label_DoS Slowhttptest",
        "Label_DoS slowloris",
        "Label_FTP-Patator",
        "Label_SSH-Patator",
    ]

    X = data.drop(columns=selected_labels)
    # The dissertation frames the task as benign vs anomalous classification.
    # Label_BENIGN is retained as the binary target.
    y = data["Label_BENIGN"].astype(int)

    # Keep all remaining non-feature label columns out of X.
    X = X.drop(columns=[column for column in label_columns if column in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    # Conv1D expects (samples, timesteps/features, channels).
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train_one_hot = to_categorical(y_train.to_numpy(), num_classes=2)
    y_test_one_hot = to_categorical(y_test.to_numpy(), num_classes=2)

    model = build_model((X_train.shape[1], 1))
    model.summary()

    model.fit(
        X_train,
        y_train_one_hot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test_one_hot),
        verbose=1,
    )


if __name__ == "__main__":
    main()
