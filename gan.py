#!/usr/bin/env python3
"""GAN experiment for network anomaly detection on CICIDS 2017.

This script preserves the original dissertation experiment flow while removing
notebook-specific clutter and machine-specific paths.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

RANDOM_STATE = 42
DATA_PATH = Path("preprocessed_data.csv")


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set basic random seeds for more reproducible experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessing steps used in the dissertation experiment."""
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

    # Preserve the label selection used in the original experiment.
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


class Generator(nn.Module):
    """Generator network used in the dissertation GAN experiment."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator network used in the dissertation GAN experiment."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def main() -> None:
    set_seed()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}\n"
            "Place preprocessed_data.csv in the same directory as this script."
        )

    data = pd.read_csv(DATA_PATH)
    data = clean_data(data)

    label_columns = [column for column in data.columns if column.startswith("Label_")]
    X = data.drop(columns=label_columns)
    y = data["Label_BENIGN"]

    # Preserve the original dissertation experiment flow.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("Balanced data:")
    print(y_resampled.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    x_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32, device=device)

    input_dim = X_train.shape[1]
    generator = Generator(input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    learning_rate_gen = 1e-5
    learning_rate_dis = 3e-6
    optimizer_gen = optim.Adam(
        generator.parameters(), lr=learning_rate_gen, betas=(0.5, 0.999)
    )
    optimizer_dis = optim.Adam(
        discriminator.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999)
    )
    loss_function = nn.BCELoss()

    epochs = 50
    batch_size = 512
    batch_count = x_train.shape[0] // batch_size

    progress = tqdm(range(epochs * batch_count), desc="GAN training")

    for epoch in range(epochs):
        for index in range(batch_count):
            progress.update(1)

            real_samples = x_train[
                index * batch_size : (index + 1) * batch_size
            ]

            # Discriminator update.
            noise = torch.randn(batch_size, input_dim, device=device)
            generated_samples = generator(noise)
            combined_samples = torch.cat([generated_samples, real_samples], dim=0)

            discriminator_targets = torch.cat(
                [
                    torch.full((batch_size,), 0.2, device=device),
                    torch.full((batch_size,), 0.8, device=device),
                ]
            )

            discriminator.train()
            optimizer_dis.zero_grad()
            discriminator_output = discriminator(combined_samples).squeeze()
            discriminator_loss = loss_function(
                discriminator_output, discriminator_targets
            )
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), max_norm=1.0
            )
            optimizer_dis.step()

            # Generator update: five updates per discriminator update,
            # matching the original experiment.
            for _ in range(5):
                noise = torch.randn(batch_size, input_dim, device=device)
                generator_targets = torch.full(
                    (batch_size,), 0.9, device=device
                )

                optimizer_gen.zero_grad()
                generated_samples = generator(noise)
                generator_output = discriminator(generated_samples).squeeze()
                generator_loss = loss_function(
                    generator_output, generator_targets
                )
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_gen.step()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"[D loss: {discriminator_loss.item():.4f}] "
            f"[G loss: {generator_loss.item():.4f}]"
        )

    progress.close()

    discriminator.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, x_test.shape[0], batch_size):
            batch = x_test[start : start + batch_size]
            score = discriminator(batch).cpu().numpy()
            predictions.extend(score)

    scores = np.asarray(predictions).reshape(-1)
    y_test_np = y_test_tensor.cpu().numpy()

    # Preserve the anomaly thresholding rule used in the original experiment.
    threshold = np.percentile(scores, 1)
    y_pred = np.where(scores > threshold, 0, 1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_np, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_test_np, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


if __name__ == "__main__":
    main()
