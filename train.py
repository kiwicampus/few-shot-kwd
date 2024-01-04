"""
train.py
Training script for the few-shot keyword detection model.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from typer import Typer, run

from python_path import PythonPath

with PythonPath("./"):
    from multilingual_kws.data_pipeline import (
        standard_microspeech_model_settings,
        file2spec,
    )
    from multilingual_kws.embedding.transfer_learning import (
        transfer_learn,
    )

app = Typer()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = Path(__file__)
data_dir = file_path.parent.parent.parent / "data/keyword_dataset"


def main(
    toy: Optional[bool] = True,
    num_epochs: Optional[int] = 1,
    num_batches: Optional[int] = 1,
    backprop_into_embedding: Optional[bool] = False,
    unknown_percentage: Optional[float] = 50.0,
) -> None:
    """
    Main function for training the few-shot keyword detection model.

    Args:
        toy (Optional[bool]): Whether to use a toy dataset for training.
        num_epochs (Optional[int]): Number of epochs to train the model.
        num_batches (Optional[int]): Number of batches to train the model.
        backprop_into_embedding (Optional[bool]): Whether to backpropagate into the
            embedding model.
        unknown_percentage (Optional[float]): Percentage of unknown samples to use
            for training.
    """
    print(f" --- Starting training, using toy as {toy} ---")

    # Using subprocess to call ls command and save the output in list format

    # Original keyword examples are the ones recorded by the bot on field
    original_keyword_examples = (
        subprocess.check_output(
            [
                "find",
                os.path.join(data_dir, "ours", "original-keyword-examples"),
                "-type",
                "f",
                "-name",
                "*.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )
    # Crawling keyword examples are the examples taken and curated from the data
    # crawling process, these took max 15 audios per person and set the file to the
    # the correct format

    crawling_keyword_examples = (
        subprocess.check_output(
            [
                "find",
                os.path.join(data_dir, "ours", "crawling-keyword-examples"),
                "-type",
                "f",
                "-name",
                "*.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )

    # Synthetic keyword examples are the ones generated synthetically from the
    # using Suno generative model
    synthetic_keyword_examples = (
        subprocess.check_output(
            [
                "find",
                os.path.join(data_dir, "ours", "synthetic-keyword-examples"),
                "-type",
                "f",
                "-name",
                "*.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )

    # Other keyword examples are the ones taken on the field by the bot using the ability to record
    # the audio that it classifies as other, these are curated to select only the true positives
    other_keyword_examples = (
        subprocess.check_output(
            [
                "find",
                os.path.join(data_dir, "ours", "other-keyword-examples"),
                "-type",
                "f",
                "-name",
                "*.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )

    positive_files = list(
        set(
            original_keyword_examples
            + crawling_keyword_examples
            + synthetic_keyword_examples
            + other_keyword_examples
        )
    )
    positive_files.remove("")
    print(f"Founded {len(positive_files)} positive files")

    # Negative keyword examples are the ones taken on the field by the bot using the ability to record
    # the audio that it classifies as other, these are curated to select only the false positives
    negative_keyword_examples = (
        subprocess.check_output(
            [
                "find",
                os.path.join(data_dir, "ours", "negative-keyword-examples"),
                "-type",
                "f",
                "-name",
                "*.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )

    # Away negative keyword examples are the ones taken on the field by the bot using the ability to record continously
    # conversations, these are separated into 1-second chunks and convert to the correct format
    away_negative_keyword_examples = (
        subprocess.check_output(
            [
                "find",
                os.path.join(data_dir, "ours", "away-negative-keyword-examples"),
                "-type",
                "f",
                "-name",
                "*.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )
    # Call gsutil ls command to get all files in the bucket, this is done because call them
    # with find command is too slow

    # The google speech command benchmark dataset is used as negative examples
    google_dataset = (
        subprocess.check_output(
            [
                "gsutil",
                "ls",
                "gs://autonomy-vision/audios/external-datasets/google_dataset/**.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )

    # The mswc dataset is also used as negative examples, we just donwload part of the data of english,
    # spanish and french speakers
    mswc = (
        subprocess.check_output(
            [
                "gsutil",
                "ls",
                "gs://autonomy-vision/audios/external-datasets/mswc/**.wav",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )

    negative_files = list(
        set(
            negative_keyword_examples
            + away_negative_keyword_examples
            + google_dataset
            + mswc
        )
    )
    negative_files.remove("")
    negative_files = [f for f in negative_files if ".wav" in f]
    print(f"Founded {len(negative_files)} negative files")

    # Convert the obtained URIs to local path
    external_dataset_path = data_dir / "external"
    external_dataset_path_str = str(external_dataset_path)

    negative_files = [
        f.replace(
            "gs://autonomy-vision/audios/external-datasets",
            external_dataset_path_str,
        )
        for f in negative_files
    ]

    # Convert files to Path
    positive_files = [Path(f) for f in positive_files]
    negative_files = [Path(f) for f in negative_files]

    if toy:
        print(" --- Using toy dataset ---")
        positive_files = positive_files[:200]
        negative_files = negative_files[:2000]

    model_dir = file_path.parent.parent.parent / "models" / "kwd" / "embedding_model"
    # base_model = tf.keras.models.load_model(model_dir)
    # embedding = tf.keras.models.Model(
    #     name="embedding_model",
    #     inputs=base_model.inputs,
    #     outputs=base_model.get_layer(name="dense_2").output,
    # )
    # embedding.trainable = False

    # Prepare data
    negative_files_str = [str(path) for path in negative_files]

    # Split the list of files into train, validation, and test sets
    train_negatives, test_negatives = train_test_split(
        negative_files_str, test_size=0.1, random_state=42
    )

    # Print the number of files in each set
    print(f"Number of train negative files: {len(train_negatives)}")
    print(f"Number of test negative files: {len(test_negatives)}")

    # TODO: Change this split with one that takes into account the origin
    # of the recordings (the bot, synthetic, etc.) and to avoid data leakage
    positive_files_str = [str(path) for path in positive_files]
    # Split the list of files into train, validation, and test sets
    train_samples, test_samples = train_test_split(
        positive_files_str, test_size=0.2, random_state=42
    )
    train_samples, valid_samples = train_test_split(
        train_samples, test_size=0.3, random_state=42
    )

    # Print the number of files in each set
    print(f"Number of train files: {len(train_samples)}")
    print(f"Number of validation files: {len(valid_samples)}")
    print(f"Number of test files: {len(test_samples)}")

    background_noise = data_dir / "ours" / "noise"

    # Prepare the model
    model_settings = standard_microspeech_model_settings(3)

    # If training is not toy, the following parameters are recommended:
    # num_epochs = 10
    # num_batches = 2
    # backprop_into_embedding = True

    print("---Training model---")
    _, model, _ = transfer_learn(
        target="hey_kiwibot",
        train_files=train_samples,
        val_files=valid_samples,
        unknown_files=train_negatives,
        num_epochs=num_epochs,
        num_batches=num_batches,
        batch_size=64,
        primary_lr=0.001,
        backprop_into_embedding=backprop_into_embedding,
        embedding_lr=0.0005,
        model_settings=model_settings,
        base_model_path=model_dir,
        base_model_output="dense_2",
        UNKNOWN_PERCENTAGE=unknown_percentage,
        bg_datadir=background_noise,
        csvlog_dest=str(model_dir.parent / "training_log.csv"),
        continue_training=False,
    )

    model.save(model_dir.parent / "finetuned_model")

    test_spectrograms = np.array([file2spec(model_settings, f) for f in test_samples])
    # fetch softmax predictions from the finetuned model:
    # (class 0: other, class 1: target)
    predictions = model.predict(test_spectrograms)
    categorical_predictions = np.argmax(predictions, axis=1)
    # which predictions match the target class?
    accuracy = (
        categorical_predictions[categorical_predictions == 1].shape[0]
        / predictions.shape[0]
    )
    print(f"Test accuracy on testset positive: {accuracy:0.2f}")

    test_spectrograms = np.array(
        [file2spec(model_settings, f) for f in test_negatives[:1000]]
    )
    predictions = model.predict(test_spectrograms)
    categorical_predictions = np.argmax(predictions, axis=1)
    accuracy = (
        categorical_predictions[categorical_predictions == 0].shape[0]
        / predictions.shape[0]
    )
    print(f"Test accuracy on testset negative: {accuracy:0.2f}")


if __name__ == "__main__":
    run(main)
