from time import time

import tensorflow as tf
import typer
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)


def to_micro_spectrogram(model_settings, audio):
    """
    Converts audio data to a micro spectrogram.

    Args:
        model_settings (dict): A dictionary containing model settings.
        audio (Tensor): The audio data to convert.

    Returns:
        Tensor: The micro spectrogram output.

    """
    start = time()
    sample_rate = model_settings["sample_rate"]
    window_size_ms = (model_settings["window_size_samples"] * 1000) / sample_rate
    window_step_ms = (model_settings["window_stride_samples"] * 1000) / sample_rate
    int16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
    micro_frontend = frontend_op.audio_microfrontend(
        int16_input,
        sample_rate=sample_rate,
        window_size=window_size_ms,
        window_step=window_step_ms,
        num_channels=model_settings["fingerprint_width"],
        out_scale=1,
        out_type=tf.float32,
    )
    output = tf.multiply(micro_frontend, (10.0 / 256.0))
    finish = time()
    print(f"to_micro_spectrogram took {(finish - start)*1000} miliseconds")
    return output


def file2spec(model_settings, filepath):
    """there's a version of this that adds bg noise in AudioDataset"""
    start_file = time()
    audio_binary = tf.io.read_file(filepath)
    finish_file = time()
    start_decode = time()
    audio, _ = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=model_settings["desired_samples"],
    )
    audio = tf.squeeze(audio, axis=-1)
    finish_decode = time()
    print(f"loading file took {(finish_file - start_file)*1000} miliseconds")
    print(f"decode took {(finish_decode - start_decode)*1000} miliseconds")
    return to_micro_spectrogram(model_settings, audio)


def main(filepath: str):
    """
    Main function for this script.
    """
    model_settings = {
        "desired_samples": 16000,
        "window_size_samples": 480,
        "window_stride_samples": 320,
        "spectrogram_length": 49,
        "fingerprint_width": 40,
        "fingerprint_size": 1960,
        "label_count": 1,
        "sample_rate": 16000,
        "preprocess": "micro",
        "average_window_width": -1,
    }

    print(f"Filepath given: {filepath}")
    tensor_def = file2spec(model_settings, filepath)
    print(tensor_def.shape)


if __name__ == "__main__":
    typer.run(main)
