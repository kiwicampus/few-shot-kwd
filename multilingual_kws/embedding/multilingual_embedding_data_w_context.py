import csv
import functools
import glob
import multiprocessing
import os
import pathlib
import pickle
import shutil
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sox
import textgrid


def extract_one_second(duration_s: float, start_s: float, end_s: float):
    """
    return one second around the midpoint between start_s and end_s
    """
    if duration_s < 1:
        return (0, duration_s)
    center_s = start_s + ((end_s - start_s) / 2.0)
    new_start_s = center_s - 0.5
    new_end_s = center_s + 0.5
    if new_end_s > duration_s:
        new_end_s = duration_s
        new_start_s = duration_s - 1.0
    if new_start_s < 0:
        new_start_s = 0
        new_end_s = np.minimum(duration_s, new_start_s + 1.0)
    return (new_start_s, new_end_s)


def extract_shot_from_mp3(
    mp3name_no_ext,
    start_s,
    end_s,
    dest_dir,
    include_context,
    cv_clipsdir=pathlib.Path(
        "/home/mark/tinyspeech_harvard/common_voice/cv-corpus-6.1-2020-12-11/en/clips"
    ),
):
    """
    Extracts a shot from an MP3 file based on the provided start and end times.

    Parameters:
    - mp3name_no_ext: The name of the MP3 file without the extension.
    - start_s: The start time of the shot in seconds.
    - end_s: The end time of the shot in seconds.
    - dest_dir: The directory where the extracted shot will be saved.
    - include_context: A boolean indicating whether to include context in the shot.
    - cv_clipsdir: The directory where the MP3 file is located.

    Returns:
    None
    """
    mp3path = cv_clipsdir / (mp3name_no_ext + ".mp3")
    if not os.path.exists(mp3path):
        raise ValueError("could not find", mp3path)

    duration = sox.file_info.duration(mp3path)
    if end_s - start_s < 1 and not include_context:
        pad_amt_s = (1.0 - (end_s - start_s)) / 2.0
    else:
        # either (a) utterance is longer than 1s, trim instead of pad
        # or (b) include 1s of context
        start_s, end_s = extract_one_second(duration, start_s, end_s)
        pad_amt_s = 0

    if not os.path.isdir(dest_dir):
        raise ValueError(dest_dir, "does not exist")
    dest = dest_dir / (mp3name_no_ext + ".wav")
    # words can appear multiple times in a sentence: above should have filtered these
    if os.path.exists(dest):
        raise ValueError("already exists:", dest)

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.trim(start_s, end_s)
    # use smaller fadein/fadeout since we are capturing just the word
    # TODO(mmaz) is this appropriately sized?
    transformer.fade(fade_in_len=0.025, fade_out_len=0.025)
    if pad_amt_s > 0:
        transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
    transformer.build(str(mp3path), str(dest))


source_data = Path("/home/mark/tinyspeech_harvard/multilang_embedding/")

with open(source_data / "train_files.txt", "r") as fh:
    train_files = fh.read().splitlines()
with open(source_data / "val_files.txt", "r") as fh:
    val_files = fh.read().splitlines()
with open(source_data / "commands.txt", "r") as fh:
    commands = fh.read().splitlines()
print("SOURCE DATA")
print(len(train_files), train_files[0])
print(len(val_files), val_files[0])
print(len(commands))
print("-----")

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")


# build a nested map of common_voice_id to start and end times
# lang_isocode -> word -> commonvoice_id -> (start_s, end_s)
timing_dict = {}
for lang_isocode in ["ca", "de", "en", "es", "fa", "fr", "it", "nl", "rw"]:
    print(lang_isocode)
    timing_dict[lang_isocode] = {}
    for word_csv in glob.glob(str(frequent_words / lang_isocode / "timings" / "*.csv")):
        word_csv = Path(word_csv)
        word = word_csv.stem
        d = {}

        with open(word_csv, "r") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            for row in reader:
                common_voice_id = row[0]
                start_s = row[1]
                end_s = row[2]

                if common_voice_id in d:
                    continue
                d[common_voice_id] = (start_s, end_s)
        timing_dict[lang_isocode][word] = d

# how many extracted words were taken from common voice clips that have multiple occurences of the word
#    frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")
#    multi_extraction = 0
#
#    for ix, t in enumerate(train_files):
#        if ix % int(len(train_files) / 5) == 0:  # 20%
#            print(ix)
#        t = Path(t)
#        wav_noext = t.stem
#
#        if wav_noext.count("_") > 3:
#            multi_extraction += 1
#            continue
#
#    print(multi_extraction, multi_Apr 28, 2021extraction / len(train_files))
#    # around 3%

# build sister dataset with context surrounding extractions
dest_base = Path("/media/mark/hyperion/frequent_words_w_context")


def extract_context(training_sample):
    """
    Extracts the context from a training sample.

    Parameters:
    - training_sample: The path to the training sample.

    Returns:
    None
    """
    t = Path(training_sample)
    lang_isocode = t.parts[5]
    word = t.parts[7]
    wav = t.parts[8]
    wav_noext = t.stem

    if wav_noext.count("_") > 3:
        return

    start_s, end_s = timing_dict[lang_isocode][word][wav_noext]
    start_s = float(start_s)
    end_s = float(end_s)

    if lang_isocode == "es":
        cv_clipsdir = Path(
            "/media/mark/hyperion/common_voice/cv-corpus-5.1-2020-06-22/es/clips"
        )
    else:
        cv_clipsdir = Path(
            f"/media/mark/hyperion/common_voice/cv-corpus-6.1-2020-12-11/{lang_isocode}/clips"
        )

    dest_dir = dest_base / lang_isocode / "clips" / word
    os.makedirs(dest_dir, exist_ok=True)

    dest_file = dest_dir / wav
    if os.path.exists(dest_file):
        return  # already generated from a previous run

    source_mp3 = cv_clipsdir / (wav_noext + ".mp3")
    if not os.path.exists(source_mp3):
        print("warning: source mp3 not found", source_mp3)
        return

    word_extraction.extract_shot_from_mp3(
        mp3name_no_ext=wav_noext,
        start_s=start_s,
        end_s=end_s,
        dest_dir=dest_dir,
        include_context=True,
        cv_clipsdir=cv_clipsdir,
    )


pool = multiprocessing.Pool()
num_processed = 0
for _ in pool.imap_unordered(extract_context, train_files + val_files, chunksize=4000):
    num_processed += 1
print(num_processed)
