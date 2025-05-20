"""
%load_ext autoreload
%autoreload 2
"""
from dataclasses import dataclass
from typing import Iterator
from tqdm import tqdm
import builtins
import pandas as pd
import numpy as np
import numpy.typing as npt
import functools
from memoized_koinapy_wrapper.sequence_ops import (
    preprocess_sequences_for_prosit_timstof_2023,
)
from tqdm import tqdm
from pathlib import Path

from memoized_koinapy_wrapper.main import (
    SimpleKoinapyWrapper,
    get_annotation_encoder_and_decoder,
)
from cachemir.main import SimpleLMDB

pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)


real_inputs = pd.read_csv("/tmp/real_inputs.csv")
real_inputs = real_inputs[
    ["peptide_sequences", "precursor_charges", "collision_energies"]
]
real_inputs["peptide_sequences"] = preprocess_sequences_for_prosit_timstof_2023(
    real_inputs.peptide_sequences
)


# Direct call also works.
inputs_df = real_inputs.iloc[:1000]

db = SimpleLMDB(path="/home/matteo/tmp/test25.lmbd")
with db.open("r") as txn:
    print(txn.get("__meta__", None))
    print(len(txn))

prosit_2023_timsTOF = SimpleKoinapyWrapper(db)

results = prosit_2023_timsTOF.predict_compact(real_inputs)


def test_SimpleKoinapyWrapper():
    temp_dir = Path(tempfile.mkdtemp(prefix="koina_wrapper_test_", dir="/tmp"))

    try:
        INPUTS = ["peptide_sequences", "precursor_charges", "collision_energies"]
        test_call_data = get_test_call_data()[INPUTS]
        db = SimpleLMDB(path=temp_dir)
        prosit_2023_timsTOF = SimpleKoinapyWrapper(db)
        predictions = prosit_2023_timsTOF.predict(inputs_df=test_call_data)
        predictions_lst = prosit_2023_timsTOF.predict_compact(inputs_df=test_call_data)

        for (inputs, real_call), cached_call in zip(
            predictions.groupby(INPUTS, sort=False), predictions_lst
        ):
            assert np.all(
                real_call["intensities"].to_numpy() == cached_call["intensities"]
            ), "Saved and retrieved spectrum did not match."

            for real_anontation, type, ordinal, charge in zip(
                real_call.annotation,
                cached_call["type"],
                cached_call["ordinal"],
                cached_call["charge"],
            ):
                real_anontation = real_anontation.decode()
                assert real_anontation == f"{type}{ordinal}+{charge}"

    finally:
        # Clean up the temporary cache directory
        shutil.rmtree(temp_dir, ignore_errors=True)
