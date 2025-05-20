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
    KoinaWrapper,
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

koina = KoinaWrapper(
    cache_path="/home/matteo/tmp/test22",
    server_url="192.168.1.73:8500",
    ssl=False,
)

predictions = koina.predict(inputs_df=inputs_df)


# rm -rf /home/matteo/tmp/test25.lmbd
cache_path = "/home/matteo/tmp/test25.lmbd"


def iter_evaluate_koina(
    inputs_df: pd.DataFrame,
) -> Iterator[tuple[tuple, pd.DataFrame]]:
    out_cols = ["intensities", "annotation"]
    predictions = koina.predict(inputs_df)
    annotation_encoder, annotation_decoder = get_annotation_encoder_and_decoder()
    predictions.annotation = predictions.annotation.map(annotation_encoder)
    yield from predictions.groupby(list(inputs_df), sort=False)[out_cols]


# need to update the pytest in koinapy_wrapper.

db = SimpleLMDB(cache_path)
with db.open("r") as txn:
    print(txn["__meta__"])

it = db.iter_IO(
    iter_eval=iter_evaluate_koina,
    inputs_df=inputs_df,
    meta=dict(sofware="koinapy", model="Prosit2023"),
)
xx = list(tqdm(it, total=len(inputs_df)))
