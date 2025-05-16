"""
%load_ext autoreload
%autoreload 2
"""
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)

from dataclasses import dataclass
from koinapy import Koina
from memoized_koinapy_wrapper.main import get_annotation_encoder_and_decoder
from memoized_koinapy_wrapper.sequence_ops import *
from tqdm import tqdm

from cachemir.main import MemoizedOutput
from cachemir.main import get_index_and_stats
from memoized_koinapy_wrapper.main import get_annotations

import numba

from memoized_koinapy_wrapper.main import KoinaWrapper
from memoized_koinapy_wrapper.main import get_annotations


real_inputs = pd.read_csv("/tmp/real_inputs.csv")
real_inputs = real_inputs[
    ["peptide_sequences", "precursor_charges", "collision_energies"]
]
real_inputs["peptide_sequences"] = preprocess_sequences_for_prosit_timstof_2023(
    real_inputs.peptide_sequences
)

# model = Koina("Prosit_2023_intensity_timsTOF", "192.168.1.73:8500", ssl=False)
# predictions = model.predict(real_inputs)

encode_annotations, decode_annotations = get_annotation_encoder_and_decoder()

koina = KoinaWrapper(cache_path="/home/matteo/tmp/test11")
predictions = koina.predict(inputs_df=real_inputs.iloc[:1000])


predictions[["intensities", "annotation_idx"]]


@numba.njit
def numbers_are_consecutive_but_possibly_repeated(xx):
    x_prev = xx[0]
    for x in xx:
        diff = x - x_prev
        if diff != 0 and diff != 1:
            return False
        x_prev = x
    return True


def iter_results(**inputs):
    predictions = koina.predict(**inputs)
    assert numbers_are_consecutive_but_possibly_repeated(predictions.index.to_numpy())
    predictions["annotation_idx"] = predictions.annotation.map(encode_annotations)

    numbers_are_consecutive_but_possibly_repeated(predictions.index.to_numpy())
    for _input, gd in predictions.groupby(list(koina.input_columns))[
        ["intensities", "annotation_idx"]
    ]:
        yield MemoizedOutput(
            input=_input,
            stats=(),
            data=gd.reset_index(drop=True),
        )


index_and_stats, raw_data = get_index_and_stats(
    path="/home/matteo/tmp/test13",
    inputs_df=real_inputs,
    results_iter=iter_results,
    input_types=dict(
        peptide_sequences=str,
        precursor_charges=int,
        collision_energies=float,
    ),
    stats_types={},
    verbose=True,
)


predictions = koina.predict(real_inputs)
for K in tqdm(range(predictions.index[-1] + 1)):
    idx, cnt = index_and_stats.iloc[K]
    assert np.all(
        predictions.loc[K].intensities.to_numpy()
        == raw_data.intensities.iloc[idx : idx + cnt].to_numpy()
    )
