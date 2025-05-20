"""
%load_ext autoreload
%autoreload 2
"""
from cachemir.main import SimpleIndex
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
from cachemir.serialization import name_to_type, type_to_name
from cachemir.main import Index, input_to_bytes, ITERTUPLES

import msgpack
import msgpack_numpy as m


pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)


real_inputs = pd.read_csv("/tmp/real_inputs.csv")
real_inputs = real_inputs[
    ["peptide_sequences", "precursor_charges", "collision_energies"]
]
real_inputs["peptide_sequences"] = preprocess_sequences_for_prosit_timstof_2023(
    real_inputs.peptide_sequences
)

# real_inputs.iloc[:1000].to_csv("/tmp/test_call_data.csv")

koina = KoinaWrapper(
    cache_path="/home/matteo/tmp/test22",
    server_url="192.168.1.73:8500",
    ssl=False,
)
index_and_stats, raw_data = koina.get_index_and_stats(inputs_df=real_inputs)


inputs_df = real_inputs.loc[[0, 0, 0, 1, 1]].copy()
inputs_df = inputs_df.reset_index(drop=True)
index_and_stats, raw_data = koina.get_index_and_stats(inputs_df=inputs_df, verbose=True)


# Direct call also works.
inputs_df = real_inputs.iloc[:1000]
predictions = koina.predict(inputs_df=inputs_df)
predict = koina.predict

assert np.all(
    predictions.intensities.to_numpy() == raw_data.intensities.to_numpy()
), "Saved and retrieved intensities do not match."

for K in tqdm(range(predictions.index[-1] + 1)):
    idx, cnt = index_and_stats.iloc[K]
    assert np.all(
        predictions.loc[K].intensities.to_numpy()
        == raw_data.intensities.iloc[idx : idx + cnt].to_numpy()
    ), "Saved and retrieved spectrum did not match."


def preprocess_prosit_results(predictions: pd.DataFrame, *args, **kwargs):
    annotation_encoder, annotation_decoder = get_annotation_encoder_and_decoder()
    predictions.annotation = predictions.annotation.map(annotation_encoder)
    return predictions


def df2dct(df: pd.DataFrame) -> dict[str, npt.NDArray]:
    return {col: df[col].to_numpy() for col in df}


results_preprocessor = preprocess_prosit_results
predictions = results_preprocessor(predictions)
results = predictions


encode = functools.partial(msgpack.packb, default=m.encode)
decode = functools.partial(msgpack.unpackb, object_hook=m.decode)






cache_path = "/home/matteo/tmp/test24.lmbd"
input_types = dict(
    peptide_sequences=str, precursor_charges=int, collision_energies=float
)
saved_columns = ["intensities", "annotation"]
verbose = True

grouping_cols = list(input_types)
cache_path = Path(cache_path)
index = Index(cache_path)

in2bytes = functools.partial(
    input_to_bytes,
    types=tuple(input_types.values()),
)

from dataclasses import dataclass


class LmdbMemoization:
    def __init__(
        self,
        cache_path: Path,
        input_types: dict[str, type],
    ):
        self.lmdb = Index(self.cache_path)
        self.input_types = input_types

    def write(self, results: pd.DataFrame):


def lmdb_memoize(cache_path, input_types: dict[str, type]):
    def outer_wrapper(foo):
        def inner_wrapper(*arg, **kwargs):
            return foo(*arg, **kwargs)



# with index.open("w") as txn:
#     input_types = txn.get(b"__input_types__", None)
#     if input_types is None:
#         txn.put(
#             b"__input_types__",
#             encode({c: type_to_name[t] for c,t in input_types.items()}),
#         )
#     grouped_data = predictions.groupby(grouping_cols, sort=False)[saved_columns]
#     if verbose:
#         grouped_data = tqdm(grouped_data, desc="Saving")
#     for group, data in grouped_data:
#         txn.put(in2bytes(group), encode(df2dct(data)))


inputs_df = real_inputs
predict(inputs_df)
with index.open("w") as txn:
    print(type(txn))

with index.open("w") as txn:
    txn.put(
        b"__input_types__",
        encode(
            txn.get(
                b"__input_types__",
                {c: type_to_name[t] for c,t in input_types.items()}
            )
        )
    )

    for inputs in ITERTUPLES(inputs_df):
        txn.get(encode(inputs))


    grouped_data = results.groupby(grouping_cols, sort=False)[saved_columns]
    if verbose:
        grouped_data = tqdm(grouped_data, desc="Saving")
    for group, data in grouped_data:
        txn.put(in2bytes(group), encode(df2dct(data)))



calls = predictions[grouping_cols].drop_duplicates()


def get_input_types(txn, name_to_type=name_to_type):
    input_types = txn.get(b"__input_types__", None)
    assert (
        not input_types is None
    ), "Pass in `__input_types__` or make sure it is saved in lmdb."
    input_types = decode(input_types)
    return {c: type_to_name[t] for c, t in input_types.items()}


with index.open("w") as txn:
    if input_types is None:
        input_types = get_input_types(txn, name_to_type)
    in2bytes = functools.partial(
        input_to_bytes,
        types=tuple(input_types.values()),
    )
    for call in tqdm(ITERTUPLES(calls), total=len(calls)):
        res = txn.get(in2bytes(call), None)
        assert res is not None
        res = decode(res)
        yield call, res

x = msgpack.packb(df2dct(data), default=m.encode)
y = msgpack.unpackb(x, object_hook=m.decode)
