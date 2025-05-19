"""
%load_ext autoreload
%autoreload 2
"""
import pandas as pd
import numpy as np

from memoized_koinapy_wrapper.sequence_ops import (
    preprocess_sequences_for_prosit_timstof_2023,
)
from tqdm import tqdm

from memoized_koinapy_wrapper.main import KoinaWrapper

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


preds = koina.predict(inputs_df=inputs_df, verbose=True)
index_and_stats, raw_data = koina.get_index_and_stats(inputs_df=inputs_df, verbose=True)


# Direct call also works.
predictions = koina.predict(inputs_df=real_inputs)


assert np.all(
    predictions.intensities.to_numpy() == raw_data.intensities.to_numpy()
), "Saved and retrieved intensities do not match."

for K in tqdm(range(predictions.index[-1] + 1)):
    idx, cnt = index_and_stats.iloc[K]
    assert np.all(
        predictions.loc[K].intensities.to_numpy()
        == raw_data.intensities.iloc[idx : idx + cnt].to_numpy()
    ), "Saved and retrieved spectrum did not match."
