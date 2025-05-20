"""
%load_ext autoreload
%autoreload 2
"""
import pandas as pd

from cachemir.main import SimpleLMDB
from memoized_koinapy_wrapper.sequence_ops import (
    preprocess_sequences_for_prosit_timstof_2023,
)

from memoized_koinapy_wrapper.main import SimpleKoinapyWrapper
from pathlib import Path
from tqdm import tqdm

pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)


real_inputs = pd.read_csv("/tmp/real_inputs.csv")
real_inputs = real_inputs[
    ["peptide_sequences", "precursor_charges", "collision_energies"]
]
real_inputs["peptide_sequences"] = preprocess_sequences_for_prosit_timstof_2023(
    real_inputs.peptide_sequences
)

db = SimpleLMDB(path="/home/matteo/tmp/test25.lmbd")
with db.open("r") as txn:
    print(txn.get("__meta__", None))
    print(len(txn))

prosit_2023_timsTOF = SimpleKoinapyWrapper(db)
results = prosit_2023_timsTOF.predict_compact(real_inputs)
results[0]

# direct call
results_direct = prosit_2023_timsTOF.predict(real_inputs)
