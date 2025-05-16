#!/usr/bin/env python3
import functools
import pandas as pd

from importlib.resources import files


@functools.cache
def get_annotations() -> pd.DataFrame:
    return pd.from_csv(
        files("memoized_koinapy_wrapper.data").joinpath("annotations.csv")
    )
