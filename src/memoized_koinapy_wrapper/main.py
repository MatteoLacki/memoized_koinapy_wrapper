import functools

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable

import koinapy
import numpy.typing as npt
import pandas as pd
import tqdm

from cachemir.main import MemoizedOutput
from cachemir.main import get_index_and_stats

from memoized_koinapy_wrapper.checks import (
    numbers_are_consecutive_but_possibly_repeated,
)


@functools.cache
def get_annotations() -> pd.DataFrame:
    return pd.read_csv(
        files("memoized_koinapy_wrapper.data").joinpath("annotations.csv")
    )


def get_annotation_encoder_and_decoder() -> tuple[dict, pd.DataFrame]:
    X = get_annotations()
    decode_annotations = X.drop(columns="index")
    encode_annotations = dict(
        zip(map(lambda a: a.encode(), X["annotation"]), X["index"])
    )
    return encode_annotations, decode_annotations


@dataclass
class KoinaWrapper:
    """Wrapper around koinapy.

    This model was downloaded from ZENODO: https://zenodo.org/records/8211811
    PAPER : https://doi.org/10.1101/2023.07.17.549401
    """

    model_name: str = "Prosit_2023_intensity_timsTOF"
    server_url: str = "192.168.1.73:8500"
    ssl: bool = False
    cache_path: Path | str | None = None
    input_columns: tuple[str, str, str] = (
        "peptide_sequences",
        "precursor_charges",
        "collision_energies",
    )

    def __post_init__(self):
        self.cache_path = Path(self.cache_path)
        self.model = koinapy.Koina(
            self.model_name,
            server_url=self.server_url,
            ssl=self.ssl,
        )

    def predict(self, inputs_df: pd.DataFrame | None = None, **kwargs):
        """
        Call the model.
        """
        inputs_df = (
            pd.DataFrame({col: kwargs[col] for col in self.input_columns}, copy=False)
            if inputs_df is None
            else inputs_df[list(self.input_columns)]
        )
        return self.model.predict(inputs_df)

    def iter_predict_intensities(
        self,
        peptide_sequences: npt.NDArray | pd.Series,
        precursor_charges: npt.NDArray | pd.Series,
        collision_energies: npt.NDArray | pd.Series,
    ):
        predictions = self.predict(
            peptide_sequences=peptide_sequences,
            precursor_charges=precursor_charges,
            collision_energies=collision_energies,
        )
        assert numbers_are_consecutive_but_possibly_repeated(
            predictions.index.to_numpy()
        )
        encode_annotations, _ = get_annotation_encoder_and_decoder()
        predictions["annotation_idx"] = predictions.annotation.map(encode_annotations)

        for inputs, results_df in predictions.groupby(
            list(self.input_columns),
            sort=False,  # CRUCIAL!!! DO NOT DARE TO CHANGE!!!
        )[["intensities", "annotation_idx"]]:
            yield MemoizedOutput(
                input=inputs,
                stats=(),
                data=results_df.reset_index(drop=True),
            )

    def get_index_and_stats(
        self,
        inputs_df: pd.DataFrame | None = None,
        cache_path: Path | str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if cache_path is None:
            assert self.cache_path is not None
            cache_path = self.cache_path
        cache_path = Path(cache_path)

        inputs_df = (
            pd.DataFrame({col: kwargs[col] for col in self.input_columns}, copy=False)
            if inputs_df is None
            else inputs_df[list(self.input_columns)]
        )

        index_and_stats, raw_data = get_index_and_stats(
            path=cache_path,
            inputs_df=inputs_df,
            results_iter=self.iter_predict_intensities,
            input_types=dict(
                peptide_sequences=str,
                precursor_charges=int,
                collision_energies=float,
            ),
            stats_types={},
            verbose=verbose,
        )
        return index_and_stats, raw_data

    __call__ = get_index_and_stats
