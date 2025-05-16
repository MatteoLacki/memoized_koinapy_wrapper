import functools
import numpy.typing as npt
import pandas as pd

from pathlib import Path
from typing import Iterable

from dataclasses import dataclass
from importlib.resources import files

import koinapy


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

    def __call__(self):
        pass

    def iter_predict_intensities(
        self,
        sequences: npt.NDArray,
        charges: npt.NDArray,
        collision_energies: npt.NDArray,
    ):
        pass
