import functools
import tempfile
import shutil

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Iterator

import koinapy
import numpy.typing as npt
import numpy as np
import pandas as pd
import tqdm

from cachemir.main import MemoizedOutput
from cachemir.main import get_index_and_stats
from cachemir.main import SimpleLMDB


from memoized_koinapy_wrapper.checks import (
    numbers_are_consecutive_but_possibly_repeated,
)


@functools.cache
def get_annotations() -> pd.DataFrame:
    return pd.read_csv(
        files("memoized_koinapy_wrapper.data").joinpath("annotations.csv")
    )


@functools.cache
def get_test_call_data() -> pd.DataFrame:
    return pd.read_csv(
        files("memoized_koinapy_wrapper.data").joinpath("test_call_data.csv")
    )


@functools.cache
def get_annotation_encoder_and_decoder() -> tuple[dict, pd.DataFrame]:
    X = get_annotations()
    decode_annotations = X.drop(columns="index")
    encode_annotations = dict(
        zip(map(lambda a: a.encode(), X["annotation"]), X["index"])
    )
    # decode_annotations = dict(
    #     zip(
    #         decode_annotations.index,
    #         decode_annotations.itertuples(name=None, index=False),
    #     )
    # )
    decode_annotations = {
        col: decode_annotations[col].to_numpy() for col in ["type", "ordinal", "charge"]
    }
    return encode_annotations, decode_annotations


(
    default_annotations_encoder,
    default_annotations_decoder,
) = get_annotation_encoder_and_decoder()


def to_numpy(xx):
    if isinstance(xx, pd.Series):
        return xx.to_numpy()
    return xx


@dataclass
class SimpleKoinapyWrapper:
    """Simple wrapper around koinapy.

    This model was downloaded from ZENODO: https://zenodo.org/records/8211811
    PAPER : https://doi.org/10.1101/2023.07.17.549401
    """

    def __init__(
        self,
        db: SimpleLMDB,
        koinapy_kwargs: dict = dict(
            model_name="Prosit_2023_intensity_timsTOF",
            server_url="192.168.1.73:8500",
            ssl=False,
        ),
        annotations_encoder: dict = default_annotations_encoder,
        annotations_decoder: dict = default_annotations_decoder,
        input_types: dict[str, type] = dict(
            peptide_sequences=str,
            precursor_charges=int,
            collision_energies=float,
        ),
        columns_to_save: list[str] | None = ["intensities", "annotation"],
        meta: dict[str, str | float | int] = dict(
            sofware="koinapy", model="Prosit_2023_intensity_timsTOF"
        ),
    ):
        self.db = db
        self.koinapy_kwargs = koinapy_kwargs
        self.annotations_encoder = annotations_encoder
        self.annotations_decoder = annotations_decoder
        self.input_types = input_types
        self.columns_to_save = columns_to_save
        self.meta = meta

    @functools.cached_property  # be lazy
    def model(self):
        return koinapy.Koina(**self.koinapy_kwargs)

    def predict(self, inputs_df):
        return self.model.predict(inputs_df)

    def iter_eval(self, inputs_df: pd.DataFrame):
        columns_to_save = (
            inputs_df.columns if self.columns_to_save is None else self.columns_to_save
        )
        inputs_df = inputs_df[list(self.input_types)]
        predictions = self.predict(inputs_df)
        predictions.annotation = predictions.annotation.map(self.annotations_encoder)
        yield from predictions.groupby(list(self.input_types), sort=False)[
            columns_to_save
        ]

    def iter(self, inputs_df: pd.DataFrame):
        yield from self.db.iter_IO(
            iter_eval=self.iter_eval, inputs_df=inputs_df, meta=self.meta
        )

    def predict_compact(self, inputs_df: pd.DataFrame) -> list[dict[str, npt.NDArray]]:
        results = []
        for _, outputs in self.iter(inputs_df):
            annotations = outputs.pop("annotation")
            for col, decoding in self.annotations_decoder.items():
                outputs[col] = decoding[annotations]
            results.append(outputs)
        return results


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


#### ALL BELOW FOR COMPATIBILITY: USE CODE ABOVE


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
        if self.cache_path is not None:
            self.cache_path = Path(self.cache_path)

    @functools.cached_property  # not calling server when all cached
    def model(self):
        return koinapy.Koina(
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
    ) -> Iterator[MemoizedOutput]:
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


def test_koinapy_wrapper():
    temp_dir = Path(tempfile.mkdtemp(prefix="koina_wrapper_test_", dir="/tmp"))

    try:
        test_call_data = get_test_call_data()
        koina = KoinaWrapper(
            cache_path=temp_dir,
            server_url="192.168.1.73:8500",
            ssl=False,
        )
        index_and_stats, raw_data = koina(inputs_df=test_call_data, verbose=True)

        # Direct call also works.
        predictions = koina.predict(inputs_df=test_call_data)

        assert np.all(
            predictions.intensities.to_numpy() == raw_data.intensities.to_numpy()
        ), "Saved and retrieved intensities do not match."

        for K in tqdm.tqdm(range(predictions.index[-1] + 1)):
            idx, cnt = index_and_stats.iloc[K]
            assert np.all(
                predictions.loc[K].intensities.to_numpy()
                == raw_data.intensities.iloc[idx : idx + cnt].to_numpy()
            ), "Saved and retrieved spectrum did not match."

    finally:
        # Clean up the temporary cache directory
        shutil.rmtree(temp_dir, ignore_errors=True)
