import re

from typing import Callable
from typing import Iterable


def remove_n_terminal_acetylation(sequence: str) -> str:
    if sequence.startswith("[UNIMOD:1]"):
        sequence = sequence[len("[UNIMOD:1]") :]
    return sequence


def exchange_selenocysteine_for_cysteine(sequence: str) -> str:
    # This regex splits the input into parts outside and inside brackets
    parts = re.split(r"(\[.*?\])", sequence)
    # Replace 'U' with 'C' only in parts outside of brackets
    parts = [
        part.replace("U", "C") if not part.startswith("[") else part for part in parts
    ]
    return "".join(parts)


def apply_preprocessings_to_sequence(
    sequence: str,
    preprocessings: list[Callable[[str], str]],
) -> str:
    """Apply a sequence of preprocessing on a sequence."""
    for preprocessing in preprocessings:
        sequence = preprocessing(sequence)
    return sequence


def preprocess_sequences(
    sequences: Iterable[str],
    preprocessings: list[Callable[[str], str]],
) -> list[str]:
    return [
        apply_preprocessings_to_sequence(sequence, preprocessings)
        for sequence in sequences
    ]


def preprocess_sequences_for_prosit_timstof_2023(
    sequences: Iterable[str],
    preprocessings: list[Callable[[str], str]] = (
        remove_n_terminal_acetylation,
        exchange_selenocysteine_for_cysteine,
    ),
) -> list[str]:
    """Preprocess peptide sequences to be used with `koinapy`.

    By default we:
        1. remove n-terminal acetylation modification
        2. exchange selenocystein U for cystein C.

    BY NO MEANS IS THIS WHAT YOU MIGHT WANT. I WANT IT.

    Arguments:
        sequences (Iterable[str]): An iterable o sequences to preprocess.
    """
    return preprocess_sequences(sequences, preprocessings)
