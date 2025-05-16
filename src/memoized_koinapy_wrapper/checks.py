import numba
import numpy.typing as npt


@numba.njit
def numbers_are_consecutive_but_possibly_repeated(xx: npt.NDArray) -> bool:
    """Check if provided array of ints is non-decreasing and each jump is at most equal to 1."""
    x_prev = xx[0]
    for x in xx:
        diff = x - x_prev
        if diff != 0 and diff != 1:
            return False
        x_prev = x
    return True
