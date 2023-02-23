from typing import Sequence
import numpy as np


EPS = np.finfo(float).eps


def pooled_stdev(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate the pooled standard deviation.

    Parameters
    ----------
    x : array-like
        The first array of values.
    y : array-like
        The second array of values.

    Returns
    -------
    s : float
        The pooled standard deviation.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Compute the pooled standard deviation
    n = x.size + y.size - 2
    s = np.sqrt(((x.size - 1) / n) * x.var(ddof=1) + ((y.size - 1) / n) * y.var(ddof=1))

    return s


def cohen_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Cohen's d.

    Parameters
    ----------
    x : array-like
        The first array of values.
    y : array-like
        The second array of values.

    Returns
    -------
    d : float
        Cohen's d.

    Examples
    --------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [6, 7, 8, 9, 10]
    >>> cohen_d(x, y)
    1.0
    """

    x = np.asarray(x)
    y = np.asarray(y)

    d = (x.mean() - y.mean()) / (pooled_stdev(x, y) + EPS)
    return d


def norm_diff_stdev(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate the normalized difference in standard deviation.

    Parameters
    ----------
    x : array-like
        The first array of values.
    y : array-like
        The second array of values.

    Returns
    -------
    d : float
        The normalized difference in standard deviation.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    d = (x.std(ddof=1) - y.std(ddof=1)) / (pooled_stdev(x, y) + EPS)
    return d
