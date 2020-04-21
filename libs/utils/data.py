import numpy as np
import pandas as pd
from typing import Union, List, Callable, Tuple
from scipy import stats

def aggregate(select: pd.Series, values: pd.Series, steps: Union[int, List], func: Callable[[List], float] = np.mean):
    """
    Aggregate values in a pd.Series using a given function `func`
    """
    bounds = []
    results = []
    if isinstance(steps, int):
        n_step = steps
        m = select.min()
        M = select.max()
        step = (M - m) / n_step
        steps = np.arange(m, M+step, step)
    for a, b in zip(steps, steps[1:]):
        data = values.loc[(select > a) & (select <= b)]
        if not len(data):
            continue
        bounds.append(a)
        results.append(func(data))
    return bounds, results

def mean_confidence_interval(data: List, confidence: float = 0.95):
    """
    Return the mean of `data` along with its confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def jitter(arr, amount=0.01):
    """
    Add random noise in an array (useful for adding jitter when plotting discrete values)
    """
    stdev = amount*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev
