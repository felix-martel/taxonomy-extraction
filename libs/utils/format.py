import math
import os
from collections import Counter
from typing import Union


def millify(n):
    """Pretty-print long numbers (e.g 10^4 becomes 10K, 10^7 becomes 10M, 10^10 becomes 10B and so on)"""
    prefixes = ['','K','M','B','T']
    i = max(0,min(len(prefixes)-1,int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    prec = 1 if i > 1 else 0
    return '{:.{prec}f}{}'.format(n / 10**(3 * i), prefixes[i], prec=prec)

def shorten_path(path):
    """Shorten path for display purpose"""
    full_path, file = os.path.split(path)
    others, parent = os.path.split(full_path)
    elements = [file]
    if parent:
        elements.append(parent)
    if others:
        elements.append("...")
    return "/".join(reversed(elements))

def format_counter(counter: Counter, max_items: Union[None, int] = 5, min_threshold: Union[None, float] = 0.1,
                   cum_sum: Union[None, float] = 0.95, max_line_length: int = 40, min_line_length: int = 6) -> str:
    """Pretty-print the most frequent items of a Counter"""
    n = sum(counter.values())
    l = max(map(len, counter))
    l = max(min_line_length, min(max_line_length, l))

    current_sum = 0
    SEP = "  "
    lines = [
        f"{'':{l}.{l}}{SEP}{'#':>6}{SEP}{'':>5}%",
        "-" * (l + 2 * len(SEP) + 6 + 5 + 1),
    ]
    n_items = 0
    for name, count in counter.most_common(max_items):
        if count / n < min_threshold or n_items > max_items:
            break
        line = f"{name:{l}.{l}}{SEP}{millify(count):<6}{SEP}{count/n:5.1f}%"
        lines.append(line)
        n_items += 1
        current_sum += count / n
        if current_sum > cum_sum:
            break
    return "\n".join(lines)
