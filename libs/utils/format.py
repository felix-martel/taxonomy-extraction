import math
import os


def millify(n):
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
