from IPython.display import HTML, display
from itertools import chain, zip_longest
def tag_line(line, tag): return [f"<{tag}>{x}</{tag}>" for x in line]
def bold_line(line): return tag_line(line, "b")
def italic_line(line): return tag_line(line, "i")

def join(left, right, columns=None):
    left = list(left)
    right = list(right)
    if not left and not right:
        return []
    n = len(left[0] if left else right[0])
    
    header = [
        bold_line(["Left"] + (n-1)*[""] + ["Right"])
    ]
    if columns is not None: 
        columns = columns + [""] * (n - len(columns))
        header.append(italic_line(2 * columns))
        
    joined_table = header + [[*l, *r] for l, r in zip_longest(left, right, fillvalue=("",)*n)]
    return joined_table

def create_section(a, tab):
    print_html(a, tag="b")
    display_table(tab)

def print_html(*args, tag=None, **kwargs):
    if tag is not None:
        display(HTML(wrap(tag, *args, **kwargs)))
    else:
        display(HTML(*args, **kwargs))
            
def wrap(tag, *args, func=str, **kwargs):
    x = "".join(func(arg, **kwargs) for arg in args)
    return f"<{tag}>{x}</{tag}>"

def print_line(line, length, fillchar=""):
    def print_item(x):
        if isinstance(x, float): x = "{:.2f}".format(x)
        return wrap("td", x)
    missing_length = length - len(line)
    fixed_length_line = chain(line, [""] * missing_length)
    return wrap("tr", *fixed_length_line, func=print_item)
    
def display_table(lines, cols=None):
    if not lines: return
    n_cols = max(map(len, lines))
    if cols is not None:
        n_cols = max(len(cols), n_cols)
        display(HTML(wrap("table", cols, *lines, func=print_line, length=n_cols)))
    else:
        display(HTML(wrap("table", *lines, func=print_line, length=n_cols)))
        
        
def to_html(lines, cols=None):
    n_cols = max(map(len, lines))
    if cols is not None:
        n_cols = max(len(cols), n_cols)
        return wrap("table", cols, *lines, func=print_line, length=n_cols)
    else:
        return wrap("table", *lines, func=print_line, length=n_cols)