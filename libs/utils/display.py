# TODO: make it work when IPython is not available
try:
    from IPython.display import HTML, display
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    display = print
    def HTML(*args, **kwargs): pass



from itertools import chain


def print_html(*args, tag=None, **kwargs):
    if tag is not None:
        display(HTML(wrap(tag, *args, **kwargs)))
    else:
        display(HTML(*args, **kwargs))


def wrap(tag, *args, func=str, **kwargs):
    """
    Wrap content into HTML tags

    Apply `func` to all `args`, join the result, and wrap it into a 'tag' tag.

    Example:
    ```python
    >>> wrap('h1', 'My Title Here')
    '<h1>My Title Here</h1>'
    >>> wrap("ul", range(3), func=lambda x: wrap("li", x))
    '<ul>
    <li>0</li>
    <li>1</li>
    <li>2</li>
    </ul>'
    ```
    """
    x = "".join(func(arg, **kwargs) for arg in args)
    if HTML_AVAILABLE:
        return f"<{tag}>{x}</{tag}>"
    return x


def print_line(line, length):
    def print_item(x):
        if isinstance(x, float): x = "{:.2f}".format(x)
        return wrap("td", x)

    missing_length = length - len(line)
    fixed_length_line = chain(line, [""] * missing_length)
    return wrap("tr", *fixed_length_line, func=print_item)


def display_table(lines, cols=None):
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