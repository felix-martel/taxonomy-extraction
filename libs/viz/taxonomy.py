from IPython.core.display import display, HTML, Javascript


def clean_name(name):
    return name.replace("<", "").replace(">", "")

def get_content(name, content=""):
    extra = " colored" if content else ""
    inner = f"<div class='inner'>{content}</div>" if content else ""
    outer = f"<div class='box{extra}'>{name}{inner}</div>"
    return outer

def void_func(*args, **kwargs):
    return ""

def get_html_tree(taxo, hover=void_func, **kwargs):
    lines = []
    def print_tree(current_node, childattr='children', nameattr='name', display_func=print,
                   indent='', last='updown', max_depth=None, halting_func=None, print_params={}):
        """
        TODO: comment arguments
        nameattr:
        - if string, then the name of a node will be the value of node.{nameattr}
        - if callable, then the name of a node will be nameattr(node)
        - else, use str(node)
        """
        print_children = not ( max_depth == 0 or (halting_func is not None and halting_func(current_node)))
        if max_depth is not None:
            max_depth -= 1
        if callable(nameattr):
            name = lambda node: nameattr(node)
        elif hasattr(current_node, nameattr):
            name = lambda node: getattr(node, nameattr)
        else:
            name = lambda node: str(node)

        children = lambda node: getattr(node, childattr)
        nb_children = lambda node: sum(nb_children(child) for child in children(node)) + 1
        size_branch = {child: nb_children(child) for child in children(current_node)}
        SPACE = "&nbsp;"

        if print_children:
            """ Creation of balanced lists for "up" branch and "down" branch. """
            up = sorted(children(current_node), key=lambda node: nb_children(node))
            down = []
            while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
                down.append(up.pop())

            """ Printing of "up" branch. """
            for child in up:     
                next_last = 'up' if up.index(child) is 0 else ''
                next_indent = '{0}{1}{2}'.format(indent, SPACE if 'up' in last else '│', SPACE * len(name(current_node)))
                print_tree(child, childattr, nameattr, display_func, next_indent, next_last, max_depth, halting_func, print_params)

        """ Printing of current node. """
        if last == 'up': start_shape = '┌'
        elif last == 'down': start_shape = '└'
        elif last == 'updown': start_shape = SPACE
        else: start_shape = '├'

        if print_children:
            if up: end_shape = '┤'
            elif down: end_shape = '┐'
            else: end_shape = ''
        else:
            end_shape = ""

        lines.append('{0}{1}{2}{3}'.format(indent, start_shape, get_content(name(current_node), hover(current_node)), end_shape))

        if print_children:
            """ Printing of "down" branch. """
            for child in down:
                next_last = 'down' if down.index(child) is len(down) - 1 else ''
                next_indent = '{0}{1}{2}'.format(indent, SPACE if 'down' in last else '│', SPACE * len(name(current_node)))
                print_tree(child, childattr, nameattr, display_func, next_indent, next_last, max_depth, halting_func, print_params)
            # end for
        #end if
        
    print_tree(taxo.root, **kwargs)
    return lines

def get_style(stylesheet=None):
    if stylesheet is None:
        stylesheet = "libs/viz/taxonomy.css"
    with open(stylesheet, "r") as f:
        style = f.read()
    return style


def print_html(taxo, filename=None, stylesheet=None, **kwargs):
    """
    User hover=func node->str to customize the content of the toolbox
    """
    style = get_style(stylesheet)
    lines = get_html_tree(taxo, **kwargs)
    html = style + "\n" + "\n".join(f"<div class='line'>{l}</div>" for l in lines)
    
    if filename is not None:
        with open(filename, "w") as f:
            f.write(html)
    
    display(HTML(html))
    return html