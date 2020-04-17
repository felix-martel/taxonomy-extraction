from warnings import warn

warn("'libs.pptree' is deprecated. Use 'libs.tree.pprint' instead", category=DeprecationWarning)

class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)


def print_tree(current_node, childattr='children', nameattr='name', 
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
    
    if print_children:
        """ Creation of balanced lists for "up" branch and "down" branch. """
        up = sorted(children(current_node), key=lambda node: nb_children(node))
        down = []
        while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
            down.append(up.pop())

        """ Printing of "up" branch. """
        for child in up:     
            next_last = 'up' if up.index(child) is 0 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', ' ' * len(name(current_node)))
            print_tree(child, childattr, nameattr, next_indent, next_last, max_depth, halting_func, print_params)

    """ Printing of current node. """
    if last == 'up': start_shape = '┌'
    elif last == 'down': start_shape = '└'
    elif last == 'updown': start_shape = ' '
    else: start_shape = '├'
    
    if print_children:
        if up: end_shape = '┤'
        elif down: end_shape = '┐'
        else: end_shape = ''
    else:
        end_shape = ""

    print('{0}{1}{2}{3}'.format(indent, start_shape, name(current_node), end_shape), **print_params)

    if print_children:
        """ Printing of "down" branch. """
        for child in down:
            next_last = 'down' if down.index(child) is len(down) - 1 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', ' ' * len(name(current_node)))
            print_tree(child, childattr, nameattr, next_indent, next_last, max_depth, halting_func, print_params)