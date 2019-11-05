"""Simple decision tree implementation"""


class Node(object):
    def __init__(self, data, level=0, indices=None, terminal=False, parent=None):
        self._data = data
        self._level = level
        self.terminal = terminal

        indices = set(indices) if indices is not None else set(range(len(data)))

        self._indices_distributed = set()
        self._indices_remaining = indices

        if level and not parent:
            raise ValueError(f"Non-root node (level {level}) must have a parent")

        self._parent = parent
        self._children = []

    def __str__(self):
        root = '' if self.level else ' (root)'
        leaf = '(leaf)' if self.terminal else f'with {len(self.children)} children'

        return f"Level {self.level} tree node{root}; {'' if self.resolved else 'not '}resolved {leaf}"

    @property
    def data(self):
        return self._data

    @property
    def level(self):
        return self._level

    @property
    def indices_distributed(self):
        return self._indices_distributed

    @property
    def indices_remaining(self):
        return self._indices_remaining

    @property
    def indices(self):
        return self._indices_distributed | self._indices_remaining  # set union

    @property
    def resolved(self):
        return True if (self.terminal or not self._indices_remaining) else False

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def _add_child(self, child):
        if not isinstance(child, type(self)):
            raise TypeError(f"Child should be of type {type(self)}")

        chis = child.indices

        if not chis.issubset(self._indices_remaining):
            raise ValueError("Child indices are not a subset of the parent indices remaining to be distributed)")

        self._indices_distributed |= chis   # set union
        self._indices_remaining -= chis     # set difference
        self._children.append(child)

    def add_new_child(self, indices):
        child = self.__class__(self._data, level=self.level+1, indices=indices, parent=self)
        self._add_child(child)

    def add_final_child(self):
        self.add_new_child(self.indices_remaining)

