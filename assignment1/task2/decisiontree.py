"""Simple decision tree implementation"""


class Node(object):
    def __init__(self, data, indices: set = None, parent=None):
        self._data = data

        if indices is not None:
            self._indices = indices
        else:
            self._indices = set(range(len(data)))

        self._parent = parent
        self._children = set()

    @property
    def data(self):
        return self._data

    @property
    def indices(self):
        return self._indices

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def add_child(self, child):
        if not isinstance(child, type(self)):
            raise TypeError(f"Child should be of type {type(self)}")

        if not child.indices.issubset(self.indices):
            raise ValueError("Child instance indices are not a subset of the parent indices)")

        self._children.add(child)

    def new_child(self, indices):
        child = self.__class__(self._data, indices, parent=self)
        self.add_child(child)

