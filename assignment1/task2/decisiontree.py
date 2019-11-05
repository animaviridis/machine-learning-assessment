"""Simple decision tree implementation"""


class Node(object):
    def __init__(self, data, indices=None, parent=None):
        self._data = data

        indices = set(indices) if indices is not None else set(range(len(data)))

        self._indices_distributed = set()
        self._indices_remaining = indices

        self._parent = parent
        self._children = set()

    @property
    def data(self):
        return self._data

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
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def add_child(self, child):
        if not isinstance(child, type(self)):
            raise TypeError(f"Child should be of type {type(self)}")

        chis = child.indices

        if not chis.issubset(self._indices_remaining):
            raise ValueError("Child indices are not a subset of the parent indices remaining to be distributed)")

        self._indices_distributed |= chis   # set union
        self._indices_remaining -= chis     # set difference
        self._children.add(child)

    def new_child(self, indices):
        child = self.__class__(self._data, indices, parent=self)
        self.add_child(child)

