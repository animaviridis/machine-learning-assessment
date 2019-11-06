"""Simple decision tree implementation"""

import pandas as pd
from math import inf
import logging
import coloredlogs


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class Node(object):
    def __init__(self, data: pd.DataFrame, level: int=0, indices=None, terminal=False, parent=None):
        self._data = data
        self._level = level
        self.terminal = terminal

        if level and not parent:
            raise ValueError(f"Non-root node (level {level}) must have a parent")

        if not level and parent:
            raise ValueError("Root node should not have any parent")

        self._parent = parent
        self._children = []

        if not level:
            indices = set(range(len(data)))
        else:
            if indices is None:
                raise ValueError("indices=None not allowed for a non-root node (use empty set if necessary)")
            else:
                indices = set(indices)

        self._indices_distributed = set()
        self._indices_remaining = indices

    def __str__(self):
        root = '' if self.level else ' (root)'
        leaf = '(leaf)' if self.terminal else f'with {len(self.children)} children'

        return f"Level {self.level} tree node{root}; {'' if self.resolved else 'not '}resolved {leaf}"

    @property
    def full_data(self):
        return self._data

    @property
    def data(self):
        return self._data.loc[self.indices]

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

    def split(self, attribute, thresholds):
        """Split node on a continuous attribute."""

        th = sorted(list(thresholds) + [-inf, inf])

        vals = self.data[attribute]

        for i in range(1, len(th)):
            indices = vals.index[(vals >= th[i-1]) & (vals < th[i])]

            if not len(indices):
                logger.warning(f"No observations in value range [{th[i-1]}, {th[i]}) for attribute '{attribute}'")

            self.add_new_child(indices)

        if not self.resolved:
            logger.warning(f"Couldn't perform full split on attribute {attribute} - possibly missing values")
            self.add_final_child()

