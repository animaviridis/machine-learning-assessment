"""Simple decision tree implementation"""

import numpy as np
import pandas as pd
from math import inf
import logging

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


class Node(object):
    def __init__(self, data: pd.DataFrame, target_column=0, level: int=0,
                 indices=None, terminal=False, parent=None, which_child=0):

        self._level = level
        self._depth = 0
        self._terminal = terminal
        self._class = None

        if level and not parent:
            raise ValueError(f"Non-root node (level {level}) must have a parent")

        if not level and parent:
            raise ValueError("Root node should not have any parent")

        self._parent = parent

        self._target = target_column if not level else parent.target_column
        self._validate_data(data, self._target)
        self._data = data

        self._which_child = which_child

        self._children = []

        self._split_attribute = None
        self._split_thresholds = []

        if not level:
            indices = set(range(len(data)))
        else:
            if indices is None:
                raise ValueError("indices=None not allowed for a non-root node (use empty set if necessary)")
            else:
                indices = set(indices)

        self._indices_distributed = set()
        self._indices_remaining = indices

        self._entropy = self.calculate_entropy_labels((self.class_labels.to_list()))

    @staticmethod
    def _validate_data(data, target_column):
        target = data.keys().to_list()[target_column]
        labels = data[target]

        if labels.dtype not in [int, np.int64]:
            raise TypeError(f"Class labels (column '{target}') must be integers (got {labels.dtype})")

    def __str__(self):
        if self.level:
            attr, th = self.get_creation_stamp()
            root = f'for {th[0]} < {attr} <= {th[1]}, trace: {self.trace()}'
        else:
            root = 'root'

        leaf = '(leaf)' if self._terminal else f'with {len(self.children)} children'

        split = ""
        if self._split_attribute:
            split = f"; split at attribute '{self._split_attribute}' with thresholds: {self._split_thresholds[1:-1]}"

        sub = f'; subtree depth: {self._depth}' if len(self._children) else ''

        return f"Level {self.level} tree node ({root}); {'' if self.resolved else 'not '}resolved {leaf}{sub}{split}"

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
    def depth(self):
        return self._depth

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
        return True if (self._terminal or not self._indices_remaining) else False

    @property
    def split_thresholds(self):
        return self._split_thresholds

    def split_thresholds_for_child(self, which_child):
        return self._split_thresholds[which_child:which_child+2]

    def get_creation_stamp(self):
        return self.parent.split_attribute, self.parent.split_thresholds_for_child(self._which_child)

    @property
    def split_attribute(self):
        return self._split_attribute

    @property
    def target_attribute(self):
        return self.data.keys().to_list()[self.target_column]

    @property
    def input_attributes(self):
        all_keys = self.data.keys().to_list()
        return all_keys[:self.target_column] + all_keys[self.target_column+1:]

    @property
    def target_column(self):
        return self._target

    @property
    def class_labels(self):
        return self.data[self.target_attribute]

    @property
    def n_classes(self):
        return len(set(self.class_labels))

    @property
    def n_points(self):
        return len(self.data)

    @property
    def uniform(self):
        return self.n_classes == 1

    @staticmethod
    def get_label_occurrences(labels: list):
        return [labels.count(c) for c in set(labels)]

    @staticmethod
    def get_prevalent_label(labels: list):
        return list(set(labels))[np.argmax(Node.get_label_occurrences(labels))]

    @staticmethod
    def calculate_entropy_probs(probs):
        """Calculate variable entropy from variable value probabilities (list of probabilities/value occurrences)"""

        probs_norm = np.array(probs)
        probs_norm = probs_norm / probs_norm.sum()

        return - (probs_norm * np.log2(probs_norm)).sum()

    @staticmethod
    def calculate_entropy_labels(labels):
        """Calculate variable entropy from variable values (list of labels)"""

        return Node.calculate_entropy_probs(Node.get_label_occurrences(labels))

    def entropy(self):
        """Calculate entropy of the node (considering occurrences of each class label)"""

        return self._entropy

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def get_child(self, obs):
        """Return the relevant child instance based on the range the attribute value falls into"""

        if isinstance(obs, pd.DataFrame):
            if len(obs) > 1:
                raise ValueError(f"Observation should be a single-line DataFrame (got {len(obs)})")
            return self._get_child_from_df(obs)
        elif isinstance(obs, (int, float)):
            return self._get_child_from_value(obs)
        else:
            raise TypeError(f"Observation should be either a single-line DataFrame instance or a number (int, float)")

    def _get_child_from_value(self, attr_value):
        """Pick the relevant child instance based on the attribute value (assume it is the split attribute)"""

        return self.children[np.searchsorted(self._split_thresholds, attr_value) - 1]

    def _get_child_from_df(self, observation):
        """Pick the relevant child instance based on a new DataFrame-like object (pick the attribute first)"""

        return self._get_child_from_value(observation[self._split_attribute].values.item())

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
        child = self.__class__(self._data, level=self.level+1, indices=indices, parent=self,
                               which_child=len(self.children))
        self._add_child(child)

    def add_final_child(self):
        self.add_new_child(self.indices_remaining)

    def _get_split_indices(self, attribute, thresholds):
        th = sorted(list(thresholds) + [-inf, inf])

        vals = self.data[attribute]

        all_indices = []

        for i in range(1, len(th)):
            indices = vals.index[(vals > th[i - 1]) & (vals <= th[i])]
            all_indices.append(indices)

        return th, all_indices

    def get_split_information_gain(self, attribute, thresholds):
        """Calculate expected information gain after splitting at given attribute with given thresholds"""

        _, all_indices = self._get_split_indices(attribute, thresholds)
        split_labels = [self.class_labels.loc[indices].to_list() for indices in all_indices]

        n_tot = len(self.data)
        remainder = sum([(len(labels)/n_tot * self.calculate_entropy_labels(labels)) for labels in split_labels])

        return self.entropy() - remainder

    def split_at(self, attribute, thresholds):
        """Split node on a continuous attribute."""

        if attribute == self.target_attribute:
            raise ValueError(f"Cannot split on the target attribute ('{attribute}')")

        if self.resolved:
            logger.warning("Splitting an already resolved node - existing children will be removed")
            self.undo_split()

        th, all_indices = self._get_split_indices(attribute, thresholds)

        for i, indices in enumerate(all_indices):
            if not len(indices):
                logger.warning(f"No observations in value range ({th[i]}, {th[i+1]}] for attribute '{attribute}'")
            self.add_new_child(indices)

        if not self.resolved:
            logger.warning(f"Could not perform full split on attribute {attribute} - possibly missing values")
            self.add_final_child()

        self._split_thresholds = th
        self._split_attribute = attribute

    def undo_split(self):
        logger.debug(f"Undoing split at node {self.trace()}")
        self._children = []
        self._indices_remaining = self._indices_distributed.copy()
        self._indices_distributed = set()

    def choose_split_threshold(self, attribute, n=10):
        vals = np.sort(np.array(self.data[attribute]))  # values for the attribute
        th_cand = 0.5 * (vals[1:] + vals[:-1])  # threshold candidates - consecutive mid-points
        th_cand = th_cand[::n]  # check every n-th

        gains = [self.get_split_information_gain(attribute, [th]) for th in th_cand]

        idx = np.argmax(gains)

        chosen_gain = gains[idx]
        chosen_threshold = th_cand[idx]
        logger.debug(f"For attribute '{attribute}', best gain is {chosen_gain:.2g} "
                     f"(at threshold {chosen_threshold:.3g})")

        return chosen_gain, chosen_threshold

    def choose_split_attribute(self, **kwargs):
        all_attributes = self.input_attributes
        all_gains = len(all_attributes) * [0]
        all_thresholds = all_gains[:]

        logger.debug(f"Choosing split attribute for {self}")
        for i, attribute in enumerate(all_attributes):
            all_gains[i], all_thresholds[i] = self.choose_split_threshold(attribute, **kwargs)

        idx = np.argmax(all_gains)
        chosen_attribute = all_attributes[idx]
        logger.debug(f"Chosen attribute: {chosen_attribute} (expected gain: {all_gains[idx]:.3g})")

        return chosen_attribute, [all_thresholds[idx]]

    def split(self, **kwargs):
        s = self.choose_split_attribute(**kwargs)
        logger.info(f"Splitting at attribute '{s[0]}' with threshold: {s[1][0]:.2g}")
        self.split_at(*s)

    def terminate(self):
        self._terminal = True
        self._class = self.get_prevalent_label(self.class_labels.to_list())

    def learn(self, max_depth=5, **kwargs):
        if max_depth < 0:
            raise ValueError(f"Invalid maximal depth ({max_depth})")

        if max_depth == 0:
            logger.info(f"Reached the maximal depth (at {self.trace()}) - no further splitting")
            self.terminate()
            return 1

        if self.uniform:
            logger.info(f"Node {self.trace()} is an uniform node - no further splitting")
            self.terminate()
            return 1

        if self._terminal:
            logger.info(f"Splitting a node previously marked as terminal: {self.trace()}")
            self._terminal = False

        logger.info(f"Performing split of node {self.trace()}")
        self.split(**kwargs)

        logger.debug(f"Learning children of node {self.trace()}")
        depths = []
        for child in self.children:
            depths.append(child.learn(max_depth=max_depth-1, **kwargs))

        self._depth = max(depths)
        return self._depth + 1

    def print_terminal_labels(self):
        if len(self.children):
            for child in self.children:
                child.print_terminal_labels()

        else:
            print(f"Level {self.level} node, {self.trace()}: class {self._class} ({self.class_labels.to_list()})")

    def trace(self):
        if self.parent:
            return self.parent.trace() + [self._which_child]

        else:
            return []

    def prune(self, min_points=2):
        if len(self.children):
            if any(child.n_points < min_points for child in self.children):
                logger.info(f"Pruning at node {self.trace()}")
                self.undo_split()
                self.terminate()

            else:
                for child in self.children:
                    child.prune(min_points=min_points)

