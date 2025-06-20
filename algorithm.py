import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gini=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.steps = []

    def fit(self, X, y):
        self.steps = []
        self.root = self._grow_tree(X, y, depth=0)

    def _gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = Counter(y)
        return 1 - sum((count/m)**2 for count in counts.values())

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None, None

        parent_gini = self._gini_impurity(y)
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx

                if sum(left_idx) < self.min_samples_split or sum(right_idx) < self.min_samples_split:
                    continue

                left_gini = self._gini_impurity(y[left_idx])
                right_gini = self._gini_impurity(y[right_idx])
                weighted_gini = (sum(left_idx) * left_gini + sum(right_idx) * right_gini) / m

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            value = Counter(y).most_common(1)[0][0]
            leaf = Node(value=value, gini=self._gini_impurity(y))
            self.steps.append({
                'action': 'Create leaf',
                'node': f'Depth {depth}',
                'gini': leaf.gini
            })
            return leaf

        feature, threshold, gini = self._best_split(X, y)
        if feature is None:
            value = Counter(y).most_common(1)[0][0]
            leaf = Node(value=value, gini=self._gini_impurity(y))
            self.steps.append({
                'action': 'Create leaf (no split)',
                'node': f'Depth {depth}',
                'gini': leaf.gini
            })
            return leaf

        node = Node(feature=feature, threshold=threshold, gini=gini)
        self.steps.append({
            'action': 'Split node',
            'node': f'Depth {depth}',
            'gini': gini,
            'split': {'feature': feature+1, 'threshold': threshold}
        })

        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        node.left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return node

    def get_steps(self):
        return self.steps