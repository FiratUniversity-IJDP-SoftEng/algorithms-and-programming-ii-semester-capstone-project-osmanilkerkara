import unittest
import numpy as np
from algorithm import DecisionTree, Node

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        """Set up a small dataset for testing"""
        self.X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]])
        self.y = np.array([0, 0, 1, 1])
        self.dt = DecisionTree(max_depth=2, min_samples_split=2)

    def test_gini_impurity(self):
        """Test Gini impurity calculation"""
        dt = DecisionTree()
        y = np.array([0, 0, 1, 1])
        gini = dt._gini_impurity(y)
        expected = 1 - (0.5**2 + 0.5**2)  # Two classes, equal distribution
        self.assertAlmostEqual(gini, expected, places=4)

    def test_best_split(self):
        """Test finding the best split"""
        feature, threshold, gini = self.dt._best_split(self.X, self.y)
        self.assertIsNotNone(feature, "Should find a valid feature")
        self.assertIsNotNone(threshold, "Should find a valid threshold")
        self.assertGreaterEqual(gini, 0, "Gini should be non-negative")

    def test_tree_growth(self):
        """Test if tree grows correctly"""
        self.dt.fit(self.X, self.y)
        self.assertIsNotNone(self.dt.root, "Root node should exist")
        self.assertTrue(isinstance(self.dt.root, Node), "Root should be a Node")
        self.assertLessEqual(len(self.dt.get_steps()), 5, "Steps should respect max_depth")

    def test_leaf_node(self):
        """Test leaf node creation for small dataset"""
        small_X = self.X[:2]
        small_y = self.y[:2]
        node = self.dt._grow_tree(small_X, small_y, depth=3)  # Exceeds max_depth
        self.assertIsNotNone(node.value, "Node should be a leaf")
        self.assertEqual(node.value, 0, "Leaf value should be majority class")

    def test_min_samples_split(self):
        """Test min_samples_split stopping criterion"""
        small_X = self.X[:1]
        small_y = self.y[:1]
        node = self.dt._grow_tree(small_X, small_y, depth=0)
        self.assertIsNotNone(node.value, "Node should be a leaf due to min_samples_split")
        self.assertEqual(node.value, 0, "Leaf value should be class 0")

if __name__ == '__main__':
    unittest.main()
