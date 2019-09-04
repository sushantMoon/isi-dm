from utils import is_numeric, info_gain, entropy
from utils import accuracy, f1score, class_counts
import numpy as np


class decision_question:
    def __init__(self, column, value, feature_names):
        self.column = column
        self.critical_value = value
        self.feature_names = feature_names

    def match(self, row):
        if is_numeric(self.critical_value):
            return row[self.column] >= self.critical_value
        else:
            return row[self.column] == self.critical_value

    def partition(self, df):
        true_df = None
        false_df = None
        index = None
        if is_numeric(self.critical_value):
            index = df[:, self.column] >= self.critical_value
        else:
            index = df[:, self.column] == self.critical_value
        true_df = df[index]
        false_df = df[~index]
        return true_df, false_df

    def __repr__(self):
        condition = "=="
        if is_numeric(self.critical_value):
            condition = ">="
        return "Split on {} {} {}".format(
            self.feature_names[self.column],
            condition,
            self.critical_value
        )


class decision_node:
    def __init__(self, decision_question, true_branch, false_branch):
        self.decision_question = decision_question
        self.true_branch = true_branch
        self.false_branch = false_branch


class leaf:
    def __init__(self, df):
        counts = np.column_stack(class_counts(df))
        self.prediction = {
            row[0]: row[1] for row in counts
        }


class decision_tree:
    def __init__(self, data, feature_names, max_depth=32, min_sample_split=2):
        self.data = data
        self.feature_names = feature_names
        self.depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None

    def train(self):
        self.root = self._build_tree(self.data, self.depth)

    def find_best_split(self, data):
        best_gain = 0
        best_decision_question = None
        n_features = data.shape[1] - 1
        # as the last feature is the target feature
        current_entrophy = entropy(data)
        for col in range(n_features):
            values = np.unique(data[:, col])
            for val in values:
                question = decision_question(col, val, self.feature_names)
                true_rows, false_rows = question.partition(data)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gain = info_gain(true_rows, false_rows, current_entrophy)
                if gain >= best_gain:
                    best_gain, best_decision_question = gain, question
        return best_gain, best_decision_question

    def _build_tree(self, data, depth):
        if data.shape[0] < self.min_sample_split:
            return leaf(data)
        gain, question = self.find_best_split(data)
        if gain == 0 or depth == 0:
            return leaf(data)
        true_rows, false_rows = question.partition(data)
        true_branch = self._build_tree(true_rows, depth-1)
        false_branch = self._build_tree(false_rows, depth-1)
        return decision_node(question, true_branch, false_branch)

    def predict(self, row):
        prediction = self._classify(self.root, row)
        return prediction

    def _classify(self, node, data):
        if isinstance(node, leaf):
            maxcount = 0
            maxlabel = None
            for k, v in node.prediction.items():
                if v >= maxcount:
                    maxcount = v
                    maxlabel = k
            return maxlabel
        if node.decision_question.match(data):
            return self._classify(node.true_branch, data)
        else:
            return self._classify(node.false_branch, data)

    def test(self, data, labels):
        predictions = np.zeros(data.shape[0])
        for i, row in enumerate(data):
            predictions[i] = self.predict(row)

        acc = accuracy(predictions.astype(int), labels.astype(int))
        f1 = f1score(
            predictions.astype(int),
            labels.astype(int),
            len(self.feature_names)
        )

        return acc, f1

    def print_decision_tree(self):
        queue = []
        queue.append(self.root)
        queue.append('|')
        while len(queue) > 0:
            flag = False
            if type(queue[0]) != decision_node:
                msg = queue.pop(0)
                if msg != '|':
                    print(msg)
                else:
                    print("-"*10)
            else:
                dn = queue.pop(0)
                print(dn.decision_question.__repr__())
                if type(dn.true_branch) == decision_node:
                    queue.append(dn.true_branch)
                    flag = True
                else:
                    queue.append("Leaf Node on true branch")
                if type(dn.false_branch) == decision_node:
                    queue.append(dn.false_branch)
                    flag = True
                else:
                    queue.append("Leaf Node on false branch")
                if flag:
                    queue.append('|')
