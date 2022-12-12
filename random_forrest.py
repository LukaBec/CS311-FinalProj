import argparse, os, sys
from math import remainder
from typing import Any, Dict, Sequence, Tuple, Union
from random import sample
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold #, metrics

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]


class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode]):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.attr = attr
        self.branches = branches

    def predict(self, x: pd.Series):
        """Return predicted labeled for array-like example x"""
        # TODO: Implement prediction based on value of self.attr in x
        return (self.branches[x[self.attr]]).predict(x)

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Test Feature", self.attr)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, self.attr, "=", val, "->", end=" ")
            subtree.display(indent + 1)


class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        """Create leaf node in decision tree

        Args:
            label: Label for this node
        """
        self.label = label

    def predict(self, x):
        """Return predicted labeled for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)


def entropy(k_counts):
    """
    info_gain helper func
    """
    p = k_counts[k_counts!=0] / np.sum(k_counts)
    return -np.sum(p * np.log2(p))

def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X,y by attr"""
    # TODO: Implement information gain metric for selecting attributes
    counts = X.groupby([attr, y]).size()
    gain = entropy(counts.groupby(level=1).sum())
    for k, k_counts in counts.groupby(level=0):
        gain -=(entropy(k_counts)*np.sum(k_counts))/X.shape[0]
    return gain


def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.Series,
) -> DecisionNode:
    """Recursively learn the decision tree

    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attrs (Sequence[str]): Possible attributes to split examples
        y_parent (pd.Series): array-like example labels for parents (parent target values)

    Returns:
        DecisionNode: Learned decision tree node
    """
    # TODO: Implement recursive tree construction based on pseudo code in class
    # and the assignment

    if X.empty:
        return DecisionLeaf(y_parent.mode().iloc[0]) #need to return the parent node within the series
    elif np.all(y == y.iloc[0]):
        return DecisionLeaf(y.iloc[0])
    elif len(attrs) == 0:
        return DecisionLeaf(y.mode().iloc[0])
    else:
        max = -1
        branches = {}
        for attr in attrs:
            gain = information_gain(X,y,attr)
            if max < gain:
                max = gain
                current = attr
        for k, x_k in X.groupby(current):
            branches[k] = learn_decision_tree(x_k, y.loc[x_k.index],[a for a in attrs if a!= current], y)
        return DecisionBranch(current, branches)
    """
    function LEARN-DECISION-TREE(examples, attributes, parent_examples):
        if examples is empty then return PLURALITY-VALUE(parent_examples)
        else if all examples have the same classification then return the classification
        else if attributes is empty then return PLURALITY-VALUE(examples)
        else
            A←argmaxa∈attributes INFORMATION-GAIN(a, examples)
            tree←a new decision tree with root testing attribute A
            for each value v of A do
            exs ← {e: e∈examples and e.A = v}
            subtree←LEARN-DECISION-TREE(exs, attributes-A, examples)
            add a branch to tree with label (A = v) and subtree subtree
        return tree 
    """
    return None


def fit(X: pd.DataFrame, y: pd.Series) -> DecisionBranch:
    """Return train decision tree on examples, X, with labels, y"""
    # You can change the implementation of this function, but do not modify the signature
    return learn_decision_tree(X, y, X.columns, y)


def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree"""

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.
    return X.apply(lambda row: tree.predict(row), axis=1)


def load_adult(feature_file: str, label_file: str):

    # Load the feature file
    examples = pd.read_table(
        feature_file,
        dtype={
            "age": int,
            "workclass": "category",
            "education": "category",
            "marital-status": "category",
            "occupation": "category",
            "relationship": "category",
            "race": "category",
            "sex": "category",
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": "category",
        },
    )
    labels = pd.read_table(label_file).squeeze().rename("label")

    # TODO: Select columns and choose a discretization for any continuos columns. Our decision tree algorithm
    # only supports discretized features and so any continuous columns (those not already "category") will need
    # to be discretized.

    # For example the following discretizes "hours-per-week" into "part-time" [0,40) hours and
    # "full-time" 40+ hours. Then returns a data table with just "education" and "hours-per-week" features.

    examples["hours-per-week"] = pd.cut(
        examples["hours-per-week"],
        bins=[0, 40, sys.maxsize],
        right=False,
        labels=["part-time", "full-time"],
    )

    examples["age"] = pd.cut(
        examples["age"],
        bins=[0, 35, 54, sys.maxsize],
        right=False,
        labels=["young", "middle", "old"],
    )

    examples["capital-gain"] = pd.cut(
        examples["capital-gain"],
        bins=[0, 1, sys.maxsize],
        right=False,
        labels=["non-invester", "investor"],
    )

    examples["capital-loss"] = pd.cut(
        examples["capital-loss"],
        bins=[0, 1, sys.maxsize],
        right=False,
        labels=["non-invester", "investor"],
    )

    return examples[["age", "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]], labels


# You should not need to modify anything below here


def load_examples(
    feature_file: str, label_file: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load example features and labels. Additional arguments are passed to
    the pandas.read_table function.

    Args:
        feature_file (str): Delimited file of categorical features
        label_file (str): Single column binary labels. Column name will be renamed to "label".

    Returns:
        Tuple[pd.DataFrame,pd.Series]: Tuple of features and labels
    """
    return (
        pd.read_table(feature_file, dtype="category", **kwargs),
        pd.read_table(label_file, **kwargs).squeeze().rename("label"),
    )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


class random_forest:
    def __init__(self, num_trees: int, num_attrs: int, X: pd.DataFrame, y: pd.Series):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.num_trees = num_trees
        self.num_attrs = num_attrs
        self.X = X
        self.y = y
        #Do we need to initialize our forest as a variable to keep track of it?
    
    def add_trees(self):
        """
        Create a forest of trees by randomly picking the attributes and rows for each tree
        """
        forest = []
        for _ in range(self.num_trees):
            rand_cols = sample(list(self.X.columns), self.num_attrs)
            #rand_rows = self.X[sample(list(self.X.columns), num_attrs)]
            #rand_rows = ?? - pandas.sampling for rows
            tree = fit(train_data[rand_cols], self.y) #rand_rows replaces X
            forest.append(tree)
        return forest

    def forest_predict(self, forest: list, X: pd.DataFrame):
        """
        Takes in list of trees and calculates prediction, then takes argmax across all predictions
        """
        results = []
        percentage = pd.DataFrame(columns=self.y.unique())
        final = [] # argmax of prediction percentage
        #create expected value for each tree
        for tree in forest:
            prediction = predict(tree, X)
            results.append(prediction)
        #create percentage dataframe for percentage of trees with expected values of 0 and 1
        for row in range(len(X)):
            count = [0,0]
            for pred in results: 
                count[int(pred[row])]+=1
            for col in range(len(percentage.columns)):
                percentage[col] = count[col] / np.sum(count)
            print(percentage)
        #Taking argmax of each row and creating final prediction for each row.
        # for _, row in percentage.iterrows():
        #     final.append(np.argmax(row))
        # return results

        
    
    def eval(self, y_pred: object, y_true: pd.Series) -> float:
        return metrics.accuracy_score(y_true, y_pred)

def load_leukemia(feature_file: str, label_file: str, **kwargs):
    """
    load our data for the forest
    Similar to load_example, taking in the the labels, skipping the first line and the last label, being leukimia & splitting the column to be our y

    """
    data = pd.read_table(feature_file, sep=",", dtype="category").drop(["row"], axis=1)
    return (data.drop(["leukemia"], axis=1), data["leukemia"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test decision tree learner")
    parser.add_argument(
        "-p",
        "--prefix",
        default="small1",
        help="Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, hepatitis, adult.",
    )
    parser.add_argument(
        "-k",
        "--k_splits",
        default=10,
        type=int,
        help="Number of splits for stratified k-fold testing",
    )


    args = parser.parse_args()

    if args.prefix != "adult":
        # Derive input files names for test sets
        train_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}_train_data.txt"
        )
        train_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}_train_label.txt"
        )
        # test_data_file = os.path.join(
        #     os.path.dirname(__file__), "data", f"{args.prefix}.test_data.txt"
        # )
        # test_labels_file = os.path.join(
        #     os.path.dirname(__file__), "data", f"{args.prefix}.test_label.txt"
        # )

        # Load training data and learn decision tree
        train_data, train_labels = load_leukemia(train_data_file, train_labels_file)
        forest = random_forest(5, 3, train_data, train_labels)
        trees = forest.add_trees()
        prediction = forest.forest_predict(trees, train_data)
        # accuracy = forest.eval(prediction, train_labels)
        # print(accuracy)
        # for index, row in train_data.iterrows():
        #     print(row)

        """
        train_data, train_labels = load_examples(train_data_file, train_labels_file)
        tree = fit(train_data, train_labels)
        tree.display()

        # Load test data and predict labels with previously learned tree
        test_data, test_labels = load_examples(test_data_file, test_labels_file)
        
        pred_labels = predict(tree, test_data)

        # Compute and print accuracy metrics
        predict_metrics = compute_metrics(test_labels, pred_labels)
        for met, val in predict_metrics.items():
            print(
                met.capitalize(),
                ": ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
        """
    else:
        # We use a slightly different procedure with "adult". Instead of using a fixed split, we split
        # the data k-ways (preserving the ratio of output classes) and test each split with a Decision
        # Tree trained on the other k-1 splits.
        data_file = os.path.join(os.path.dirname(__file__), "data", "adult.data.txt")
        labels_file = os.path.join(os.path.dirname(__file__), "data", "adult.label.txt")
        data, labels = load_adult(data_file, labels_file)

        scores = []

        kfold = StratifiedKFold(n_splits=args.k_splits)
        for train_index, test_index in kfold.split(data, labels):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            tree = fit(X_train, y_train)
            y_pred = predict(tree, X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

            tree.display()

        print(
            f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(scores)} ({np.std(scores)})"
        )
