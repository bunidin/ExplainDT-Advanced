from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import json
# source of randomness is not really important here,
# so I'll just use Python's default.
import random


# This file contains different "sections" for dealing with decision trees.
# Here's a high level description of the different elements in this file.
# 1.) Sklearn to JSON.
# We will learn decision trees with the sklearn library, and thus the resulting trees are instances
# of the `DecisionTreeClassifier` class.

# We will serialize a DecisionTreeClassifier into a JSON with a specific format.
# This is done recursively through the `traverse` function,
# which is wrapped in the `tree_to_dict` function, which returns a Python dictionary.
# Finally, the wrapper `serialize_tree_to_json` will write build the dictionary out of the tree and then
# serialize it into a target JSON file.

# NOTE:In order to use them on `main.py`, we will need to implement a deserializer that reads the JSON file
# and returns a tree in the representation used in `main.py`.
# TODO: Implement deserializer


# 2.) Generating random decision trees
# We learn a decision tree trained on a random dataset
# This will be mostly useful for running experiments.
# The main function here is `generate_random_dt`, which receives the target dimension, number of leaves, and number of training samples.
# it returns the dictionary representation. A wrapper function `random_dt_to_json` will serialize the resulting decision tree
# into a target JSON file.


#  1.) Sklearn -> Dict -> JSON

def argmax(narr):  # avoids loading numpy only for this method
    arr = narr[0]
    imx = 0
    mx = -1e9
    for i, v in enumerate(arr):
        if v >= mx:
            mx = v
            imx = i
    return imx


# recursive creation of the dictionary
def traverse(tree, root, ans, feature_names, class_names):
    left = tree.children_left[root]
    right = tree.children_right[root]

    leaf = (left == -1)

    ans[root] = {
        "id": root,
        "type": "leaf" if leaf else "internal",
    }

    if leaf:
        ans[root].update({
            "class": str(class_names[argmax(tree.value[root])]),
        })
    else:
        ans[root].update({
            "feature_name": str(feature_names[tree.feature[root]]),
            "feature_index": int(tree.feature[root]),
            "threshold": float(tree.threshold[root]),
            "id_left": int(left),
            "id_right": int(right)
        })
        traverse(tree, int(left), ans, feature_names, class_names)
        traverse(tree, int(right), ans, feature_names, class_names)


def tree_to_dict(feature_names, feature_types, class_names, dt, acc=None):
    ans = {
        "feature_names": list(map(str, feature_names)),
        "feature_types": list(map(str, feature_types)),
        "class_names": list(map(str, class_names)),
        "positive": str(class_names[0]),
        "nodes": {}
    }
    if acc is not None:
        ans["accuracy"] = acc
    traverse(dt.tree_, 0, ans["nodes"], feature_names, class_names)
    return ans


def serialize_dict_to_json(target_file, dt_dict):
    with open(target_file, 'w') as tf:
        tf.write(json.dumps(dt_dict, indent=2))


def serialize_tree_to_json(target_file, dt, feature_names, feature_types, class_names, acc=None):
    serialize_dict_to_json(target_file,
                           tree_to_dict(feature_names, feature_types, class_names, dt, acc))


# ----- GENERATING RANDOM DECISION TREES -------- #
def random_boolean_instance(dimension):
    return [random.randint(0, 1) for _ in range(dimension)]


def generate_random_dataset(n_samples, dimension, force_label=None):
    X = [random_boolean_instance(dimension) for sample in range(n_samples)]
    if force_label is None:
        y = random_boolean_instance(n_samples)
    else:
        y = [force_label for _ in range(n_samples)]
    return X, y


def generate_random_dt(dimension, n_leaves, n_training_samples):
    dt = DecisionTreeClassifier(max_leaf_nodes=n_leaves, random_state=0)
    X, y = generate_random_dataset(n_training_samples, dimension)
    dt.fit(X, y)

    ft_names = [f'ft{i}' for i in range(dimension)]
    ft_types = ['boolean' for _ in range(dimension)]
    class_names = ['positive', 'negative']
    dt_dict = tree_to_dict(ft_names, ft_types, class_names, dt)

    # check that numbr of actual leaves is not too different from specified
    n_actual_leaves = len(
        list(filter(lambda x: x['type'] == 'leaf', dt_dict['nodes'].values())))
    if n_actual_leaves < n_leaves // 2:
        print(f"n_actual_leaves = {n_actual_leaves}, n_leaves = {n_leaves}")
    return dt_dict


def random_dt_to_json(target_file, dimension, n_leaves, n_training_samples):
    random_tree_dict = generate_random_dt(dimension, n_leaves, n_training_samples)
    serialize_dict_to_json(target_file, random_tree_dict)
    
