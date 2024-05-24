from sklearn.tree import DecisionTreeClassifier
import dtrees.dtree as dtree
import pandas as pd

# Load the Mushroom dataset from a local file
dataset_filename = '/Users/bunyamindincer/Downloads/ExplainDT-Advanced/datasets/agaricus-lepiota.data'
columns = [
    'class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
    'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
    'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
    'stalk_surface_below_ring', 'stalk_color_above_ring',
    'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number',
    'ring_type', 'spore_print_color', 'population', 'habitat'
]
data = pd.read_csv(dataset_filename, names=columns)

# Handle missing values: Replace '?' with the most frequent value in each column
data.replace('?', pd.NA, inplace=True)
for column in data.columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Convert categorical features to binary (one-hot encoding)
data = pd.get_dummies(data, columns=columns[1:])

# Split the dataset into features and labels
X = data.drop('class', axis=1)
y = data['class'].map({'e': 1, 'p': 0})  # e for edible (1), p for poisonous (0)

# Define feature names and types
features = [(col.replace('_', ''), 'boolean') for col in X.columns]  # Remove underscores
ft_names = [f[0] for f in features]
ft_types = [f[1] for f in features]
class_names = ['poisonous', 'edible']

# Train a DecisionTreeClassifier
mushroom_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
mushroom_classifier.fit(X, y)

print('DecisionTreeClassifier has been trained!')

# Export the decision tree to JSON
export_filename = 'mushroom_dt.json'
dtree.serialize_tree_to_json(export_filename, mushroom_classifier, ft_names, ft_types, class_names)

print(f'Exported to: {export_filename}')
